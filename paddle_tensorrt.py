import cv2
from shapely.geometry import Polygon
import pyclipper
import re
import math
import numpy as np
import tensorrt as trt
import pycuda.driver as cuda
import copy
import threading
from logger import logger as log

# from ppocr.data import create_operators, transform

_infer_lock = threading.Lock()
_det_infer_lock = threading.Lock()


class DBPostProcess(object):
    """
    The post process for Differentiable Binarization (DB).
    """

    def __init__(self,
                 thresh=0.3,
                 box_thresh=0.7,
                 max_candidates=1000,
                 unclip_ratio=2.0,
                 use_dilation=False,
                 score_mode="fast",
                 box_type='quad',
                 **kwargs):
        self.thresh = thresh
        self.box_thresh = box_thresh
        self.max_candidates = max_candidates
        self.unclip_ratio = unclip_ratio
        self.min_size = 3
        self.score_mode = score_mode
        self.box_type = box_type
        assert score_mode in [
            "slow", "fast"
        ], "Score mode must be in [slow, fast] but got: {}".format(score_mode)

        self.dilation_kernel = None if not use_dilation else np.array(
            [[1, 1], [1, 1]])

    def polygons_from_bitmap(self, pred, _bitmap, dest_width, dest_height):
        '''
        _bitmap: single map with shape (1, H, W),
            whose values are binarized as {0, 1}
        '''

        bitmap = _bitmap
        height, width = bitmap.shape

        boxes = []
        scores = []

        contours, _ = cv2.findContours((bitmap * 255).astype(np.uint8),
                                       cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

        for contour in contours[:self.max_candidates]:
            epsilon = 0.002 * cv2.arcLength(contour, True)
            approx = cv2.approxPolyDP(contour, epsilon, True)
            points = approx.reshape((-1, 2))
            if points.shape[0] < 4:
                continue

            score = self.box_score_fast(pred, points.reshape(-1, 2))
            if self.box_thresh > score:
                continue

            if points.shape[0] > 2:
                box = self.unclip(points, self.unclip_ratio)
                if len(box) > 1:
                    continue
            else:
                continue
            box = box.reshape(-1, 2)

            _, sside = self.get_mini_boxes(box.reshape((-1, 1, 2)))
            if sside < self.min_size + 2:
                continue

            box = np.array(box)
            box[:, 0] = np.clip(
                np.round(box[:, 0] / width * dest_width), 0, dest_width)
            box[:, 1] = np.clip(
                np.round(box[:, 1] / height * dest_height), 0, dest_height)
            boxes.append(box.tolist())
            scores.append(score)
        return boxes, scores

    def boxes_from_bitmap(self, pred, _bitmap, dest_width, dest_height):
        '''
        _bitmap: single map with shape (1, H, W),
                whose values are binarized as {0, 1}
        '''

        bitmap = _bitmap
        height, width = bitmap.shape

        outs = cv2.findContours((bitmap * 255).astype(np.uint8), cv2.RETR_LIST,
                                cv2.CHAIN_APPROX_SIMPLE)
        if len(outs) == 3:
            img, contours, _ = outs[0], outs[1], outs[2]
        elif len(outs) == 2:
            contours, _ = outs[0], outs[1]

        num_contours = min(len(contours), self.max_candidates)

        boxes = []
        scores = []
        for index in range(num_contours):
            contour = contours[index]
            points, sside = self.get_mini_boxes(contour)
            if sside < self.min_size:
                continue
            points = np.array(points)
            if self.score_mode == "fast":
                score = self.box_score_fast(pred, points.reshape(-1, 2))
            else:
                score = self.box_score_slow(pred, contour)
            if self.box_thresh > score:
                continue

            box = self.unclip(points, self.unclip_ratio).reshape(-1, 1, 2)
            box, sside = self.get_mini_boxes(box)
            if sside < self.min_size + 2:
                continue
            box = np.array(box)

            box[:, 0] = np.clip(
                np.round(box[:, 0] / width * dest_width), 0, dest_width)
            box[:, 1] = np.clip(
                np.round(box[:, 1] / height * dest_height), 0, dest_height)
            boxes.append(box.astype("int32"))
            scores.append(score)
        return np.array(boxes, dtype="int32"), scores

    def unclip(self, box, unclip_ratio):
        poly = Polygon(box)
        distance = poly.area * unclip_ratio / poly.length
        offset = pyclipper.PyclipperOffset()
        offset.AddPath(box, pyclipper.JT_ROUND, pyclipper.ET_CLOSEDPOLYGON)
        expanded = np.array(offset.Execute(distance))
        return expanded

    def get_mini_boxes(self, contour):
        bounding_box = cv2.minAreaRect(contour)
        points = sorted(list(cv2.boxPoints(bounding_box)), key=lambda x: x[0])

        index_1, index_2, index_3, index_4 = 0, 1, 2, 3
        if points[1][1] > points[0][1]:
            index_1 = 0
            index_4 = 1
        else:
            index_1 = 1
            index_4 = 0
        if points[3][1] > points[2][1]:
            index_2 = 2
            index_3 = 3
        else:
            index_2 = 3
            index_3 = 2

        box = [
            points[index_1], points[index_2], points[index_3], points[index_4]
        ]
        return box, min(bounding_box[1])

    def box_score_fast(self, bitmap, _box):
        '''
        box_score_fast: use bbox mean score as the mean score
        '''
        h, w = bitmap.shape[:2]
        box = _box.copy()
        xmin = np.clip(np.floor(box[:, 0].min()).astype("int32"), 0, w - 1)
        xmax = np.clip(np.ceil(box[:, 0].max()).astype("int32"), 0, w - 1)
        ymin = np.clip(np.floor(box[:, 1].min()).astype("int32"), 0, h - 1)
        ymax = np.clip(np.ceil(box[:, 1].max()).astype("int32"), 0, h - 1)

        mask = np.zeros((ymax - ymin + 1, xmax - xmin + 1), dtype=np.uint8)
        box[:, 0] = box[:, 0] - xmin
        box[:, 1] = box[:, 1] - ymin
        cv2.fillPoly(mask, box.reshape(1, -1, 2).astype("int32"), 1)
        return cv2.mean(bitmap[ymin:ymax + 1, xmin:xmax + 1], mask)[0]

    def box_score_slow(self, bitmap, contour):
        '''
        box_score_slow: use polyon mean score as the mean score
        '''
        h, w = bitmap.shape[:2]
        contour = contour.copy()
        contour = np.reshape(contour, (-1, 2))

        xmin = np.clip(np.min(contour[:, 0]), 0, w - 1)
        xmax = np.clip(np.max(contour[:, 0]), 0, w - 1)
        ymin = np.clip(np.min(contour[:, 1]), 0, h - 1)
        ymax = np.clip(np.max(contour[:, 1]), 0, h - 1)

        mask = np.zeros((ymax - ymin + 1, xmax - xmin + 1), dtype=np.uint8)

        contour[:, 0] = contour[:, 0] - xmin
        contour[:, 1] = contour[:, 1] - ymin

        cv2.fillPoly(mask, contour.reshape(1, -1, 2).astype("int32"), 1)
        return cv2.mean(bitmap[ymin:ymax + 1, xmin:xmax + 1], mask)[0]

    def __call__(self, outs_dict, shape_list):
        pred = outs_dict['maps']
        # if isinstance(pred, paddle.Tensor):
        #     pred = pred.numpy()
        pred = pred[:, 0, :, :]
        segmentation = pred > self.thresh

        boxes_batch = []
        for batch_index in range(pred.shape[0]):
            src_h, src_w, ratio_h, ratio_w = shape_list[batch_index]
            if self.dilation_kernel is not None:
                mask = cv2.dilate(
                    np.array(segmentation[batch_index]).astype(np.uint8),
                    self.dilation_kernel)
            else:
                mask = segmentation[batch_index]
            if self.box_type == 'poly':
                boxes, scores = self.polygons_from_bitmap(pred[batch_index],
                                                          mask, src_w, src_h)
            elif self.box_type == 'quad':
                boxes, scores = self.boxes_from_bitmap(pred[batch_index], mask,
                                                       src_w, src_h)
            else:
                raise ValueError("box_type can only be one of ['quad', 'poly']")

            boxes_batch.append({'points': boxes})
        return boxes_batch


def resize_norm_img(img, max_wh_ratio, img_c=3, img_h=48, img_w=200):
    imgC, imgH, imgW = img_c, img_h, img_w

    assert imgC == img.shape[2]
    imgW = int((imgH * max_wh_ratio))

    h, w = img.shape[:2]
    ratio = w / float(h)
    # print("jiayanjun h:", h, "w ", w, " ratio:", ratio)
    if math.ceil(imgH * ratio) > imgW:
        resized_w = imgW
    else:
        resized_w = int(math.ceil(imgH * ratio))
    resized_image = cv2.resize(img, (resized_w, imgH))
    resized_image = resized_image.astype('float32')
    resized_image = resized_image.transpose((2, 0, 1)) / 255
    resized_image -= 0.5
    resized_image /= 0.5
    padding_im = np.zeros((imgC, imgH, imgW), dtype=np.float32)
    padding_im[:, :, 0:resized_w] = resized_image
    return padding_im


def cv2_letterbox_image(image, expected_size):
    ih, iw = image.shape[0:2]
    ew, eh = expected_size
    scale = min(eh / ih, ew / iw)
    nh = int(ih * scale)
    nw = int(iw * scale)
    image = cv2.resize(image, (nw, nh), interpolation=cv2.INTER_LINEAR)
    new_img = np.zeros((eh, ew, 3), dtype=np.uint8)

    top = (eh - nh) // 2
    bottom = nh + top
    left = (ew - nw) // 2
    right = nw + left
    new_img[top:bottom, left:right, :] = image
    # new_img = new_img / 255.
    # new_img = cv2.copyMakeBorder(image, top, bottom, left, right, cv2.BORDER_CONSTANT)
    return new_img


def resize_norm_img_det(img, img_c=3, img_h=96, img_w=192, im_mean=(0.485, 0.456, 0.406), im_std=(0.229, 0.224, 0.225)):
    imgC, imgH, imgW = img_c, img_h, img_w
    im_scale = 1 / 255.
    assert imgC == img.shape[2]
    ih, iw = img.shape[0:2]
    ew, eh = imgW, imgH
    scale = min(eh / ih, ew / iw)
    nh = int(ih * scale)
    nw = int(iw * scale)
    image = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_LINEAR)
    new_img = np.zeros((eh, ew, 3), dtype=np.uint8)

    top = 0
    bottom = nh
    left = 0
    right = nw
    new_img[top:bottom, left:right, :] = image
    im_mean = np.array(im_mean).reshape(1, 1, 3).astype('float32')
    im_std = np.array(im_std).reshape(1, 1, 3).astype('float32')
    # cv2.imshow("tt", new_img)
    # cv2.waitKey()
    resized_image = (new_img.astype('float32') * im_scale - im_mean) / im_std
    resized_image = resized_image.transpose((2, 0, 1))

    mid_w = iw * ew / nw
    mid_h = ih * eh / nh
    return resized_image, [mid_h, mid_w, scale, scale]


def get_rotate_crop_image(img, points):
    '''
    img_height, img_width = img.shape[0:2]
    left = int(np.min(points[:, 0]))
    right = int(np.max(points[:, 0]))
    top = int(np.min(points[:, 1]))
    bottom = int(np.max(points[:, 1]))
    img_crop = img[top:bottom, left:right, :].copy()
    points[:, 0] = points[:, 0] - left
    points[:, 1] = points[:, 1] - top
    '''
    assert len(points) == 4, \
        print("shape of points must be 4*2 -----------", points)
    img_crop_width = int(
        max(
            np.linalg.norm(points[0] - points[1]),
            np.linalg.norm(points[2] - points[3])))
    img_crop_height = int(
        max(
            np.linalg.norm(points[0] - points[3]),
            np.linalg.norm(points[1] - points[2])))
    pts_std = np.float32([[0, 0], [img_crop_width, 0],
                          [img_crop_width, img_crop_height],
                          [0, img_crop_height]])
    M = cv2.getPerspectiveTransform(points, pts_std)
    dst_img = cv2.warpPerspective(
        img,
        M, (img_crop_width, img_crop_height),
        borderMode=cv2.BORDER_REPLICATE,
        flags=cv2.INTER_CUBIC)
    dst_img_height, dst_img_width = dst_img.shape[0:2]
    if dst_img_height * 1.0 / dst_img_width >= 1.5:
        dst_img = np.rot90(dst_img)
    return dst_img


def clip_det_res(points, img_height, img_width):
    for pno in range(points.shape[0]):
        points[pno, 0] = int(min(max(points[pno, 0], 0), img_width - 1))
        points[pno, 1] = int(min(max(points[pno, 1], 0), img_height - 1))
    return points


def order_points_clockwise(pts):
    rect = np.zeros((4, 2), dtype="float32")
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]
    tmp = np.delete(pts, (np.argmin(s), np.argmax(s)), axis=0)
    diff = np.diff(np.array(tmp), axis=1)
    rect[1] = tmp[np.argmin(diff)]
    rect[3] = tmp[np.argmax(diff)]
    return rect


def filter_tag_det_res(dt_boxes, image_shape):
    img_height, img_width = image_shape[0:2]
    dt_boxes_new = []
    for box in dt_boxes:
        if type(box) is list:
            box = np.array(box)
        box = order_points_clockwise(box)
        box = clip_det_res(box, img_height, img_width)
        rect_width = int(np.linalg.norm(box[0] - box[1]))
        rect_height = int(np.linalg.norm(box[0] - box[3]))
        if rect_width <= 3 or rect_height <= 3:
            continue
        dt_boxes_new.append(box)
    dt_boxes = np.array(dt_boxes_new)
    return dt_boxes


def filter_tag_det_res_only_clip(dt_boxes, image_shape):
    img_height, img_width = image_shape[0:2]
    dt_boxes_new = []
    for box in dt_boxes:
        if type(box) is list:
            box = np.array(box)
        box = clip_det_res(box, img_height, img_width)
        dt_boxes_new.append(box)
    dt_boxes = np.array(dt_boxes_new)
    return dt_boxes


def sorted_boxes(dt_boxes):
    """
    Sort text boxes in order from top to bottom, left to right
    args:
        dt_boxes(array):detected text boxes with shape [4, 2]
    return:
        sorted boxes(array) with shape [4, 2]
    """
    num_boxes = dt_boxes.shape[0]
    sorted_boxes = sorted(dt_boxes, key=lambda x: (x[0][1], x[0][0]))
    _boxes = list(sorted_boxes)

    for i in range(num_boxes - 1):
        for j in range(i, -1, -1):
            if abs(_boxes[j + 1][0][1] - _boxes[j][0][1]) < 10 and \
                    (_boxes[j + 1][0][0] < _boxes[j][0][0]):
                tmp = _boxes[j]
                _boxes[j] = _boxes[j + 1]
                _boxes[j + 1] = tmp
            else:
                break
    return _boxes


def cv2_letterbox_image(image, expected_size):
    ih, iw = image.shape[0:2]
    ew, eh = expected_size
    scale = min(eh / ih, ew / iw)
    nh = int(ih * scale)
    nw = int(iw * scale)
    image = cv2.resize(image, (nw, nh), interpolation=cv2.INTER_LINEAR)
    new_img = np.zeros((eh, ew, 3), dtype=np.uint8)

    top = (eh - nh) // 2
    bottom = nh + top
    left = (ew - nw) // 2
    right = nw + left
    new_img[top:bottom, left:right, :] = image
    # new_img = new_img / 255.
    # new_img = cv2.copyMakeBorder(image, top, bottom, left, right, cv2.BORDER_CONSTANT)
    return new_img


class BaseRecLabelDecode(object):
    """ Convert between text-label and text-index """

    def __init__(self, character_dict_path=None, use_space_char=False):
        self.beg_str = "sos"
        self.end_str = "eos"
        self.reverse = False
        self.character_str = []

        if character_dict_path is None:
            self.character_str = "0123456789abcdefghijklmnopqrstuvwxyz"
            dict_character = list(self.character_str)
        else:
            with open(character_dict_path, "rb") as fin:
                lines = fin.readlines()
                for line in lines:
                    line = line.decode('utf-8').strip("\n").strip("\r\n")
                    self.character_str.append(line)
            if use_space_char:
                self.character_str.append(" ")
            dict_character = list(self.character_str)
            if 'arabic' in character_dict_path:
                self.reverse = True

        dict_character = self.add_special_char(dict_character)
        self.dict = {}
        for i, char in enumerate(dict_character):
            self.dict[char] = i
        self.character = dict_character

    def pred_reverse(self, pred):
        pred_re = []
        c_current = ''
        for c in pred:
            if not bool(re.search('[a-zA-Z0-9 :*./%+-]', c)):
                if c_current != '':
                    pred_re.append(c_current)
                pred_re.append(c)
                c_current = ''
            else:
                c_current += c
        if c_current != '':
            pred_re.append(c_current)

        return ''.join(pred_re[::-1])

    def add_special_char(self, dict_character):
        return dict_character

    def decode(self, text_index, text_prob=None, is_remove_duplicate=False):
        """ convert text-index into text-label. """
        result_list = []
        ignored_tokens = self.get_ignored_tokens()
        batch_size = len(text_index)
        for batch_idx in range(batch_size):
            selection = np.ones(len(text_index[batch_idx]), dtype=bool)
            if is_remove_duplicate:
                selection[1:] = text_index[batch_idx][1:] != text_index[
                                                                 batch_idx][:-1]
            for ignored_token in ignored_tokens:
                selection &= text_index[batch_idx] != ignored_token

            char_list = [
                self.character[text_id]
                for text_id in text_index[batch_idx][selection]
            ]
            if text_prob is not None:
                conf_list = text_prob[batch_idx][selection]
            else:
                conf_list = [1] * len(selection)
            if len(conf_list) == 0:
                conf_list = [0]

            text = ''.join(char_list)

            if self.reverse:  # for arabic rec
                text = self.pred_reverse(text)

            result_list.append((text, np.mean(conf_list).tolist()))
        return result_list

    def get_ignored_tokens(self):
        return [0]  # for ctc blank


class CTCLabelDecode(BaseRecLabelDecode):
    """ Convert between text-label and text-index """

    def __init__(self, character_dict_path=None, use_space_char=False,
                 **kwargs):
        super(CTCLabelDecode, self).__init__(character_dict_path,
                                             use_space_char)

    def __call__(self, preds, label=None, *args, **kwargs):
        if isinstance(preds, tuple) or isinstance(preds, list):
            preds = preds[-1]
        # if isinstance(preds, paddle.Tensor):
        #     preds = preds.numpy()
        preds_idx = preds.argmax(axis=2)
        preds_prob = preds.max(axis=2)
        text = self.decode(preds_idx, preds_prob, is_remove_duplicate=True)
        if label is None:
            return text
        label = self.decode(label)
        return text, label

    def add_special_char(self, dict_character):
        dict_character = ['blank'] + dict_character
        return dict_character


class TextDetTrtInfer(object):
    def __init__(self, engine_path="weights/onnx/text_det_x_3_x_x.engine", max_batch=6, max_w=960, max_h=960, gup_id=0):

        self.post_process = DBPostProcess(thresh=0.3,
                                          box_thresh=0.6,
                                          max_candidates=1000,
                                          unclip_ratio=1.5,
                                          use_dilation=False,
                                          score_mode="fast",
                                          box_type="quad")

        self.det_image_shape = [3, max_h, max_w]
        cuda.init()
        self.device = cuda.Device(gup_id)
        self.cuda_context = self.device.make_context()
        self.logger = trt.Logger(trt.Logger.ERROR)
        self.cuda_context.push()
        self.max_batch = max_batch
        if max_batch > self.max_batch:
            self.max_batch = max_batch
        try:
            self.stream = cuda.Stream()
            with open(engine_path, "rb") as f, trt.Runtime(self.logger) as runtime:
                self.engine = runtime.deserialize_cuda_engine(f.read())
            self.context = self.engine.create_execution_context()
            assert self.engine
            assert self.context

            # Setup I/O bindings
            self.inputs = []
            self.outputs = []
            self.allocations = []
            for i in range(self.engine.num_bindings):
                is_input = False
                if self.engine.binding_is_input(i):
                    is_input = True
                name = self.engine.get_binding_name(i)
                dtype = self.engine.get_binding_dtype(i)
                shape = self.engine.get_binding_shape(i)
                if shape[0] < 0:
                    shape[0] = self.max_batch

                if shape[2] < 0:
                    shape[2] = self.det_image_shape[1]
                if shape[3] < 0:
                    shape[3] = self.det_image_shape[2]
                # if is_input:
                #     if shape[2] < 0:
                #         shape[2] = self.det_image_shape[1]
                #     if shape[3] < 0:
                #         shape[3] = self.det_image_shape[2]
                # else:
                #     if shape[2] < 0:
                #         shape[2] = self.det_image_shape[1]
                #     if shape[3] < 0:
                #         shape[3] = self.det_image_shape[2]
                np_type = np.float32
                if dtype.name == "BOOL":
                    np_type = np.float32
                elif dtype.name == "HALF":
                    np_type = np.float16
                elif dtype.name == "INT32":
                    np_type = np.int32
                elif dtype.name == "INT8":
                    np_type = np.int32
                # size = np.dtype(trt.nptype(dtype)).itemsize
                size = dtype.itemsize
                for s in shape:
                    size *= s
                allocation = cuda.mem_alloc(size)
                binding = {
                    'index': i,
                    'name': name,
                    'dtype': np_type,
                    'shape': list(shape),
                    'allocation': allocation,
                }
                self.allocations.append(allocation)
                if self.engine.binding_is_input(i):
                    self.inputs.append(binding)
                else:
                    self.outputs.append(binding)

            assert len(self.inputs) > 0
            assert len(self.outputs) > 0
            assert len(self.allocations) > 0
        except Exception as e:
            log.get().error(
                " Paddle ocr det trt engine init fail : " + repr(e) + " engine path: " + engine_path)
            pass
        finally:
            self.cuda_context.pop()

    def input_spec_max(self):
        """
        Get the specs for the input tensor of the network. Useful to prepare memory allocations.
        :return: Two items, the shape of the input tensor and its (numpy) datatype.
        """
        return self.inputs[0]['shape'], self.inputs[0]['dtype']

    def output_spec_max(self):
        """
        Get the specs for the output tensor of the network. Useful to prepare memory allocations.
        :return: Two items, the shape of the output tensor and its (numpy) datatype.
        """
        return self.outputs[0]['shape'], self.outputs[0]['dtype']

    def infer(self, batch):
        """
        Execute inference on a batch of images. The images should already be batched and preprocessed, as prepared by
        the ImageBatcher class. Memory copying to and from the GPU device will be performed here.
        :param batch: A numpy array holding the image batch.
        :param top: The number of classes to return as top_predicitons, in descending order by their score. By default,
        setting to one will return the same as the maximum score class. Useful for Top-5 accuracy metrics in validation.
        :return: Three items, as numpy arrays for each batch image: The maximum score class, the corresponding maximum
        score, and a list of the top N classes and scores.
        """

        # Prepare the output data
        out_shape = self.output_spec_max()
        out_shape[0][0] = batch.shape[0]
        output = np.zeros(*out_shape)
        self.cuda_context.push()
        try:
            self.context.set_binding_shape(0, batch.shape)
            # Process I/O and execute the network
            cuda.memcpy_htod_async(self.inputs[0]['allocation'], np.ascontiguousarray(batch), self.stream)
            self.context.execute_async_v2(self.allocations, stream_handle=self.stream.handle)
            cuda.memcpy_dtoh_async(output, self.outputs[0]['allocation'], self.stream)

        except Exception as e:
            log.get().error(
                " ocr infer fail : " + repr(e) + " in shape: " + str(batch.shape) + "out put:" + str(out_shape))
            pass
        finally:
            self.cuda_context.pop()

        return output

    def __del__(self):
        if self.cuda_context:
            self.cuda_context.pop()

    def predict(self, image_list):
        img_num = len(image_list)
        # Calculate the aspect ratio of all text bars
        width_list = []
        for img in image_list:
            width_list.append(img.shape[1] / float(img.shape[0]))
        # Sorting can speed up the recognition process
        indices = np.argsort(np.array(width_list))
        batch_num = self.max_batch
        filter_det_res = [{}] * img_num
        for beg_img_no in range(0, img_num, batch_num):
            end_img_no = min(img_num, beg_img_no + batch_num)
            norm_img_batch = []
            image_shape_list = []

            imgC, imgH, imgW = self.det_image_shape[:3]
            max_wh_ratio = imgW / imgH
            # max_wh_ratio = 0
            for ino in range(beg_img_no, end_img_no):
                h, w = image_list[indices[ino]].shape[0:2]
                wh_ratio = w * 1.0 / h
                max_wh_ratio = max(max_wh_ratio, wh_ratio)
            for ino in range(beg_img_no, end_img_no):
                # data = {'image': image_list[indices[ino]]}
                # norm_img = transform(data, self.preprocess_op)
                norm_img, shape_list = resize_norm_img_det(img=image_list[indices[ino]], img_c=imgC, img_h=imgH,
                                                           img_w=imgW)
                norm_img = norm_img[np.newaxis, :]
                norm_img_batch.append(norm_img)
                image_shape_list.append(shape_list)
            norm_img_batch = np.concatenate(norm_img_batch)
            norm_img_batch = norm_img_batch.copy()

            with _det_infer_lock:
                det_outputs = self.infer(norm_img_batch)

            preds = {'maps': det_outputs}
            det_res = self.post_process(preds, image_shape_list)
            for ino in range(beg_img_no, end_img_no):
                filter_det_res[indices[ino]] = det_res[ino - beg_img_no]
            # print(rec_res)
            # filter_rec_res = []
            # for rec_result in rec_res:
            #     text, score = rec_result
            #     if score >= self.drop_score:
            #         filter_rec_res.append(rec_result)
        return filter_det_res


class TextRecognitionTrtInfer(object):
    def __init__(self, engine_path='weights/onnx/number_recog_x_3_48_x.engine', max_batch=8, gup_id=0):
        # Load TRT engine
        self.rec_image_shape = [int(v) for v in '3, 48, 320'.split(",")]
        self.rec_postprocess_op = CTCLabelDecode('weights/onnx/en_dict.txt', True)
        self.drop_score = 0.5
        cuda.init()
        self.device = cuda.Device(gup_id)
        self.cuda_context = self.device.make_context()
        self.logger = trt.Logger(trt.Logger.ERROR)
        self.cuda_context.push()
        self.max_batch = 6
        if max_batch > self.max_batch:
            self.max_batch = max_batch
        try:
            self.stream = cuda.Stream()
            with open(engine_path, "rb") as f, trt.Runtime(self.logger) as runtime:
                self.engine = runtime.deserialize_cuda_engine(f.read())
            self.context = self.engine.create_execution_context()
            assert self.engine
            assert self.context

            # Setup I/O bindings
            self.inputs = []
            self.outputs = []
            self.allocations = []
            for i in range(self.engine.num_bindings):
                is_input = False
                if self.engine.binding_is_input(i):
                    is_input = True
                name = self.engine.get_binding_name(i)
                dtype = self.engine.get_binding_dtype(i)
                shape = self.engine.get_binding_shape(i)
                if shape[0] < 0:
                    shape[0] = self.max_batch
                if is_input:
                    if shape[3] < 0:
                        shape[3] = self.rec_image_shape[2]
                else:
                    if shape[1] < 0:
                        shape[1] = int(self.rec_image_shape[2] / 8)
                np_type = np.float32
                if dtype.name == "BOOL":
                    np_type = np.float32
                elif dtype.name == "HALF":
                    np_type = np.float16
                elif dtype.name == "INT32":
                    np_type = np.int32
                elif dtype.name == "INT8":
                    np_type = np.int32
                # size = np.dtype(trt.nptype(dtype)).itemsize
                size = dtype.itemsize
                for s in shape:
                    size *= s
                allocation = cuda.mem_alloc(size)
                binding = {
                    'index': i,
                    'name': name,
                    'dtype': np_type,
                    'shape': list(shape),
                    'allocation': allocation,
                }
                self.allocations.append(allocation)
                if self.engine.binding_is_input(i):
                    self.inputs.append(binding)
                else:
                    self.outputs.append(binding)

            assert len(self.inputs) > 0
            assert len(self.outputs) > 0
            assert len(self.allocations) > 0
        except Exception as e:
            log.get().error(
                " Paddle ocr trt engine init fail : " + repr(e) + " engine path: " + engine_path)
            pass
        finally:
            self.cuda_context.pop()

    def input_spec_max(self):
        """
        Get the specs for the input tensor of the network. Useful to prepare memory allocations.
        :return: Two items, the shape of the input tensor and its (numpy) datatype.
        """
        return self.inputs[0]['shape'], self.inputs[0]['dtype']

    def output_spec_max(self):
        """
        Get the specs for the output tensor of the network. Useful to prepare memory allocations.
        :return: Two items, the shape of the output tensor and its (numpy) datatype.
        """
        return self.outputs[0]['shape'], self.outputs[0]['dtype']

    def infer(self, batch):
        """
        Execute inference on a batch of images. The images should already be batched and preprocessed, as prepared by
        the ImageBatcher class. Memory copying to and from the GPU device will be performed here.
        :param batch: A numpy array holding the image batch.
        :param top: The number of classes to return as top_predicitons, in descending order by their score. By default,
        setting to one will return the same as the maximum score class. Useful for Top-5 accuracy metrics in validation.
        :return: Three items, as numpy arrays for each batch image: The maximum score class, the corresponding maximum
        score, and a list of the top N classes and scores.
        """

        # Prepare the output data
        out_shape = self.output_spec_max()
        out_shape[0][0] = batch.shape[0]
        output = np.zeros(*out_shape)
        self.cuda_context.push()
        try:
            self.context.set_binding_shape(0, batch.shape)
            # Process I/O and execute the network
            cuda.memcpy_htod_async(self.inputs[0]['allocation'], np.ascontiguousarray(batch), self.stream)
            self.context.execute_async_v2(self.allocations, stream_handle=self.stream.handle)
            cuda.memcpy_dtoh_async(output, self.outputs[0]['allocation'], self.stream)

        except Exception as e:
            log.get().error(
                " ocr infer fail : " + repr(e) + " in shape: " + str(batch.shape) + "out put:" + str(out_shape))
            pass
        finally:
            self.cuda_context.pop()

        return output

    def __del__(self):
        if self.cuda_context:
            self.cuda_context.pop()

    def predict(self, image_list):
        img_num = len(image_list)
        # Calculate the aspect ratio of all text bars
        width_list = []
        for img in image_list:
            width_list.append(img.shape[1] / float(img.shape[0]))
        # Sorting can speed up the recognition process
        indices = np.argsort(np.array(width_list))
        # rec_res = [['', 0.0]] * img_num
        batch_num = self.max_batch
        filter_rec_res = [['', 0.0]] * img_num
        for beg_img_no in range(0, img_num, batch_num):
            end_img_no = min(img_num, beg_img_no + batch_num)
            norm_img_batch = []

            imgC, imgH, imgW = self.rec_image_shape[:3]
            max_wh_ratio = imgW / imgH
            # max_wh_ratio = 0
            # for ino in range(beg_img_no, end_img_no):
            #     h, w = image_list[indices[ino]].shape[0:2]
            #     wh_ratio = w * 1.0 / h
            #     max_wh_ratio = max(max_wh_ratio, wh_ratio)
            for ino in range(beg_img_no, end_img_no):
                norm_img = resize_norm_img(image_list[indices[ino]],
                                           max_wh_ratio, imgC, imgH, imgW)
                norm_img = norm_img[np.newaxis, :]
                norm_img_batch.append(norm_img)
            norm_img_batch = np.concatenate(norm_img_batch)
            norm_img_batch = norm_img_batch.copy()

            with _infer_lock:
                rec_outputs = self.infer(norm_img_batch)
            # preds = rec_outputs
            rec_res = self.rec_postprocess_op(rec_outputs)
            # print(rec_res)
            for ino in range(beg_img_no, end_img_no):
                text, score = rec_res[ino - beg_img_no]
                if score >= self.drop_score:
                    filter_rec_res[indices[ino]] = rec_res[ino - beg_img_no]
        return filter_rec_res


def draw_text_det_res(dt_boxes, img):
    b_len = len(dt_boxes)
    s_i = -1
    area = 0
    for j in range(b_len):
        box = dt_boxes[j]
        box = np.array(box).astype(np.int32).reshape(-1, 2)
        d_x = box[1] - box[0]
        d_y = box[3] - box[0]
        w_len = np.sqrt(np.dot(d_x, d_x))
        h_len = np.sqrt(np.dot(d_y, d_y))
        area = w_len * h_len
    for box in dt_boxes:
        box = np.array(box).astype(np.int32).reshape(-1, 2)
        a = box[1] - box[0]
        b = box[2] - box[3]
        c = box[3] - box[0]
        d = box[2] - box[1]
        print("a=", a, "b=", d)
        cv2.polylines(img, [box], True, color=(255, 255, 0), thickness=2)
    return img


def getPolygonArea(dt_boxes):
    list1 = []
    for points in dt_boxes:
        area = points[-1][0] * points[0][1] - points[0][0] * points[-1][1]
        for i in range(1, len(points)):
            v = i - 1
            area += (points[v][0] * points[i][1])
            area -= (points[i][0] * points[v][1])
        list1.append(abs(0.5 * area))
    try:
        i = list1.index(max(list1))
    except:
        i = list1
    return list1, i


def getPolygonCertroid(points):
    x_sum, y_sum = 0., 0.
    for point in points:
        x_sum += point[0]
        y_sum += point[1]
    try:
        centroid = [x_sum / len(points), y_sum / len(points)]
    except:
        centroid = [0., 0.]

    return centroid


def defineOtherPosition(dt_boxes, index, position):
    number_box = dt_boxes[index]
    number_centroid = getPolygonCertroid(number_box)

    other_boxes = np.delete(dt_boxes, index, axis=0)
    # print(other_boxes)
    top_left, top_right, bot_left, bot_right = np.array([number_box]), np.array([number_box]), np.array(
        [number_box]), np.array([number_box])
    for point in other_boxes:
        point_centroid = getPolygonCertroid(point)
        if point_centroid[0] < number_centroid[0] and point_centroid[1] < number_centroid[1]:
            top_left = np.append(top_left, [point], axis=0)
        if point_centroid[0] > number_centroid[0] and point_centroid[1] < number_centroid[1]:
            top_right = np.append(top_right, [point], axis=0)
        if point_centroid[0] < number_centroid[0] and point_centroid[1] > number_centroid[1]:
            bot_left = np.append(bot_left, [point], axis=0)
        if point_centroid[0] > number_centroid[0] and point_centroid[1] > number_centroid[1]:
            bot_right = np.append(bot_right, [point], axis=0)
    if position == 0:
        return top_left
    if position == 1:
        return top_right
    if position == 2:
        return bot_left
    if position == 3:
        return bot_right


def remove_all(list, item):
    return [i for i in list if i != item]


class TextDetRecognitionTrtInfer(object):
    def __init__(self, det_engine_path='weights/onnx/text_det_x_3_x_x.engine',
                 rec_engine_path='weights/onnx/number_recog_x_3_48_x.engine', gup_id=0):
        self.det_model = TextDetTrtInfer(engine_path=det_engine_path, max_h=192, max_w=192, max_batch=6, gup_id=gup_id)
        self.rec_model = TextRecognitionTrtInfer(engine_path=rec_engine_path, max_batch=6, gup_id=gup_id)
        self.ratio_w_h = 320 / 48.
        # self.test_index = 0
        self.is_special_number = True
        self.position = 0

    def predict(self, image_list):
        img_num = len(image_list)

        dets = self.det_model.predict(image_list)
        nlen = len(dets)

        img_crop_list_src_index = []
        for i in range(nlen):
            img_crop_list = []
            dt_boxes = dets[i]["points"]
            src_img = image_list[i]

            # h, w, _ = src_img.shape
            # b_len = len(dt_boxes)
            # s_i = -1
            # max_area = 0
            # for j in range(b_len):
            #     box = dt_boxes[j]
            #     box = np.array(box).astype(np.float32).reshape(-1, 2)
            #     d_x = box[1] - box[0]
            #     d_y = box[3] - box[0]
            #     w_len = np.sqrt(np.dot(d_x, d_x))
            #     h_len = np.sqrt(np.dot(d_y, d_y))
            #     area = w_len * h_len
            #     ratio = w_len / h_len
            #     if area > max_area and ratio < self.ratio_w_h + 0.00001:
            #         max_area = area
            #         s_i = j
            # if -1 < s_i < b_len:
            #     img_crop = get_rotate_crop_image(src_img, dt_boxes[s_i].astype(np.float32))
            #     # self.test_index += 1
            #     # cv2.imwrite('test/'+str(self.test_index)+"_src.jpg", src_img)
            #     # cv2.imwrite('test/'+str(self.test_index)+"_crop.jpg", img_crop)
            #     img_crop_list.append(img_crop)
            #     img_crop_list_src_index.append(i)

            try:
                if self.is_special_number:
                    dict, index = getPolygonArea(dt_boxes)
                    new_dt_boxes = defineOtherPosition(dt_boxes, index, self.position)
                    dt_boxes = new_dt_boxes
                    # print('--------------------', dt_boxes)
                    dt_boxes = sorted_boxes(dt_boxes)

                for bno in range(len(dt_boxes)):
                    tmp_box = copy.deepcopy(dt_boxes[bno])
                    if len(tmp_box) > 0:
                        img_crop = get_rotate_crop_image(src_img, tmp_box.astype(np.float32))

                        img_crop_list.append(img_crop)

                filter_rec_res = [['', 0.0]] * len(img_crop_list)
                if len(img_crop_list) > 0:
                    rec_result = self.rec_model.predict(img_crop_list)
                    rec_len = len(rec_result)

                    # need process one image has mutil crop
                    for i_rec in range(rec_len):
                        filter_rec_res[i_rec] = rec_result[i_rec]
                        # print(filter_rec_res)
                filter_rec_res = remove_all(filter_rec_res, ['', 0.0])

                a, b = '', filter_rec_res[0][1]
                # filter_rec_res1 = []
                for filter in filter_rec_res:
                    a += filter[0]
                    b = (b + filter[1]) / 2.
                    # filter_rec_res1.append((a, float(b)))
                img_crop_list_src_index += [(a, float(b))]
            except:
                continue

        return img_crop_list_src_index


if __name__ == '__main__':
    # ocr_test = TextRecognitionTrtInfer()
    # img1 = cv2.imread('/home/yanjun/SourceCode/python_tool/search/recog/A0042_18.jpg')
    # img = cv2.imread('/home/yanjun/SourceCode/python_tool/search/recog/3.jpg')
    img1 = cv2.imread('1.jpg')
    img = cv2.imread('/home/kls/SourceCode/PC/KLAICenter/PaddleOCR/3.jpg')
    # img = cv2.resize(img,(360, 48))
    # ret = ocr_test.predict([img1, img])

    # det = TextDetTrtInfer(max_w=160, max_h=64)
    # hh, shap_list = resize_norm_img_det(img, img_h=160, img_w=64)
    # ret = det.predict([img, img,img, img,img, img])
    # boxes = sorted_boxes(ret[0]["points"])

    # draw_text_det_res(boxes, img)
    # cv2.imshow("hh", img)
    # cv2.waitKey()
    # pass
    # print(ret)
    test = TextDetRecognitionTrtInfer()
    print(test.predict([img, img1]))
    pass
