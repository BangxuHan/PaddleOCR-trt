import base64
import json
import cv2
import numpy as np
import copy
import time
import config as cfg
from flask import Flask, request
from flask_restful import Resource, Api
from tools.infer import predict_system
import tools.infer.predict_rec as predict_rec
import tools.infer.predict_det as predict_det
import tools.infer.utility as utility
from tools.infer.utility import get_rotate_crop_image
from ppocr.data import create_operators, transform
import threading
import argparse
app = Flask("paddleOCR")
api = Api(app)


_infer_lock = threading.Lock()


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


class TextRecognition(object):
    def __init__(self, args, seq_len):
        self.args = args
        self.seq_len = cfg.seq_len

        args.det_model_dir = 'weights/en_PP-OCR_v3_det_infer/Teacher'
        args.rec_model_dir = 'weights/en_PP-OCRv4_rec_infer'

        self.pred = predict_system.TextSystem(args)

        # self.text_detector = predict_det.TextDetector(args)
        # self.text_recognizer = predict_rec.TextRecognizer(args)

        self.drop_score = args.drop_score
        self.position = 0

    def __call__(self, img):
        time_dict = {'det': 0, 'rec': 0, 'cls': 0, 'all': 0}
        start = time.time()
        ori_im = img.copy()
        dt_boxes, elapse = self.pred.text_detector(img)
        time_dict['det'] = elapse

        if dt_boxes is None:
            return None, None
        dt_boxes = sorted_boxes(dt_boxes)

        dict, index = self.pred.text_detector.getPolygonArea(dt_boxes)
        new_dt_boxes = self.pred.text_detector.defineOtherPosition(dt_boxes, index, self.position)
        print('new', new_dt_boxes)
        dt_boxes = new_dt_boxes

        img_crop_list = []
        for bno in range(len(dt_boxes)):
            tmp_box = copy.deepcopy(dt_boxes[bno])
            img_crop = get_rotate_crop_image(ori_im, tmp_box)

            img_crop_list.append(img_crop)
        # try:
        rec_res, elapse = self.pred.text_recognizer(img_crop_list)
        # except Exception as e:
        #     print(str(e))
        print(rec_res)
        time_dict['rec'] = elapse

        filter_boxes, filter_rec_res = [], []
        for box, rec_result in zip(dt_boxes, rec_res):
            text, score = rec_result
            if score >= self.drop_score:
                filter_boxes.append(box)
                filter_rec_res.append(rec_result)

        end = time.time()
        time_dict['all'] = end - start
        return filter_boxes, filter_rec_res, time_dict

    def predict(self, imgs):
        # imgs = [img]
        ocr_res = []
        for idx, img in enumerate(imgs):
            dt_boxes, rec_res, _ = self.__call__(img)
            tmp_res = [[box.tolist(), res] for box, res in zip(dt_boxes, rec_res)]
            if len(rec_res) > 0:
                ocr_res.append(rec_res[0])
        return ocr_res


class TextRecognitionNoDetect(object):
    def __init__(self, args, seq_len):
        self.args = args
        self.seq_len = cfg.seq_len

        # args.det_model_dir = 'weights/en_PP-OCRv3_det_infer'
        args.rec_model_dir = 'weights/en_PP-OCRv4_rec_infer'

        # self.pred = predict_system.TextSystem(args)
        self.index = 0

        # self.text_detector = predict_det.TextDetector(args)
        self.text_recognizer = predict_rec.TextRecognizer(args)

        self.drop_score = args.drop_score

    def __call__(self, imgs):
        time_dict = {'det': 0, 'rec': 0, 'cls': 0, 'all': 0}
        start = time.time()
        # ori_im = img.copy()
        self.index += 1
        # cv2.imwrite(str(self.index)+".jpg", ori_im)
        # dt_boxes, elapse = self.pred.text_detector(img)
        # print("jia ", dt_boxes)
        # time_dict['det'] = elapse
        #
        # if dt_boxes is None:
        #     return None, None
        # dt_boxes = sorted_boxes(dt_boxes)
        #
        # img_crop_list = []
        # for bno in range(len(dt_boxes)):
        #     tmp_box = copy.deepcopy(dt_boxes[bno])
        #     img_crop = get_rotate_crop_image(ori_im, tmp_box)
        #
        #     img_crop_list.append(img_crop)
        # try:
        with _infer_lock:
            rec_res, elapse = self.text_recognizer(imgs)
        # rec_res, elapse = self.text_recognizer(imgs)
        # except Exception as e:
        #     print(str(e))
        print(rec_res)
        time_dict['rec'] = elapse

        filter_boxes, filter_rec_res = [], []
        for rec_result in rec_res:
            text, score = rec_result
            if score >= self.drop_score:
                filter_rec_res.append(rec_result)

        end = time.time()
        time_dict['all'] = end - start
        return filter_boxes, filter_rec_res, time_dict

    def predict(self, imgs):
        #ocr_res = []
        dt_boxes, rec_res, _ = self.__call__(imgs)
        #ocr_res.append(rec_res)
        return rec_res


class KLArgs():
    use_gpu = True
    use_tensorrt = False
    min_subgraph_size = 15
    precision = "fp32"
    gpu_mem = 500
    gpu_id = 0
    # params for text detector
    page_num = 0
    det_algorithm = 'DB'

    det_limit_side_len = 960
    det_limit_type = 'max'
    det_box_type = 'quad'
    # DB parmas
    det_db_thresh = 0.3
    det_db_box_thresh = 0.6
    det_db_unclip_ratio = 1.5
    max_batch_size = 10
    use_dilation = False
    det_db_score_mode = "fast"

    # params for text recognizer
    rec_algorithm = 'SVTR_LCNet'
    rec_image_inverse = True
    rec_image_shape = "3, 48, 320"
    rec_batch_num = 6
    max_text_length = 25
    rec_char_dict_path = "./ppocr/utils/en_dict.txt"
    use_space_char = True
    drop_score = 0.5
    total_process_num = 1
    show_log = True
    use_onnx = False
    det_model_dir = 'weights/en_PP-OCR_v3_det_infer/Teacher'
    rec_model_dir = 'weights/en_PP-OCRv4_rec_infer'

    def __init__(self):
        pass
        # self.use_gpu = True
        # self.use_tensorrt = False
        # self.min_subgraph_size=15
        # self.precision="fp32"
        # self.gpu_mem=500
        # self.gpu_id=0
        # # params for text detector
        # self.page_num=0
        # self.det_algorithm='DB'
        #
        # self.det_limit_side_len=960
        # self.det_limit_type='max'
        # self.det_box_type='quad'
        # # DB parmas
        # self.det_db_thresh=0.3
        # self.det_db_box_thresh=0.6
        # self.det_db_unclip_ratio=1.5
        # self.max_batch_size=10
        # self.use_dilation=False
        # self.det_db_score_mode="fast"
        #
        # # params for text recognizer
        # self.rec_algorithm='SVTR_LCNet'
        # self.rec_image_inverse=True
        # self.rec_image_shape="3, 48, 320"
        # self.rec_batch_num=6
        # self.max_text_length=25
        # self.rec_char_dict_path="./ppocr/utils/en_dict.txt"
        # self.use_space_char=True
        # self.drop_score=0.5
        # self.total_process_num=1
        # self.show_log=True
        # self.use_onnx=False


class CharRecognition(Resource):
    def post(self):
        temp = request.get_data(as_text=True)
        data = json.loads(temp)
        images = data['image']
        imagebuf = []
        for imagestr in images:
            imagedata_base64 = base64.b64decode(imagestr)
            np_arr = np.frombuffer(imagedata_base64, dtype=np.uint8)
            image = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
            # image = image_preprocessing(image, (cfg.image_size, cfg.image_size))
            imagebuf.append(image)
        # try:
        #     imagebuf = np.array(imagebuf)
        # except Exception as e:
        #     print(e)
        with _infer_lock:
            ocr_res = model.predict(imagebuf)

        words_res = []
        nlen = len(ocr_res)
        # print(ocr_res)
        for i in range(nlen):
            if ocr_res[i] is not None and len(ocr_res[i]) > 0:
                temp = {
                    "words": ocr_res[i][0],
                    "probability": {"average": ocr_res[i][1].__float__()}
                }
                words_res.append(temp)

        result = {"words_result_num": nlen,
                  "words_result": words_res
                  }
        return app.response_class(json.dumps(result), mimetype='application/json')


api.add_resource(CharRecognition, '/charrecog')
args = KLArgs()
model = TextRecognition(args, cfg.seq_len + 1)
# model = TextRecognitionNoDetect(args, cfg.seq_len + 1)
# model = TextRecognition(utility.parse_args(), cfg.seq_len + 1)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--port', type=int, default=5000, help='number recognition server port')
    args = parser.parse_args()
    port = args.port
    app.run(host='0.0.0.0', port=port, debug=False)
