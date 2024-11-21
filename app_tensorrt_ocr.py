import base64
import json
from flask import Flask, request
from flask_restful import Resource, Api
import cv2
import numpy as np
import argparse
from logger import logger as log
# from paddle_tensorrt import TextRecognitionTrtInfer
from paddle_tensorrt import TextDetRecognitionTrtInfer


app = Flask("paddleOCRTrt")
api = Api(app)


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
# model = TextRecognitionTrtInfer(engine_path='weights/onnx/number_recog_x_3_48_x.engine')

# model = TextDetRecognitionTrtInfer(det_engine_path='weights/onnx/text_det_x_3_x_x.engine', rec_engine_path='weights/onnx/number_recog_x_3_48_x.engine')
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--port', type=int, default=5000, help='number recognition server port')
    parser.add_argument('--gpuID', type=int, default=0, help='number recognition server port')
    args = parser.parse_args()
    port = args.port
    gpuId = args.gpuID
    model = TextDetRecognitionTrtInfer(det_engine_path='weights/onnx/text_det_x_3_x_x.engine',
                                       rec_engine_path='weights/onnx/number_recog_x_3_48_x.engine',gup_id=gpuId)
    log.get().info("port: " + str(port))
    app.run(host='0.0.0.0', port=port, debug=False)
