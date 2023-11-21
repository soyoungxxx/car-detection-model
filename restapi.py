# YOLOv5 🚀 by Ultralytics, AGPL-3.0 license
"""
Run a Flask REST API exposing one or more YOLOv5s models
"""

import argparse
import io
import json
import numpy as np
from ultralytics.utils.plotting import Annotator, colors

import torch
from flask import Flask, request
from PIL import Image

app = Flask(__name__)

DETECTION_URL = '/v1/object-detection/best'


@app.route(DETECTION_URL, methods=['POST'])
def predict():
    if request.method != 'POST':
        return

    if request.files.get('image'):
        # Method 1
        # with request.files["image"] as f:
        #     im = Image.open(io.BytesIO(f.read()))

        # Method 2
        im_file = request.files['image']
        im_bytes = im_file.read()
        im = Image.open(io.BytesIO(im_bytes))

        img = []
        img.append(im)
        '''
        이미지 읽어 오는 부분에서 한 장씩 처리 or 여러 장 한 번에 처리
        우선 전체적으로 여러 장을 리스트에 넣어서 한 번에 처리하는 방식으로 구현을 했습니다!
        '''

        result = model(img, size=640)

        names = result.names
        class_dict = dict()
        for i, det in enumerate(result.pred) :
            x = np.ascontiguousarray(img[i])
            annotator = Annotator(x, line_width=3, example=str(names))
            for *xyxy, cls, in reversed(det) :
                c = int(cls)
                label = f'{names[c]}'

                key = names[c]
                if (key in class_dict) :
                    value = class_dict.get(key) + 1
                    class_dict.update({key : value})
                else :
                    class_dict[key] = 1
        
                annotator.box_label(xyxy, label, color=colors(c, True))
            # bounding box 그려서 img에 저장
            img[i] = Image.fromarray(annotator.result())
        # json 파일로 변환
        output = json.dumps(class_dict)

        return output


if __name__ == '__main__':
    default_port = 5000
    model = torch.hub.load("ultralytics/yolov5", 'custom', 'D:/car-detection-model/best.pt', force_reload=True, skip_validation=True)

    app.run(host='0.0.0.0', port=default_port)  # debug=True causes Restarting with stat
