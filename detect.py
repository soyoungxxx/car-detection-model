import os
import cv2

import torch
import json

from ultralytics.utils.plotting import Annotator, colors

from dataloader import LoadImages;
from common import DetectMultiBackend

from general import (Profile, check_img_size, check_version,
                    non_max_suppression, scale_boxes)
#필요한 코드 import

def select_device(device='', batch_size=0, newline=True):
    # device = None or 'cpu' or 0 or '0' or '0,1,2,3'
    s = ''
    device = str(device).strip().lower().replace('cuda:', '').replace('none', '')  # to string, 'cuda:0' to '0'
    cpu = device == 'cpu'
    mps = device == 'mps'  # Apple Metal Performance Shaders (MPS)
    if cpu or mps:
        os.environ['CUDA_VISIBLE_DEVICES'] = '-1'  # force torch.cuda.is_available() = False
    elif device:  # non-cpu device requested
        os.environ['CUDA_VISIBLE_DEVICES'] = device  # set environment variable - must be before assert is_available()
        assert torch.cuda.is_available() and torch.cuda.device_count() >= len(device.replace(',', '')), \
            f"Invalid CUDA '--device {device}' requested, use '--device cpu' or pass valid CUDA device(s)"

    if not cpu and not mps and torch.cuda.is_available():  # prefer GPU if available
        devices = device.split(',') if device else '0'  # range(torch.cuda.device_count())  # i.e. 0,1,6,7
        n = len(devices)  # device count
        if n > 1 and batch_size > 0:  # check batch_size is divisible by device_count
            assert batch_size % n == 0, f'batch-size {batch_size} not multiple of GPU count {n}'
        space = ' ' * (len(s) + 1)
        for i, d in enumerate(devices):
            p = torch.cuda.get_device_properties(i)
            s += f"{'' if i == 0 else space}CUDA:{d} ({p.name}, {p.total_memory / (1 << 20):.0f}MiB)\n"  # bytes to MB
        arg = 'cuda:0'
    elif mps and getattr(torch, 'has_mps', False) and torch.backends.mps.is_available():  # prefer MPS if available
        s += 'MPS\n'
        arg = 'mps'
    else:  # revert to CPU
        s += 'CPU\n'
        arg = 'cpu'

    if not newline:
        s = s.rstrip()
    return torch.device(arg)

def smart_inference_mode(torch_1_9=check_version(torch.__version__, '1.9.0')):
    def decorate(fn):
        return (torch.inference_mode if torch_1_9 else torch.no_grad)()(fn)

    return decorate

@smart_inference_mode()
def run(
        weights='best.pt',
        img = ['img1.jpg', 'img2.jpg'],
        data='custom_dataset.yaml',
        imgsz=(640, 640)
):
    class_dict = dict()

#def run(yamlfile, ptfile, img, imgsz)

    # Load model
    device = select_device()
    model = DetectMultiBackend(weights, device=device, dnn=False, data=data, fp16=False)
    stride, names, pt = model.stride, model.names, model.pt
    imgsz = check_img_size(imgsz, s=stride)  # check image size

    # Dataloader
    dataset = LoadImages(img, img_size=imgsz, stride=stride, auto=pt, vid_stride=1)

    # Run inference
    model.warmup(imgsz=(1, 3, *imgsz))  # warmup
    seen, windows, dt = 0, [], (Profile(), Profile(), Profile())
    for im, im0s in dataset:
        with dt[0]:
            im = torch.from_numpy(im).to(model.device)
            im = im.half() if model.fp16 else im.float()  # uint8 to fp16/32
            im /= 255  # 0 - 255 to 0.0 - 1.0
            if len(im.shape) == 3:
                im = im[None]  # expand for batch dim

        # Inference
        with dt[1]:
            pred = model(im, augment=False, visualize=False)

        # NMS
        with dt[2]:
            pred = non_max_suppression(pred, conf_thres=0.25, iou_thres=0.45, classes=None, agnostic_nms=False, max_det=1000)
            # pred = list of detections, on (n,6) tensor per image [xyxy, conf, cls]

        # Process predictions
        for i, det in enumerate(pred):  # per image
            seen += 1
            im0, frame = im0s.copy(), getattr(dataset, 'frame', 0)

            annotator = Annotator(im0, line_width=3, example=str(names))
            if len(det):
                # Rescale boxes from img_size to im0 size
                det[:, :4] = scale_boxes(im.shape[2:], det[:, :4], im0.shape).round()

                # Write results
                for *xyxy, conf, cls in reversed(det):
                    c = int(cls)  # integer class
                    label = names[c]

                    # custom codes
                    key = names[c]
                    if (key in class_dict) :
                        value = class_dict.get(key) + 1
                        class_dict.update({key : value})
                    else :
                        class_dict[key] = 1
                    # class 별로 객체 세는 코드

                    c = int(cls)  # integer class
                    label = names[c]
                    annotator.box_label(xyxy, label, color=colors(c, True))
                    # img에 bbox 추가하는 코드

            cv2.imwrite(save_path, im0)
            # img 저장하는 부분 -> return으로 변경 필요

    with open(f'{save_dir}/data.json', 'a') as f:
        json.dump(class_dict, f, ensure_ascii=False, indent=4)
    # ---------- 클래스 별로 객체 개수 json 저장---------
    # -> return 으로 변경 필요
