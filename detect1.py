import sys
import os
import argparse
import torch
import cv2
import numpy as np

# 将当前路径添加到系统路径中
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# 从绝对路径导入 DetectMultiBackend
from models.common import DetectMultiBackend  # 确保该路径指向正确的模块
from models.export import export_formats

def letterbox(img, new_shape=(640, 640), color=(114, 114, 114), auto=True, scaleFill=False, scaleup=True):
    """Resize image to a rectangular shape while maintaining aspect ratio."""
    shape = img.shape[:2]  # current shape [height, width]
    if isinstance(new_shape, int):
        new_shape = (new_shape, new_shape)
    ratio = min(new_shape[0] / shape[0], new_shape[1] / shape[1])  # calculate the ratio to resize
    if not scaleup:  # if scale up is False, prevent enlarging
        ratio = min(ratio, 1.0)
    new_unpad = int(round(shape[1] * ratio)), int(round(shape[0] * ratio))  # new size without padding
    dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]  # calculate padding
    if auto:  # if auto padding is enabled
        dw, dh = dw / 2, dh / 2  # divide padding by 2
    elif scaleFill:  # if scale fill is enabled
        dw, dh = 0.0, 0.0
        new_unpad = new_shape[1], new_shape[0]
    else:  # if no auto or scale fill
        dw, dh = np.floor(dw), np.floor(dh)  # apply floor to padding
    # resize and pad image
    img = cv2.resize(img, new_unpad, interpolation=cv2.INTER_LINEAR)
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))  # calculate top and bottom padding
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))  # calculate left and right padding
    img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)  # add padding
    return img

def non_max_suppression(prediction, conf_thres=0.25, iou_thres=0.45):
    """Apply NMS to prediction results."""
    # Your NMS implementation here
    # This is a simplified version for illustration
    output = []
    for pred in prediction:
        if pred is not None and len(pred):
            # Process prediction here...
            output.append(pred)  # Placeholder
    return output

def detect(image_path):
    model = DetectMultiBackend(yolov5m_fracture.pt)  # 替换为您的权重文件名
    img = cv2.imread(image_path)  # 读取图像
    img = letterbox(img, new_shape=(640, 640))  # 预处理图像
    img = img.transpose((2, 0, 1))[None]  # 转换维度
    img = torch.from_numpy(img).float() / 255.0  # 归一化
    if img.ndimension() == 3:
        img = img.unsqueeze(0)

    pred = model(img, augment=False)
    pred = non_max_suppression(pred, conf_thres=0.25, iou_thres=0.45)
    return pred

def main():
    parser = argparse.ArgumentParser(description='Fracture Detection with YOLOv5')
    parser.add_argument('image_path', type=str, help='Path to the input image')
    args = parser.parse_args()

    boxes = detect(args.image_path)
    for det in boxes:
        if det is not None:
            for *xyxy, conf, cls in det:
                print(f'Detected: {cls} with confidence {conf} at {xyxy}')

if __name__ == "__main__":
    main()
