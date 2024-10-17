import torch
from PIL import Image
import cv2
import numpy as np
import sys
from models.common import DetectMultiBackend
from utils.augmentations import letterbox
from utils.general import non_max_suppression
import streamlit as st
from models.export import export_formats

# 设置 YOLOv5 的本地路径
yolov5_path = yolov5m_fracture.pt  # 替换为你克隆的 yolov5 仓库的路径
sys.path.append(yolov5_path)

# 预处理图片
def preprocess_image(image_path, input_size=640):
    img = Image.open(image_path).convert('RGB')
    img = np.array(img)  # 转换为 numpy 数组
    img = letterbox(img, input_size)[0]  # 调整大小并保持比例
    img = img.transpose((2, 0, 1))  # 转换为模型需要的 [C, H, W] 格式
    img = np.ascontiguousarray(img)  # 确保内存连续
    img = torch.from_numpy(img).float() / 255.0  # 归一化处理并转换为张量
    if img.ndimension() == 3:
        img = img.unsqueeze(0)  # 添加批次维度
    return img

# 加载模型并进行推理
def detect(image_path, model_path=yolov5m_fracture.pt, conf_threshold=0.5, input_size=640):
    # 使用 YOLOv5 的 DetectMultiBackend 加载模型
    model = DetectMultiBackend(model_path)

    # 预处理图片
    img = preprocess_image(image_path, input_size)

    # 模型推理
    pred = model(img)

    # 非极大值抑制（NMS）去除多余的检测框
    pred = non_max_suppression(pred, conf_threshold)

    # 读取原图大小用于坐标转换
    original_img = Image.open(image_path)
    original_size = original_img.size  # 原图的宽高

    # 坐标转换
    boxes = []
    if pred[0] is not None and len(pred[0]):  # 如果有检测结果
        # 将检测框坐标映射回原图尺寸
        for detection in pred[0]:
            # 提取检测框的 xyxy 格式 [x1, y1, x2, y2, confidence, class]
            x1, y1, x2, y2 = detection[:4].tolist()
            # 手动缩放到原图尺寸
            x1 = int(x1 * (original_size[0] / input_size))
            y1 = int(y1 * (original_size[1] / input_size))
            x2 = int(x2 * (original_size[0] / input_size))
            y2 = int(y2 * (original_size[1] / input_size))
            boxes.append([x1, y1, x2, y2])  # 存储为整数类型的坐标

    return boxes, original_img

# 结果可视化
def visualize_detection(original_img, boxes):
    img = np.array(original_img)
    for box in boxes:
        x1, y1, x2, y2 = box
        cv2.rectangle(img, (x1, y1), (x2, y2), (255, 0, 0), 2)  # 画出检测框
    return img

# Streamlit 界面
def main():
    st.title("骨折检测")
    
    # 上传图片
    uploaded_file = st.file_uploader("选择一张图片", type=["jpg", "jpeg", "png"])
    
    if uploaded_file is not None:
        # 将上传的图片转换为 PIL 格式
        image_path = uploaded_file.name
        with open(image_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        
        # 检测骨折
        boxes, original_img = detect(image_path)
        
        # 可视化检测结果
        result_img = visualize_detection(original_img, boxes)

        # 显示检测结果
        st.image(result_img, caption="检测结果", use_column_width=True)

if __name__ == "__main__":
    main()
