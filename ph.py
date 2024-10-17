import torch
from models.experimental import attempt_load

def main():
    weights_path = "C:\\Users\\18301\\Desktop\\yolov5n_10.14.pt"
    device = 'cpu'  # 或者使用 'cuda' 根据您的环境

    # 使用 attempt_load 加载模型
    model = attempt_load(weights_path, device=device)
    
    # 其他代码逻辑...

if __name__ == "__main__":
    main()
