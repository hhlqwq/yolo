from ultralytics import YOLO


def predict():
    model = YOLO("runs/detect/yolov11s_p2_detect_3cls/v6/weights/best.pt")

    results = model.predict(
        source="/data/users/hailong.he/datasets/opc_clean/zicai260730/05_1_paper/img_src/",
        save=True,
        conf=0.3,
        imgsz=640,
        device=0,              # 使用GPU,CPU则用 device='cpu'
    )
    


if __name__ == "__main__":
    predict()
