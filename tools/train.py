from ultralytics import YOLO


def yolov11s_clean_v1():
    model = YOLO("ultralytics/cfg/models/11/yolo11s_p2.yaml")
    model.load("runs/detect/yolov11s_p2_detect_3cls/v1/weights/best.pt")
    model.train(
        data="ultralytics/cfg/datasets/clean_v1_detect.yaml",    
        epochs=300,
        batch=12,  # batch=-1 开启自动检测  
        imgsz=640,
        name="v3",  
        project="yolov11s_p2_detect_3cls",
        exist_ok=False, # 允许覆盖同名实验
        device=0,
        # optimizer="AdamW",  # COCO 大 epoch 训练中,AdamW 比默认的 SGD 收敛更快、更稳 
        cos_lr=True,  # 使用余弦退火学习率调度器
        warmup_epochs=3,  # 预热阶段的训练轮数
        close_mosaic=10,    # 关闭马赛克数据增强的训练轮数
        amp=False,  # 使用自动混合精度训练
        workers=16,  # CPU 核心足够多时可以更高
        resume=True,
    )

def yolov11s_clean_v2():
    model = YOLO("ultralytics/cfg/models/11/yolo11s_p2.yaml")
    model.load("runs/detect/yolov11s_p2_detect_3cls/v2/weights/best.pt")
    model.train(
        data="ultralytics/cfg/datasets/clean_v2_detect.yaml",    
        epochs=300,
        batch=12,  # batch=-1 开启自动检测  
        imgsz=640,
        name="v4",  
        project="yolov11s_p2_detect_3cls",
        exist_ok=False, # 允许覆盖同名实验
        device=0,
        # optimizer="AdamW",  # COCO 大 epoch 训练中,AdamW 比默认的 SGD 收敛更快、更稳 
        cos_lr=True,  # 使用余弦退火学习率调度器
        warmup_epochs=3,  # 预热阶段的训练轮数
        close_mosaic=10,    # 关闭马赛克数据增强的训练轮数
        amp=False,  # 使用自动混合精度训练
        workers=16,  # CPU 核心足够多时可以更高
        resume=True,
    )

def yolov11s_clean_v3():
    model = YOLO("ultralytics/cfg/models/11/yolo11s_p2.yaml")
    model.load("runs/detect/yolov11s_p2_detect_3cls/v4/weights/best.pt")
    model.train(
        data="ultralytics/cfg/datasets/clean_v2_detect.yaml",    
        epochs=300,
        batch=12,  # batch=-1 开启自动检测  
        imgsz=640,
        name="v5",  
        project="yolov11s_p2_detect_3cls",
        exist_ok=False, # 允许覆盖同名实验
        device=0,
        # optimizer="AdamW",  # COCO 大 epoch 训练中,AdamW 比默认的 SGD 收敛更快、更稳 
        cos_lr=True,  # 使用余弦退火学习率调度器
        warmup_epochs=3,  # 预热阶段的训练轮数
        close_mosaic=10,    # 关闭马赛克数据增强的训练轮数
        amp=False,  # 使用自动混合精度训练
        workers=16,  # CPU 核心足够多时可以更高
        resume=True,
    )

def yolov11s_clean_v5_seg():
    model = YOLO("ultralytics/cfg/models/11/yolo11s-p2-seg.yaml")
    model.load("runs/segment/paper_zicai260721/v1_260722/weights/best.pt")
    model.train(
        data="ultralytics/cfg/datasets/clean_260728_segment.yaml",    
        epochs=300,
        batch=10,  # batch=-1 开启自动检测  
        imgsz=640,
        name="v5",  
        project="yolov11s_p2_segment_1cls",
        exist_ok=True, # 允许覆盖同名实验
        device=0,
        # optimizer="AdamW",  # COCO 大 epoch 训练中,AdamW 比默认的 SGD 收敛更快、更稳 
        cos_lr=True,  # 使用余弦退火学习率调度器
        warmup_epochs=3,  # 预热阶段的训练轮数
        close_mosaic=10,    # 关闭马赛克数据增强的训练轮数
        amp=False,  # 使用自动混合精度训练
        workers=16,  # CPU 核心足够多时可以更高
        resume=True,  
    )

def yolov11s_clean_v6():
    model = YOLO("ultralytics/cfg/models/11/yolo11s_p2.yaml")
    model.load("runs/detect/yolov11s_p2_detect_3cls/v5/weights/best.pt")
    model.train(
        data="ultralytics/cfg/datasets/clean_260730_detect.yaml",    
        epochs=50,
        batch=12,  # batch=-1 开启自动检测  
        imgsz=640,
        name="v6",  
        project="yolov11s_p2_detect_3cls",
        exist_ok=True, # 允许覆盖同名实验
        device=0,
        # optimizer="AdamW",  # COCO 大 epoch 训练中,AdamW 比默认的 SGD 收敛更快、更稳 
        cos_lr=True,  # 使用余弦退火学习率调度器
        warmup_epochs=3,  # 预热阶段的训练轮数
        close_mosaic=10,    # 关闭马赛克数据增强的训练轮数
        amp=False,  # 使用自动混合精度训练
        workers=16,  # CPU 核心足够多时可以更高
        resume=True,
    )

if __name__ == "__main__":
    # yolov11s_clean_v1()
    # yolov11s_clean_v2()
    # yolov11s_clean_v3()
    yolov11s_clean_v6()
