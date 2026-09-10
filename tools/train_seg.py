from ultralytics import YOLO


def yolov11s_seg_clean_v1():
    model = YOLO("/data/users/hailong.he/gitee/ultralytics/model/yolo11s-seg.pt")

    model.train(
        data="clean_v1_seg.yaml",           
        epochs=5,
        batch=20,  # batch=-1 开启自动检测  
        imgsz=640,
        name="clean_seg_v1",  
        project="11s-seg_test",
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

    # 执行预测
    results = model.predict(
        source="/data/users/hailong.he/datasets/clean/SteelDS/test/images",  
        save=True,             # 自动保存带标注的图片到 'runs/segment/predict/'[reference:2]
        imgsz=640,             # 输入图片尺寸,与训练时保持一致
        conf=0.25,             # 置信度阈值,可调整
        device=0,              # 使用GPU,CPU则用 device='cpu'
    )
    results = model.predict(
        source="/data/users/hailong.he/datasets/clean/BUU-WOD/test/images",  
        save=True,             # 自动保存带标注的图片到 'runs/segment/predict/'[reference:2]
        imgsz=640,             # 输入图片尺寸,与训练时保持一致
        conf=0.25,             # 置信度阈值,可调整
        device=0,              # 使用GPU,CPU则用 device='cpu'
    )


def yolov11s_seg_clean_p2_v1():
    model = YOLO("ultralytics/cfg/models/11/yolo11-seg_p2.yaml")
    model.train(
        data="ultralytics/cfg/datasets/clean_v1_seg.yaml", 
        # pretrained="runs/segment/11s-seg/clean_seg_v1/weights/best.pt",     
        epochs=150,
        batch=16,  # batch=-1 开启自动检测  
        imgsz=640,
        name="clean_seg_p2_v1",  
        project="seg_test_p2",
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

    # 执行预测
    results = model.predict(
        source="/data/users/hailong.he/datasets/clean/SteelDS/test/images",  
        save=True,             # 自动保存带标注的图片到 'runs/segment/predict/'[reference:2]
        imgsz=640,             # 输入图片尺寸,与训练时保持一致
        conf=0.25,             # 置信度阈值,可调整
        device=0,              # 使用GPU,CPU则用 device='cpu'
    )
    results = model.predict(
        source="/data/users/hailong.he/datasets/clean/BUU-WOD/test/images",  
        save=True,             # 自动保存带标注的图片到 'runs/segment/predict/'[reference:2]
        imgsz=640,             # 输入图片尺寸,与训练时保持一致
        conf=0.25,             # 置信度阈值,可调整
        device=0,              # 使用GPU,CPU则用 device='cpu'
    )


def yolov11s_seg_clean_p2_v2():
    model = YOLO("ultralytics/cfg/models/11/yolo11-seg_p2.yaml")
    model.load("runs/segment/seg_test_p2/clean_seg_p2_v1/weights/best.pt")
    model.train(
        data="ultralytics/cfg/datasets/clean_v2_seg.yaml", 
        # pretrained="runs/segment/11s-seg/clean_seg_v1/weights/best.pt",     
        epochs=200,
        batch=16,  # batch=-1 开启自动检测  
        imgsz=640,
        name="clean_seg_p2_v2",  
        project="seg_test_p2",
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

    # 执行预测
    model.predict(
        source="/data/users/hailong.he/datasets/clean/shunyu260717/val/images/",  
        save=True,             # 自动保存带标注的图片到 'runs/segment/predict/'[reference:2]
        imgsz=640,             # 输入图片尺寸,与训练时保持一致
        conf=0.25,             # 置信度阈值,可调整
        device=0,              # 使用GPU,CPU则用 device='cpu'
    )


def yolov11s_seg_clean_p2_v3():
    model = YOLO("ultralytics/cfg/models/11/yolo11-seg_p2.yaml")
    model.load("runs/segment/seg_test_p2/clean_seg_p2_v2/weights/best.pt")
    model.train(
        data="ultralytics/cfg/datasets/clean_v2_seg_zicai.yaml", 
        # pretrained="runs/segment/11s-seg/clean_seg_v1/weights/best.pt",     
        epochs=200,
        batch=8,  # batch=-1 开启自动检测  
        imgsz=640,
        name="v1_260720",  
        project="p2_paper_zicai260720",
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

    # 执行预测
    model.predict(
        source="/data/users/hailong.he/datasets/clean/zicai260720/val/images/",  
        save=True,             # 自动保存带标注的图片到 'runs/segment/predict/'[reference:2]
        imgsz=640,             # 输入图片尺寸,与训练时保持一致
        conf=0.25,             # 置信度阈值,可调整
        device=0,              # 使用GPU,CPU则用 device='cpu'
    )

def yolov11s_seg_clean_p2_v4_load():
    model = YOLO("ultralytics/cfg/models/11/yolo11-seg_p2.yaml")
    model.load("runs/segment/p2_paper_zicai260720/v1_260720-3/weights/best.pt")
    model.train(
        data="ultralytics/cfg/datasets/clean_v2_seg_zicai0721.yaml", 
        # pretrained="runs/segment/11s-seg/clean_seg_v1/weights/best.pt",     
        epochs=300,
        batch=8,  # batch=-1 开启自动检测  
        imgsz=640,
        name="v1_260721_load",  
        project="p2_paper_zicai260721",
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
def yolov11s_seg_clean_p2_v4():
    model = YOLO("ultralytics/cfg/models/11/yolo11-seg_p2.yaml")
    # model.load("runs/segment/p2_paper_zicai260720/v1_260720-3/weights/best.pt")
    model.train(
        data="ultralytics/cfg/datasets/clean_v2_seg_zicai0721.yaml", 
        # pretrained="runs/segment/11s-seg/clean_seg_v1/weights/best.pt",     
        epochs=300,
        batch=8,  # batch=-1 开启自动检测  
        imgsz=640,
        name="v1_260721",  
        project="p2_paper_zicai260721",
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


def yolov11s_seg_clean_v5():
    model = YOLO("ultralytics/cfg/models/11/yolo11s-seg.yaml")
    # model.load("runs/segment/p2_paper_zicai260720/v1_260720-3/weights/best.pt")
    model.train(
        data="ultralytics/cfg/datasets/clean_v2_seg_zicai0721.yaml", 
        # pretrained="runs/segment/11s-seg/clean_seg_v1/weights/best.pt",     
        epochs=300,
        batch=-1,  # batch=-1 开启自动检测  
        imgsz=640,
        name="v1_260722",  
        project="paper_zicai260721",
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

def yolov11s_seg_clean_v6():
    model = YOLO("ultralytics/cfg/models/11/yolo11s-seg.yaml")
    model.load("runs/segment/paper_zicai260721/v1_260722/weights/best.pt")
    model.train(
        data="ultralytics/cfg/datasets/clean_v2_seg_zicai0721.yaml", 
        # pretrained="runs/segment/11s-seg/clean_seg_v1/weights/best.pt",     
        epochs=300,
        batch=-1,  # batch=-1 开启自动检测  
        imgsz=640,
        name="v1",  
        project="seg_loss",
        exist_ok=False, # 允许覆盖同名实验
        device=0,
        # optimizer="AdamW",  # COCO 大 epoch 训练中,AdamW 比默认的 SGD 收敛更快、更稳 
        cos_lr=True,  # 使用余弦退火学习率调度器
        warmup_epochs=3,  # 预热阶段的训练轮数
        close_mosaic=10,    # 关闭马赛克数据增强的训练轮数
        amp=False,  # 使用自动混合精度训练
        workers=16,  # CPU 核心足够多时可以更高
        resume=True,
        seg=10,
    )


if __name__ == "__main__":
    # yolov11s_seg_clean_v1()
    # yolov11s_seg_clean_p2_v1()
    
    # 用户的数据  纸单类
    # yolov11s_seg_clean_p2_v2()
    
    # 用户的数据+自采的数据 
    # yolov11s_seg_clean_p2_v3()

    # 用户的数据+自采的数据 0720+0721
    # yolov11s_seg_clean_p2_v4()
    # yolov11s_seg_clean_p2_v4_load()
    
    # # 去掉p2头
    # yolov11s_seg_clean_v5()

    # 增加分割损失
    yolov11s_seg_clean_v6()