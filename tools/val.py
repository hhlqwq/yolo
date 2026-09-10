from ultralytics import YOLO


def val():
    # 用训练好的 best.pt 验证
    model = YOLO("runs/segment/p2_paper_zicai260721/v1_260721/weights/best.pt")
    # 先用小模型快速验证:YOLO("/data/users/hailong.he/gitee/ultralytics/model/yolo11s-seg.pt")

    results = model.val(
        data="ultralytics/cfg/datasets/clean_v2_seg_zicai0721.yaml",  
        imgsz=640,
        device=0,
        batch=4,
        workers=0,
    )
    print("B metrics:", {k: v for k, v in results.results_dict.items() if '(B)' in k})
    print("M metrics:", {k: v for k, v in results.results_dict.items() if '(M)' in k})


if __name__ == "__main__":
    val()
