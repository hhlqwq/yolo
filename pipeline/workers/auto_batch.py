#!/usr/bin/env python3
"""在模型所属 Conda 环境中实际探测安全训练 Batch."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from types import SimpleNamespace


RESULT_PREFIX = "PIPELINE_AUTO_BATCH_RESULT="
TARGET_MEMORY_FRACTION = 0.80


def parse_args() -> argparse.Namespace:
    """解析自动 Batch 探测参数."""
    parser = argparse.ArgumentParser(description="探测三模型安全训练Batch.")
    parser.add_argument("--task", choices=("detect", "segment", "pidnet"), required=True)
    parser.add_argument("--device", type=int, required=True)
    parser.add_argument("--max-batch", type=int, required=True)
    parser.add_argument("--imgsz", type=int, nargs=2, metavar=("HEIGHT", "WIDTH"), required=True)
    parser.add_argument("--model", type=Path)
    parser.add_argument("--weight", type=Path)
    parser.add_argument("--config", type=Path)
    parser.add_argument("--pidnet-root", type=Path)
    parser.add_argument("--amp", action="store_true")
    return parser.parse_args()


def _gpu_info(torch_module, device) -> dict[str, float | str]:
    """返回 PyTorch 视角的 GPU 信息."""
    free, total = torch_module.cuda.mem_get_info(device)
    return {
        "name": torch_module.cuda.get_device_name(device),
        "total_gb": total / 2**30,
        "free_gb": free / 2**30,
    }


def probe_yolo(args: argparse.Namespace) -> dict[str, object]:
    """调用 Ultralytics AutoBatch 探测 YOLO 并限制最大 Batch."""
    import torch
    from ultralytics import YOLO
    from ultralytics.utils.autobatch import check_train_batch_size

    if args.model is None:
        raise ValueError("YOLO AutoBatch缺少--model.")
    device = torch.device(f"cuda:{args.device}")
    torch.cuda.set_device(device)
    model = YOLO(str(args.model))
    if args.weight is not None:
        model.load(str(args.weight))
    network = model.model.to(device)
    detected = check_train_batch_size(
        network,
        imgsz=args.imgsz,
        amp=args.amp,
        batch=TARGET_MEMORY_FRACTION,
        max_num_obj=128,
    )
    selected = max(1, min(args.max_batch, int(detected)))
    result = {
        "task": args.task,
        "detected_batch": int(detected),
        "selected_batch": selected,
        "target_memory_fraction": TARGET_MEMORY_FRACTION,
        "gpu": _gpu_info(torch, device),
    }
    del network, model
    torch.cuda.empty_cache()
    return result


def _pidnet_batch_fits(
    model,
    optimizer,
    torch_module,
    device,
    batch: int,
    imgsz: list[int],
    classes: int,
) -> tuple[bool, float | None]:
    """执行完整训练步骤,判断峰值总显存是否不超过 80%."""
    optimizer.zero_grad(set_to_none=True)
    images = labels = boundaries = loss = None
    peak_fraction: float | None = None
    try:
        torch_module.cuda.empty_cache()
        free_before, total = torch_module.cuda.mem_get_info(device)
        reserved_before = torch_module.cuda.memory_reserved(device)
        external_used = max(0, total - free_before - reserved_before)
        torch_module.cuda.reset_peak_memory_stats(device)
        height, width = imgsz
        images = torch_module.randn(batch, 3, height, width, device=device)
        labels = torch_module.randint(0, classes, (batch, height, width), device=device)
        boundaries = torch_module.randint(0, 2, (batch, height, width), device=device).float()
        loss, _, _, _ = model(images, labels, boundaries)
        loss.mean().backward()
        optimizer.step()
        torch_module.cuda.synchronize(device)
        peak_reserved = torch_module.cuda.max_memory_reserved(device)
        peak_fraction = (external_used + peak_reserved) / total
        return peak_fraction <= TARGET_MEMORY_FRACTION, peak_fraction
    except RuntimeError as exc:
        if isinstance(exc, torch_module.cuda.OutOfMemoryError) or "out of memory" in str(exc).casefold():
            return False, peak_fraction
        raise
    finally:
        optimizer.zero_grad(set_to_none=True)
        del images, labels, boundaries, loss
        torch_module.cuda.empty_cache()


def probe_pidnet(args: argparse.Namespace) -> dict[str, object]:
    """寻找 PIDNet 峰值总显存不超过 80% 的最大 Batch."""
    if args.config is None or args.pidnet_root is None:
        raise ValueError("PIDNet AutoBatch缺少--config或--pidnet-root.")
    sys.path.insert(0, str(args.pidnet_root.resolve()))

    import torch
    from configs import config, update_config
    from models import pidnet
    from utils.criterion import BondaryLoss, OhemCrossEntropy
    from utils.utils import FullModel

    options: list[str] = []
    if args.weight is not None:
        options.extend(["MODEL.PRETRAINED", str(args.weight)])
    update_config(config, SimpleNamespace(cfg=str(args.config), opts=options))
    device = torch.device(f"cuda:{args.device}")
    torch.cuda.set_device(device)
    torch.backends.cudnn.benchmark = False
    network = pidnet.get_seg_model(config, imgnet_pretrained="imagenet" in config.MODEL.PRETRAINED)
    sem_loss = OhemCrossEntropy(
        ignore_label=config.TRAIN.IGNORE_LABEL,
        thres=config.LOSS.OHEMTHRES,
        min_kept=config.LOSS.OHEMKEEP,
    )
    model = FullModel(network, sem_loss, BondaryLoss()).to(device).train()
    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=config.TRAIN.LR,
        momentum=config.TRAIN.MOMENTUM,
        weight_decay=config.TRAIN.WD,
        nesterov=config.TRAIN.NESTEROV,
    )

    tested_fractions: dict[int, float | None] = {}

    def fits(candidate: int) -> bool:
        """探测一个候选 Batch 并记录峰值显存比例."""
        accepted, peak_fraction = _pidnet_batch_fits(
            model,
            optimizer,
            torch,
            device,
            candidate,
            args.imgsz,
            config.DATASET.NUM_CLASSES,
        )
        tested_fractions[candidate] = peak_fraction
        return accepted

    if fits(args.max_batch):
        largest = args.max_batch
    else:
        largest = 0
        low, high = 1, args.max_batch - 1
        while low <= high:
            candidate = (low + high) // 2
            if fits(candidate):
                largest = candidate
                low = candidate + 1
            else:
                high = candidate - 1
    if largest < 1:
        raise RuntimeError("PIDNet在batch=1时峰值总显存仍超过80%或发生显存不足.")
    selected = largest
    result = {
        "task": args.task,
        "detected_batch": largest,
        "selected_batch": selected,
        "target_memory_fraction": TARGET_MEMORY_FRACTION,
        "selected_peak_memory_fraction": tested_fractions.get(selected),
        "tested_peak_memory_fractions": tested_fractions,
        "gpu": _gpu_info(torch, device),
    }
    del optimizer, model, network
    torch.cuda.empty_cache()
    return result


def main() -> int:
    """执行对应模型探测并输出机器可解析的单行 JSON."""
    args = parse_args()
    if args.max_batch < 1 or min(args.imgsz) < 1:
        raise ValueError("--max-batch和--imgsz必须大于0.")
    result = probe_pidnet(args) if args.task == "pidnet" else probe_yolo(args)
    print(f"{RESULT_PREFIX}{json.dumps(result, ensure_ascii=False)}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
