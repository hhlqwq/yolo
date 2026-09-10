#!/usr/bin/env python3
"""在 PIDNet Conda 环境中推理清单并保存临时单通道 mask."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
from torch.nn import functional as F


MEAN = np.asarray([0.485, 0.456, 0.406], dtype=np.float32)
STD = np.asarray([0.229, 0.224, 0.225], dtype=np.float32)


def parse_args() -> argparse.Namespace:
    """解析仅由主 pipeline 传入的内部参数."""
    parser = argparse.ArgumentParser(description="PIDNet预标注内部推理器.")
    parser.add_argument("--pidnet-root", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    return parser.parse_args()


def read_image(path: Path) -> np.ndarray:
    """使用兼容中文路径的方式读取图片."""
    image = cv2.imdecode(np.fromfile(path, dtype=np.uint8), cv2.IMREAD_COLOR)
    if image is None:
        raise RuntimeError(f"无法读取图片:{path}")
    return image


def preprocess(image: np.ndarray, target_h: int, target_w: int) -> tuple[torch.Tensor, tuple[int, int, int, int]]:
    """等比例缩放并 letterbox 到 PIDNet 输入尺寸."""
    height, width = image.shape[:2]
    scale = min(target_h / height, target_w / width)
    new_h, new_w = int(round(height * scale)), int(round(width * scale))
    resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    top, left = (target_h - new_h) // 2, (target_w - new_w) // 2
    padded = cv2.copyMakeBorder(
        resized,
        top,
        target_h - new_h - top,
        left,
        target_w - new_w - left,
        cv2.BORDER_CONSTANT,
        value=(0, 0, 0),
    )
    rgb = padded[:, :, ::-1].astype(np.float32) / 255.0
    normalized = (rgb - MEAN) / STD
    tensor = torch.from_numpy(normalized.transpose(2, 0, 1).copy()).unsqueeze(0)
    return tensor, (top, left, new_h, new_w)


def load_model(pidnet_root: Path, config_path: Path, weights_path: Path, device: torch.device):
    """加载 PIDNet 配置、网络和 best.pt 参数."""
    sys.path.insert(0, str(pidnet_root))
    from configs import config
    import models

    config.defrost()
    config.merge_from_file(str(config_path))
    config.freeze()
    model = models.pidnet.get_pred_model(config.MODEL.NAME, config.DATASET.NUM_CLASSES)
    state = torch.load(weights_path, map_location="cpu", weights_only=False)
    if isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]
    if not isinstance(state, dict):
        raise RuntimeError(f"PIDNet权重不是有效state_dict:{weights_path}")
    model_state = model.state_dict()
    loaded = {}
    for key, value in state.items():
        candidates = [key]
        candidates.extend(
            key[len(prefix):]
            for prefix in ("module.model.", "module.", "model.")
            if key.startswith(prefix)
        )
        for candidate in candidates:
            if candidate in model_state and model_state[candidate].shape == value.shape:
                loaded[candidate] = value
                break
    if not loaded:
        raise RuntimeError(f"PIDNet权重与模型结构不匹配:{weights_path}")
    model_state.update(loaded)
    model.load_state_dict(model_state, strict=False)
    model.to(device).eval()
    return model, config, len(loaded)


def save_mask(path: Path, mask: np.ndarray) -> None:
    """使用兼容中文路径的方式原子保存 PNG mask."""
    path.parent.mkdir(parents=True, exist_ok=True)
    ok, encoded = cv2.imencode(".png", mask)
    if not ok:
        raise RuntimeError(f"PIDNet mask编码失败:{path}")
    temporary = path.with_suffix(".png.tmp")
    encoded.tofile(temporary)
    temporary.replace(path)


def main() -> int:
    """执行清单中的 PIDNet 推理."""
    args = parse_args()
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    entries = json.loads(args.manifest.read_text(encoding="utf-8"))
    if not isinstance(entries, list):
        raise RuntimeError(f"PIDNet推理清单必须是列表:{args.manifest}")
    model, config, loaded = load_model(args.pidnet_root, args.config, args.weights, device)
    target_w, target_h = (int(value) for value in config.TEST.IMAGE_SIZE)
    print(f"PIPELINE_PIDNET_PRELABEL_INIT loaded={loaded} device={device} images={len(entries)}")
    with torch.inference_mode():
        for index, entry in enumerate(entries, start=1):
            image_path, mask_path = Path(entry["image"]), Path(entry["mask"])
            image = read_image(image_path)
            height, width = image.shape[:2]
            tensor, (top, left, new_h, new_w) = preprocess(image, target_h, target_w)
            output = model(tensor.to(device))
            if isinstance(output, (list, tuple)):
                output = output[int(config.TEST.OUTPUT_INDEX)]
            logits = F.interpolate(
                output,
                size=(target_h, target_w),
                mode="bilinear",
                align_corners=config.MODEL.ALIGN_CORNERS,
            )
            mask = torch.argmax(logits, dim=1).squeeze(0).byte().cpu().numpy()
            mask = mask[top:top + new_h, left:left + new_w]
            mask = cv2.resize(mask, (width, height), interpolation=cv2.INTER_NEAREST)
            save_mask(mask_path, mask)
            if index % 100 == 0 or index == len(entries):
                print(f"PIPELINE_PIDNET_PRELABEL_PROGRESS {index}/{len(entries)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
