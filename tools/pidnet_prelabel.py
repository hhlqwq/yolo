#!/usr/bin/env python3
"""在 PIDNet Conda 环境中逐张推理图片,并保存单通道类别 mask."""

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
    parser = argparse.ArgumentParser(description="PIDNet 预标注 mask 生成器(由 data_pipeline.py 内部调用).")
    parser.add_argument("--pidnet-root", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    return parser.parse_args()


def read_image(path: Path) -> np.ndarray:
    image = cv2.imdecode(np.fromfile(path, dtype=np.uint8), cv2.IMREAD_COLOR)
    if image is None:
        raise RuntimeError(f"无法读取图片:{path}")
    return image


def preprocess(image: np.ndarray, target_h: int, target_w: int) -> tuple[torch.Tensor, tuple[int, int, int, int]]:
    height, width = image.shape[:2]
    scale = min(target_h / height, target_w / width)
    new_h = int(round(height * scale))
    new_w = int(round(width * scale))
    resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    pad_h = target_h - new_h
    pad_w = target_w - new_w
    top = pad_h // 2
    left = pad_w // 2
    padded = cv2.copyMakeBorder(
        resized,
        top,
        pad_h - top,
        left,
        pad_w - left,
        cv2.BORDER_CONSTANT,
        value=(0, 0, 0),
    )
    rgb = padded[:, :, ::-1].astype(np.float32) / 255.0
    normalized = (rgb - MEAN) / STD
    tensor = torch.from_numpy(normalized.transpose(2, 0, 1).copy()).unsqueeze(0)
    return tensor, (top, left, new_h, new_w)


def load_model(pidnet_root: Path, config_path: Path, weights_path: Path, device: torch.device):
    sys.path.insert(0, str(pidnet_root))
    from configs import config
    import models

    config.defrost()
    config.merge_from_file(str(config_path))
    config.freeze()

    model = models.pidnet.get_pred_model(
        name=config.MODEL.NAME,
        num_classes=config.DATASET.NUM_CLASSES,
    )
    state = torch.load(weights_path, map_location="cpu")
    if isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]
    if not isinstance(state, dict):
        raise RuntimeError(f"PIDNet 权重不是有效 state_dict:{weights_path}")

    model_state = model.state_dict()
    loaded: dict[str, torch.Tensor] = {}
    prefixes = ("module.model.", "module.", "model.")
    for key, value in state.items():
        candidates = [key]
        candidates.extend(key[len(prefix) :] for prefix in prefixes if key.startswith(prefix))
        for candidate in candidates:
            if candidate in model_state and model_state[candidate].shape == value.shape:
                loaded[candidate] = value
                break
    if not loaded:
        raise RuntimeError(f"PIDNet 权重与模型结构不匹配,未加载任何参数:{weights_path}")

    model_state.update(loaded)
    model.load_state_dict(model_state, strict=False)
    model.to(device).eval()
    return model, config, len(loaded)


def main() -> int:
    args = parse_args()
    pidnet_root = args.pidnet_root.expanduser().resolve()
    config_path = args.config.expanduser().resolve()
    weights_path = args.weights.expanduser().resolve()
    manifest_path = args.manifest.expanduser().resolve()
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    entries = json.loads(manifest_path.read_text(encoding="utf-8"))
    if not isinstance(entries, list):
        raise RuntimeError(f"PIDNet 推理清单必须是列表:{manifest_path}")

    model, config, loaded_count = load_model(pidnet_root, config_path, weights_path, device)
    target_w, target_h = (int(value) for value in config.TEST.IMAGE_SIZE)
    print(f"[PIDNet prelabel] 已加载 {loaded_count} 个参数,设备={device},图片={len(entries)}")

    with torch.inference_mode():
        for index, entry in enumerate(entries, start=1):
            image_path = Path(entry["image"])
            mask_path = Path(entry["mask"])
            image = read_image(image_path)
            original_shape = image.shape[:2]
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
            mask = mask[top : top + new_h, left : left + new_w]
            mask = cv2.resize(mask, (original_shape[1], original_shape[0]), interpolation=cv2.INTER_NEAREST)
            mask_path.parent.mkdir(parents=True, exist_ok=True)
            ok, encoded = cv2.imencode(".png", mask)
            if not ok:
                raise RuntimeError(f"PIDNet mask 编码失败:{mask_path}")
            encoded.tofile(mask_path)
            if index % 100 == 0 or index == len(entries):
                print(f"[PIDNet prelabel] {index}/{len(entries)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
