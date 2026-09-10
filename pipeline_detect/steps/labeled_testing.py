"""外部带标签检测测试与精度报告汇总。"""

from __future__ import annotations

import csv
import io
import json
import shutil
from pathlib import Path
from typing import Any

from pipeline.core.errors import PipelineError
from pipeline.core.io_utils import atomic_write_text, remove_tree_safely
from pipeline.core.logging_utils import conda_python_command, run_subprocess
from pipeline.core.stage_runner import StageOutcome


IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


def discover_labeled_samples(
    root: Path,
    exclude_keywords: tuple[str, ...] = (),
) -> tuple[list[dict[str, str]], dict[str, int]]:
    """递归发现包含 images/labels 的有效测试场景并按目录关键字排除。"""
    samples: list[dict[str, str]] = []
    scene_counts: dict[str, int] = {}
    scene_name_counts: dict[str, int] = {}
    for image_dir in sorted(path for path in root.rglob("images") if path.is_dir()):
        scene_dir = image_dir.parent
        relative_scene = scene_dir.relative_to(root)
        if any(part.casefold().startswith("[deprecated]") for part in relative_scene.parts):
            continue
        if any(
            keyword.casefold() in part.casefold()
            for part in relative_scene.parts
            for keyword in exclude_keywords
        ):
            continue
        label_dir = scene_dir / "labels"
        if not label_dir.is_dir():
            continue
        matched_files: list[tuple[Path, Path]] = []
        for image in sorted(image_dir.iterdir()):
            if not image.is_file() or image.suffix.casefold() not in IMAGE_SUFFIXES:
                continue
            label = next(
                (
                    candidate
                    for suffix in (".txt", ".json")
                    if (candidate := label_dir / f"{image.stem}{suffix}").is_file()
                ),
                None,
            )
            if label is None:
                continue
            matched_files.append((image, label))
        if not matched_files:
            continue
        base_group = scene_dir.name
        occurrence = scene_name_counts.get(base_group, 0) + 1
        scene_name_counts[base_group] = occurrence
        group = base_group if occurrence == 1 else f"{base_group}__{occurrence}"
        for image, label in matched_files:
            samples.append(
                {
                    "id": image.stem,
                    "image": str(image.resolve()),
                    "detect_label": str(label.resolve()),
                    "group": group,
                }
            )
        scene_counts[group] = len(matched_files)
    return samples, scene_counts


def _material_name(group: str) -> str:
    """按场景目录名提取 paper/metal/liquid/mix 材质维度。"""
    scene_name = Path(group).name
    parts = scene_name.split("_", 2)
    return parts[1].casefold() if len(parts) >= 2 else "unknown"


def _confidence_name(confidence: float) -> str:
    """将置信度转换为稳定且可读的目录名。"""
    return f"confidence_{round(confidence * 100):02d}pct"


def _metric_row(
    confidence: float,
    scope_type: str,
    scope: str,
    material: str,
    class_id: str | int,
    values: dict[str, Any],
) -> dict[str, Any]:
    """将一组指标转换为 CSV 行。"""
    return {
        "置信度": confidence,
        "范围类型": scope_type,
        "范围": scope,
        "场景材质": material,
        "类别编号": class_id,
        "类别名称": values.get("name", "全部"),
        "GT": values.get("num_gt", values.get("total_gt", 0)),
        "Pred": values.get("num_pred", values.get("total_pred", 0)),
        "TP": values.get("tp", 0),
        "FP": values.get("fp", 0),
        "FN": values.get("fn", 0),
        "Precision": f"{float(values.get('precision', 0.0)):.3f}",
        "Recall": f"{float(values.get('recall', 0.0)):.3f}",
    }


def _class_rows(confidence: float, metrics: dict[str, Any]) -> list[dict[str, Any]]:
    """生成指定置信度下的总体类别指标行。"""
    return [
        _metric_row(confidence, "全部场景类别", "全部场景", "all", class_id, values)
        for class_id, values in sorted(metrics["per_class"].items(), key=lambda item: int(item[0]))
    ]


def _scene_rows(confidence: float, groups: dict[str, Any]) -> list[dict[str, Any]]:
    """生成指定置信度下每个测试场景一行的指标。"""
    rows: list[dict[str, Any]] = []
    for group, group_metrics in sorted(groups.items()):
        rows.append(
            _metric_row(
                confidence,
                "测试场景",
                group,
                _material_name(group),
                "",
                group_metrics["overall"],
            )
        )
    return rows


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    """以 UTF-8 BOM 写入可直接用 Excel 打开的 CSV。"""
    fields = [
        "置信度", "范围类型", "范围", "场景材质", "类别编号", "类别名称",
        "GT", "Pred", "TP", "FP", "FN", "Precision", "Recall",
    ]
    buffer = io.StringIO()
    writer = csv.DictWriter(buffer, fieldnames=fields)
    writer.writeheader()
    writer.writerows(rows)
    atomic_write_text(path, "\ufeff" + buffer.getvalue())


def _problem_summary_lines(results: list[dict[str, Any]], limit: int = 5) -> list[str]:
    """根据实际 P/R、FP 和 FN 生成可读的问题摘要。"""
    if not results:
        return ["## 数据问题总结", "", "本次测试没有可分析的指标。", ""]

    def f1_score(item: dict[str, Any]) -> float:
        """计算总体指标的 F1。"""
        precision = float(item.get("precision", 0.0))
        recall = float(item.get("recall", 0.0))
        return 2 * precision * recall / max(precision + recall, 1e-9)

    ordered = sorted(results, key=lambda item: float(item["confidence"]))
    best = max(ordered, key=lambda item: f1_score(item["metrics"]["overall"]))
    confidence = float(best["confidence"])
    overall = best["metrics"]["overall"]
    classes = [
        values
        for values in best["metrics"].get("per_class", {}).values()
        if int(values.get("num_gt", 0)) > 0
    ]
    groups = [
        (name, values["overall"])
        for name, values in best.get("groups", {}).items()
    ]
    lines = [
        "## 数据问题总结",
        "",
        (
            f"- 综合最平衡的置信度为 **{confidence:.0%}**："
            f"Precision={float(overall['precision']):.3%}，"
            f"Recall={float(overall['recall']):.3%}。"
        ),
    ]
    if classes:
        weakest_precision = min(classes, key=lambda item: float(item["precision"]))
        weakest_recall = min(classes, key=lambda item: float(item["recall"]))
        lines.extend(
            [
                (
                    f"- Precision 最弱类别：**{weakest_precision['name']}** "
                    f"({float(weakest_precision['precision']):.3%})。"
                ),
                (
                    f"- Recall 最弱类别：**{weakest_recall['name']}** "
                    f"({float(weakest_recall['recall']):.3%})。"
                ),
            ]
        )
    top_false_positive = sorted(groups, key=lambda item: int(item[1].get("fp", 0)), reverse=True)[:limit]
    top_false_negative = sorted(groups, key=lambda item: int(item[1].get("fn", 0)), reverse=True)[:limit]
    if top_false_positive and int(top_false_positive[0][1].get("fp", 0)) > 0:
        values = "；".join(f"{name}={int(metrics['fp'])}" for name, metrics in top_false_positive)
        lines.append(f"- FP 较多的场景：{values}。")
    if top_false_negative and int(top_false_negative[0][1].get("fn", 0)) > 0:
        values = "；".join(f"{name}={int(metrics['fn'])}" for name, metrics in top_false_negative)
        lines.append(f"- FN 较多的场景：{values}。")
    if len(ordered) > 1:
        low = ordered[0]
        high = ordered[-1]
        low_overall = low["metrics"]["overall"]
        high_overall = high["metrics"]["overall"]
        lines.append(
            f"- 从 {float(low['confidence']):.0%} 提高到 {float(high['confidence']):.0%} 时，"
            f"Precision 从 {float(low_overall['precision']):.3%} 变为 "
            f"{float(high_overall['precision']):.3%}，Recall 从 "
            f"{float(low_overall['recall']):.3%} 变为 {float(high_overall['recall']):.3%}。"
        )
    lines.append("")
    return lines


def _write_markdown(path: Path, results: list[dict[str, Any]]) -> None:
    """生成以每类别 P/R 为中心的多置信度摘要。"""
    lines = [
        "# 外部带标签测试精度报告",
        "",
        "> P/R 按配置置信度和 IoU=0.5 固定匹配计算。",
        "",
        "## 整体指标",
        "",
        "| 置信度 | TP | FP | FN | Precision | Recall |",
        "|---:|---:|---:|---:|---:|---:|",
    ]
    for result in results:
        confidence = float(result["confidence"])
        overall = result["metrics"]["overall"]
        lines.append(
            f"| {confidence:.0%} | {overall['tp']} | {overall['fp']} | {overall['fn']} | "
            f"{overall['precision']:.3%} | {overall['recall']:.3%} |"
        )
    lines.extend(
        [
            "",
            "## 各类别 Precision / Recall",
            "",
            "| 置信度 | 类别 | GT | Pred | TP | FP | FN | Precision | Recall |",
            "|---:|---|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for result in results:
        confidence = float(result["confidence"])
        for values in result["metrics"]["per_class"].values():
            lines.append(
                f"| {confidence:.0%} | {values['name']} | {values['num_gt']} | "
                f"{values['num_pred']} | {values['tp']} | {values['fp']} | "
                f"{values['fn']} | {values['precision']:.3%} | {values['recall']:.3%} |"
            )
    lines.extend(["", "逐场景明细见各置信度目录中的场景精度报告。", ""])
    lines.extend(_problem_summary_lines(results))
    atomic_write_text(path, "\n".join(lines))


def _write_overall_markdown(path: Path, results: list[dict[str, Any]]) -> None:
    """生成模型目录使用的精简总体测试摘要。"""
    lines = [
        "# 模型测试精度摘要",
        "",
        "| 置信度 | TP | FP | FN | Precision | Recall |",
        "|---:|---:|---:|---:|---:|---:|",
    ]
    for result in results:
        confidence = float(result["confidence"])
        overall = result["metrics"]["overall"]
        lines.append(
            f"| {confidence:.0%} | {overall['tp']} | {overall['fp']} | {overall['fn']} | "
            f"{overall['precision']:.3%} | {overall['recall']:.3%} |"
        )
    lines.append("")
    lines.extend(_problem_summary_lines(results))
    atomic_write_text(path, "\n".join(lines))


def _reset_result_root(context: Any, result_root: Path) -> None:
    """安全清空本次运行的外部测试结果目录。"""
    resolved = result_root.resolve()
    configured = context.config.test_output_dir.resolve()
    if resolved != configured or resolved.name != context.config.run_name:
        raise PipelineError(f"拒绝清理非预期测试结果目录:{resolved}")

    def log_retry(path: Path, attempt: int, retries: int, reason: str) -> None:
        """记录 NAS/SMB 结果目录清理重试."""
        context.logger.warning(
            "RESULT_RESET_RETRY | path=%s | attempt=%d/%d | reason=%s",
            path,
            attempt,
            retries,
            reason,
        )

    try:
        retained_metadata = remove_tree_safely(resolved, retry_callback=log_retry)
    except OSError as exc:
        raise PipelineError(
            f"测试结果目录持续无法清理，可能仍有其他进程写入，请检查后重试:{resolved}"
        ) from exc
    if retained_metadata:
        context.logger.warning(
            "RESULT_METADATA_RETAINED | path=%s | files=%d | reason=resource_busy",
            resolved,
            len(retained_metadata),
        )


def _mirror_key_reports(context: Any, result_root: Path, results: list[dict[str, Any]]) -> Path:
    """将精简总体报告镜像到模型版本目录。"""
    mirror_root = context.work_dir / "result"
    resolved = mirror_root.resolve()
    if resolved.parent != context.work_dir.resolve() or resolved.name != "result":
        raise PipelineError(f"拒绝写入非预期报告镜像目录:{resolved}")
    if resolved.is_dir():
        shutil.rmtree(resolved)
    resolved.mkdir(parents=True)
    shutil.copy2(
        result_root / "overall_accuracy_summary.csv",
        resolved / "test_summary.csv",
    )
    _write_overall_markdown(resolved / "test_summary.md", results)
    return resolved


def run_labeled_detect_test(context: Any, onnx_path: Path) -> StageOutcome:
    """运行外部测试集多置信度推理并生成实用型汇总报告。"""
    root = context.config.labeled_test_dir
    result_root = context.config.test_output_dir
    _reset_result_root(context, result_root)
    samples, scene_counts = discover_labeled_samples(
        root,
        context.config.labeled_test_exclude_keywords,
    )
    if not samples:
        raise PipelineError(f"外部测试集没有找到 images/labels 配对图片:{root}")
    manifest = result_root / "test_samples.json"
    atomic_write_text(
        manifest,
        json.dumps(
            {
                "root": str(root.resolve()),
                "scene_count": len(scene_counts),
                "image_count": len(samples),
                "scene_counts": scene_counts,
                "samples": samples,
            },
            ensure_ascii=False,
            indent=2,
        ),
    )
    results: list[dict[str, Any]] = []
    overall_rows: list[dict[str, Any]] = []
    class_rows: list[dict[str, Any]] = []
    inference_script = context.config.repo_root / "tools" / "inference.py"
    dataset_yaml = context.work_dir / "training_configs" / "yolo_detect.yaml"
    try:
        for confidence in context.config.labeled_test_confidences:
            output_dir = result_root / _confidence_name(confidence)
            metrics_path = output_dir / "metrics.json"
            scene_metrics_path = output_dir / "scene_metrics.json"
            command = conda_python_command(
                context.config.yolo_env,
                [
                    str(inference_script), "--manifest", str(manifest),
                    "--onnx-model", str(onnx_path), "--yaml", str(dataset_yaml),
                    "--output-dir", str(output_dir), "--imgsz",
                    *map(str, context.config.requested_image_size),
                    "--conf", str(confidence), "--iou", str(context.config.test_iou),
                ],
            )
            try:
                run_subprocess(
                    command,
                    context.config.repo_root,
                    f"外部带标签检测测试 confidence={confidence:.2f}",
                    context.logger,
                    key_line_filter=lambda line: "Summary" in line or "Metrics" in line,
                )
                if not metrics_path.is_file() or not scene_metrics_path.is_file():
                    raise PipelineError(f"外部测试未生成完整指标文件:{output_dir}")
                metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
                groups = json.loads(scene_metrics_path.read_text(encoding="utf-8"))
                percent = round(confidence * 100)
                _write_csv(
                    output_dir / f"scene_accuracy_report_{percent:02d}pct.csv",
                    _scene_rows(confidence, groups),
                )
                overall_rows.append(
                    _metric_row(
                        confidence,
                        "全部场景",
                        "全部场景",
                        "all",
                        "",
                        metrics["overall"],
                    )
                )
                class_rows.extend(_class_rows(confidence, metrics))
                results.append({"confidence": confidence, "metrics": metrics, "groups": groups})
            finally:
                metrics_path.unlink(missing_ok=True)
                scene_metrics_path.unlink(missing_ok=True)
    finally:
        manifest.unlink(missing_ok=True)
    report_csv = result_root / "overall_accuracy_summary.csv"
    class_report_csv = result_root / "class_accuracy_report.csv"
    report_markdown = result_root / "test_summary.md"
    _write_csv(report_csv, overall_rows)
    _write_csv(class_report_csv, class_rows)
    _write_markdown(report_markdown, results)
    report_mirror = _mirror_key_reports(context, result_root, results)
    return StageOutcome(
        message=(
            f"外部带标签测试完成:scenes={len(scene_counts)},images={len(samples)},"
            f"confidences={len(results)}"
        ),
        artifacts={
            "result_dir": str(result_root),
            "report_csv": str(report_csv),
            "class_report_csv": str(class_report_csv),
            "report_markdown": str(report_markdown),
            "report_mirror": str(report_mirror),
        },
        metrics={
            "scenes": len(scene_counts),
            "images": len(samples),
            "confidences": list(context.config.labeled_test_confidences),
        },
        value=results,
    )
