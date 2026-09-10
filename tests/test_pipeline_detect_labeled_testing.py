"""纯检测外部带标签测试单元测试。"""

from __future__ import annotations

import json
import os
import tempfile
import unittest
from pathlib import Path

import numpy as np

from pipeline.workers.onnx_detect import (
    evaluate_predictions,
    load_gt_label,
    load_manifest_pairs,
    make_vertical_three_view,
)
from pipeline_detect.steps.labeled_testing import (
    _metric_row,
    _mirror_key_reports,
    _problem_summary_lines,
    _reset_result_root,
    _scene_rows,
    discover_labeled_samples,
)


def _prediction(boxes: list[list[float]], classes: list[int]) -> dict:
    """构造评估函数使用的预测结构。"""
    return {
        "boxes": np.asarray(boxes, dtype=np.float32),
        "scores": np.asarray([0.9] * len(boxes), dtype=np.float32),
        "classes": np.asarray(classes, dtype=np.int64),
    }


def _ground_truth(boxes: list[list[float]], classes: list[int]) -> dict:
    """构造评估函数使用的标签结构。"""
    return {
        "boxes": np.asarray(boxes, dtype=np.float32),
        "classes": np.asarray(classes, dtype=np.int64),
    }


class LabeledTestingTest(unittest.TestCase):
    """验证外部带标签测试的发现、分组和指标行为。"""

    def _temporary_directory(self) -> tempfile.TemporaryDirectory[str]:
        """在用户指定的测试临时根目录创建自动清理目录。"""
        base_dir = os.environ.get("HAILONGCODEX_TEST_TMP")
        if base_dir:
            Path(base_dir).mkdir(parents=True, exist_ok=True)
        return tempfile.TemporaryDirectory(dir=base_dir)

    def test_evaluate_predictions_uses_fixed_confidence_counts(self) -> None:
        """P/R 应使用已过置信度过滤的全部预测计算。"""
        predictions = [
            _prediction(
                [[0, 0, 10, 10], [30, 30, 40, 40], [50, 50, 60, 60]],
                [0, 0, 1],
            )
        ]
        ground_truth = [
            _ground_truth([[0, 0, 10, 10], [15, 15, 25, 25]], [0, 0])
        ]

        metrics = evaluate_predictions(
            predictions,
            ground_truth,
            {0: "paper", 1: "liquid"},
            2,
        )

        self.assertEqual(metrics["per_class"][0]["tp"], 1)
        self.assertEqual(metrics["per_class"][0]["fp"], 1)
        self.assertEqual(metrics["per_class"][0]["fn"], 1)
        self.assertEqual(metrics["per_class"][1]["fp"], 1)
        self.assertEqual(metrics["overall"]["precision"], 1 / 3)
        self.assertEqual(metrics["overall"]["recall"], 1 / 2)

    def test_make_vertical_three_view_adds_three_headers(self) -> None:
        """竖向三视图应包含三张原图高度和三个标题栏。"""
        image = np.zeros((20, 30, 3), dtype=np.uint8)

        result = make_vertical_three_view(image, image, image)

        self.assertEqual(result.shape, (3 * (20 + 32), 30, 3))

    def test_manifest_keeps_group_and_counts(self) -> None:
        """清单加载应保留测试场景分组并统计图片数。"""
        with self._temporary_directory() as directory:
            manifest = Path(directory) / "manifest.json"
            manifest.write_text(
                json.dumps(
                    {
                        "samples": [
                            {
                                "image": "a.jpg",
                                "detect_label": "a.txt",
                                "id": "a",
                                "group": "date/scene",
                            },
                            {
                                "image": "b.jpg",
                                "detect_label": "b.txt",
                                "id": "b",
                                "group": "date/scene",
                            },
                        ]
                    }
                ),
                encoding="utf-8",
            )

            pairs, counts = load_manifest_pairs(str(manifest))

        self.assertEqual(pairs[0], ("a.jpg", "a.txt", "a", "date/scene"))
        self.assertEqual(counts, {"date/scene": 2})

    def test_load_gt_label_supports_labelme_rectangle(self) -> None:
        """LabelMe 矩形应按类别名称转换为模型类别检测框。"""
        with self._temporary_directory() as directory:
            label = Path(directory) / "sample.json"
            label.write_text(
                json.dumps(
                    {
                        "shapes": [
                            {
                                "label": "paper",
                                "points": [[10.0, 20.0], [30.0, 50.0]],
                                "shape_type": "rectangle",
                            }
                        ]
                    }
                ),
                encoding="utf-8",
            )

            result = load_gt_label(str(label), 100, 80, {0: "paper", 1: "liquid", 2: "metal"})

        np.testing.assert_array_equal(result["classes"], np.asarray([0]))
        np.testing.assert_allclose(result["boxes"], np.asarray([[10.0, 20.0, 30.0, 50.0]]))

    def test_discover_labeled_samples_skips_deprecated_and_unlabeled(self) -> None:
        """场景发现应跳过废弃目录和没有同名标签的图片。"""
        with self._temporary_directory() as directory:
            root = Path(directory)
            valid = root / "date" / "TESTs001_paper_scene"
            (valid / "images").mkdir(parents=True)
            (valid / "labels").mkdir()
            (valid / "images" / "kept.jpg").write_bytes(b"image")
            (valid / "labels" / "kept.json").write_text(
                json.dumps({"shapes": []}),
                encoding="utf-8",
            )
            (valid / "images" / "missing.jpg").write_bytes(b"image")
            deprecated = root / "date" / "[deprecated]TESTs002_metal_scene"
            (deprecated / "images").mkdir(parents=True)
            (deprecated / "labels").mkdir()
            (deprecated / "images" / "old.jpg").write_bytes(b"image")
            (deprecated / "labels" / "old.txt").write_text(
                "0 0.5 0.5 0.2 0.2\n",
                encoding="utf-8",
            )

            samples, counts = discover_labeled_samples(root)

        self.assertEqual(len(samples), 1)
        self.assertEqual(samples[0]["group"], "TESTs001_paper_scene")
        self.assertEqual(counts, {"TESTs001_paper_scene": 1})

    def test_discover_labeled_samples_skips_configured_scene_keywords(self) -> None:
        """场景发现应按配置排除任一级目录名包含关键字的场景。"""
        with self._temporary_directory() as directory:
            root = Path(directory)
            for scene_name in ("TESTs001_paper_1x1CM", "TESTs002_metal_regular"):
                scene = root / "batch_1" / scene_name
                (scene / "images").mkdir(parents=True)
                (scene / "labels").mkdir()
                (scene / "images" / "sample.jpg").write_bytes(b"image")
                (scene / "labels" / "sample.txt").write_text(
                    "0 0.5 0.5 0.2 0.2\\n",
                    encoding="utf-8",
                )

            samples, counts = discover_labeled_samples(root, ("1x1cm", "1-2cm", "2x2cm"))

        self.assertEqual(len(samples), 1)
        self.assertEqual(samples[0]["group"], "TESTs002_metal_regular")
        self.assertEqual(counts, {"TESTs002_metal_regular": 1})

    def test_reset_result_root_removes_previous_results(self) -> None:
        """每次测试前应完整移除当前版本的旧外部测试结果。"""
        with self._temporary_directory() as directory:
            run_name = "V018_test"
            work_dir = Path(directory) / "model" / run_name
            result_root = Path(directory) / "6_Result" / run_name
            result_root.mkdir(parents=True)
            (result_root / "old_report.json").write_text("{}", encoding="utf-8")
            (result_root / "detect").mkdir()
            config = type(
                "Config",
                (),
                {"test_output_dir": result_root, "run_name": run_name},
            )()
            context = type("Context", (), {"work_dir": work_dir, "config": config})()

            _reset_result_root(context, result_root)

            self.assertFalse(result_root.exists())

    def test_mirror_key_reports_excludes_images(self) -> None:
        """模型目录镜像应只复制关键报告，不复制三视图。"""
        with self._temporary_directory() as directory:
            root = Path(directory)
            work_dir = root / "model" / "V018_test"
            result_root = root / "6_Result" / "V018_test"
            confidence_dir = result_root / "confidence_20pct"
            (confidence_dir / "images" / "scene").mkdir(parents=True)
            (result_root / "overall_accuracy_summary.csv").write_text("summary", encoding="utf-8")
            (confidence_dir / "accuracy_report.csv").write_text("detail", encoding="utf-8")
            (confidence_dir / "metrics.txt").write_text("metrics", encoding="utf-8")
            (confidence_dir / "images" / "scene" / "view.png").write_bytes(b"image")
            context = type("Context", (), {"work_dir": work_dir})()
            results = [
                {
                    "confidence": 0.2,
                    "metrics": {
                        "overall": {
                            "tp": 1,
                            "fp": 2,
                            "fn": 3,
                            "precision": 1 / 3,
                            "recall": 1 / 4,
                        }
                    },
                }
            ]

            mirror = _mirror_key_reports(context, result_root, results)

            self.assertTrue((mirror / "test_summary.csv").is_file())
            self.assertTrue((mirror / "test_summary.md").is_file())
            self.assertEqual([path.name for path in mirror.iterdir()], ["test_summary.csv", "test_summary.md"])

    def test_metric_row_formats_precision_and_recall_to_three_decimals(self) -> None:
        """CSV 中的 P/R 应固定保留三位小数。"""
        row = _metric_row(
            0.4,
            "测试场景",
            "scene",
            "paper",
            "",
            {"precision": 1 / 3, "recall": 2 / 3},
        )

        self.assertEqual(row["Precision"], "0.333")
        self.assertEqual(row["Recall"], "0.667")

    def test_scene_rows_contains_one_row_per_scene_without_classes(self) -> None:
        """单档报告应每个场景一行且不展开类别指标。"""
        overall = {
            "total_gt": 1,
            "total_pred": 1,
            "tp": 1,
            "fp": 0,
            "fn": 0,
            "precision": 1.0,
            "recall": 1.0,
        }
        groups = {
            "TESTs001_paper_scene": {"overall": overall, "per_class": {"0": {}}},
            "TESTs002_metal_scene": {"overall": overall, "per_class": {"2": {}}},
        }

        rows = _scene_rows(0.4, groups)

        self.assertEqual(len(rows), 2)
        self.assertEqual({row["范围"] for row in rows}, set(groups))
        self.assertTrue(all(row["类别编号"] == "" for row in rows))

    def test_problem_summary_uses_actual_class_and_scene_metrics(self) -> None:
        """问题摘要应识别弱类别以及 FP、FN 较多的场景。"""
        results = [
            {
                "confidence": 0.4,
                "metrics": {
                    "overall": {"precision": 0.8, "recall": 0.7},
                    "per_class": {
                        "0": {"name": "paper", "num_gt": 10, "precision": 0.9, "recall": 0.8},
                        "1": {"name": "liquid", "num_gt": 10, "precision": 0.6, "recall": 0.5},
                    },
                },
                "groups": {
                    "scene_fp": {"overall": {"fp": 8, "fn": 1}},
                    "scene_fn": {"overall": {"fp": 1, "fn": 9}},
                },
            }
        ]

        summary = "\n".join(_problem_summary_lines(results))

        self.assertIn("liquid", summary)
        self.assertIn("scene_fp=8", summary)
        self.assertIn("scene_fn=9", summary)

    def test_discover_labeled_samples_suffixes_duplicate_scene_names(self) -> None:
        """扁平输出遇到跨批次同名场景时应添加编号后缀。"""
        with self._temporary_directory() as directory:
            root = Path(directory)
            for batch in ("batch_a", "batch_b"):
                scene = root / batch / "TESTs001_paper_scene"
                (scene / "images").mkdir(parents=True)
                (scene / "labels").mkdir()
                (scene / "images" / f"{batch}.jpg").write_bytes(b"image")
                (scene / "labels" / f"{batch}.txt").write_text(
                    "0 0.5 0.5 0.2 0.2\n",
                    encoding="utf-8",
                )

            samples, counts = discover_labeled_samples(root)

        self.assertEqual(
            [sample["group"] for sample in samples],
            ["TESTs001_paper_scene", "TESTs001_paper_scene__2"],
        )
        self.assertEqual(
            counts,
            {"TESTs001_paper_scene": 1, "TESTs001_paper_scene__2": 1},
        )


if __name__ == "__main__":
    unittest.main()
