# Src/eval/metrics.py
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    precision_recall_fscore_support,
)

# 画图只在需要时导入 matplotlib（避免无图形环境时影响核心评估）
# from sklearn.metrics import ConfusionMatrixDisplay  # 可选


@dataclass
class MetricsOutput:
    metrics: Dict[str, Any]
    report: str
    cm: np.ndarray


def compute_prf(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    *,
    average: str = "binary",
    pos_label: Optional[int] = 1,
    zero_division: int = 0,
) -> Tuple[float, float, float]:
    """
    计算 Precision / Recall / F1
    average:
      - binary: 二分类（需给 pos_label）
      - macro/micro/weighted: 多分类也可
    """
    p, r, f1, _ = precision_recall_fscore_support(
        y_true,
        y_pred,
        average=average,
        pos_label=pos_label if average == "binary" else None,
        zero_division=zero_division,
    )
    return float(p), float(r), float(f1)


def compute_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    *,
    labels: Optional[np.ndarray] = None,
    normalize: Optional[str] = None,  # None / "true" / "pred" / "all"
) -> np.ndarray:
    cm = confusion_matrix(y_true, y_pred, labels=labels, normalize=normalize)
    return cm


def compute_report(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    *,
    target_names: Optional[list] = None,
    zero_division: int = 0,
) -> str:
    return classification_report(
        y_true,
        y_pred,
        target_names=target_names,
        digits=4,
        zero_division=zero_division,
    )


def evaluate_binary_classifier(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    *,
    average: str = "binary",
    pos_label: Optional[int] = 1,
    zero_division: int = 0,
    cm_normalize: Optional[str] = None,
    labels: Optional[np.ndarray] = None,
    target_names: Optional[list] = None,
) -> MetricsOutput:
    """
    统一评估入口：返回 dict 指标 + report + confusion matrix
    """
    acc = float(accuracy_score(y_true, y_pred))
    p, r, f1 = compute_prf(
        y_true, y_pred, average=average, pos_label=pos_label, zero_division=zero_division
    )
    cm = compute_confusion_matrix(y_true, y_pred, labels=labels, normalize=cm_normalize)
    rep = compute_report(y_true, y_pred, target_names=target_names, zero_division=zero_division)

    metrics = {
        "accuracy": acc,
        "precision": p,
        "recall": r,
        "f1": f1,
        "average": average,
        "pos_label": pos_label,
        "cm_normalize": cm_normalize,
        "confusion_matrix": cm.tolist(),
    }

    return MetricsOutput(metrics=metrics, report=rep, cm=cm)


def save_json(data: Dict[str, Any], path: Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def save_text(text: str, path: Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def plot_and_save_confusion_matrix(
    cm: np.ndarray,
    *,
    out_path: Path,
    labels: Optional[list] = None,
    title: str = "Confusion Matrix",
) -> None:
    """
    可选：绘制混淆矩阵并保存 png。
    注意：不指定颜色（遵循你项目的通用约束，避免硬编码风格）。
    """
    import matplotlib.pyplot as plt

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots()
    im = ax.imshow(cm)

    ax.set_title(title)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")

    # 坐标轴标签
    if labels is None:
        labels = [str(i) for i in range(cm.shape[0])]
    ax.set_xticks(range(len(labels)))
    ax.set_yticks(range(len(labels)))
    ax.set_xticklabels(labels)
    ax.set_yticklabels(labels)

    # 在格子里写数值
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, f"{cm[i, j]:.4g}" if isinstance(cm[i, j], float) else str(cm[i, j]),
                    ha="center", va="center")

    fig.colorbar(im, ax=ax)
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)
