# Src/data/io.py 数据的输入和输出
from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple, Dict
import pandas as pd
import config


@dataclass
class Dataset:
    """统一的数据载体：文本 + 数值标签 + 原始DataFrame"""
    df: pd.DataFrame
    texts: pd.Series
    y: pd.Series


def _assert_columns(df: pd.DataFrame, required: Tuple[str, ...]) -> None:
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}. "
                         f"Found columns: {list(df.columns)}")


def _normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    对列名做轻微标准化（去首尾空格）。
    注意：不做大小写转换，避免把真实列名改坏。
    """
    df = df.copy()
    df.columns = [str(c).strip() for c in df.columns]
    return df


def read_csv(path: Path,
             encoding: str = "utf-8",
             sep: str = ",") -> pd.DataFrame:
    """
    读取CSV。常见情况：utf-8 或 utf-8-sig（含BOM）。
    这里优先 utf-8，失败再尝试 utf-8-sig。
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"CSV file not found: {path}")

    try:
        df = pd.read_csv(path, encoding=encoding, sep=sep)
    except UnicodeDecodeError:
        # 兼容 Excel 导出的 CSV
        df = pd.read_csv(path, encoding="utf-8-sig", sep=sep)

    df = _normalize_columns(df)
    return df


def save_csv(df: pd.DataFrame,
             path: Path,
             index: bool = False,
             encoding: str = "utf-8") -> None:
    """保存CSV，默认不写index。"""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=index, encoding=encoding)


def validate_and_prepare(df: pd.DataFrame,
                         *,
                         text_col: str = config.TEXT_COL,
                         label_col: str = config.LABEL_COL,
                         id_col: Optional[str] = config.ID_COL,
                         allowed_labels: Tuple[str, ...] = config.ALLOWED_LABELS,
                         dropna: bool = True) -> pd.DataFrame:
    """
    基础校验与准备：
    - 必须包含 text_col 与 label_col（id_col 可选）
    - text 统一转成 str
    - label 统一转成 str 并校验 allowed_labels
    - 处理空值
    """
    required = (text_col, label_col) if id_col is None else (id_col, text_col, label_col)
    _assert_columns(df, required)

    out = df.copy()

    # 统一类型
    out[text_col] = out[text_col].astype(str)
    out[label_col] = out[label_col].astype(str).str.strip().str.lower()

    if id_col is not None:
        # id 不强制必须为 int，但可以转成 string 以保持一致性
        out[id_col] = out[id_col]

    # 处理空文本：如果原本是 NaN，astype(str) 会变成 "nan"
    # 这里将 "nan" 和空字符串都视为缺失
    out[text_col] = out[text_col].replace({"nan": "", "None": ""})

    if dropna:
        before = len(out)
        out = out[(out[text_col].str.strip() != "") & (out[label_col].str.strip() != "")]
        after = len(out)
        if after < before:
            # 不打印，交由脚本层决定是否log
            pass

    # 标签合法性校验
    illegal = sorted(set(out[label_col].unique()) - set(allowed_labels))
    if illegal:
        raise ValueError(
            f"Found illegal labels: {illegal}. Allowed labels: {allowed_labels}"
        )
    return out


def encode_labels(labels: pd.Series,
                  label_map: Dict[str, int] = config.LABEL_MAP) -> pd.Series:
    """将文本标签映射为整数标签。"""
    return labels.map(label_map).astype(int)


def load_dataset(path: Path,
                 *,
                 text_col: str = config.TEXT_COL,
                 label_col: str = config.LABEL_COL,
                 id_col: Optional[str] = config.ID_COL,
                 label_map: Dict[str, int] = config.LABEL_MAP) -> Dataset:
    """
    一站式读取 + 校验 + 标签编码
    返回 Dataset(texts, y, df)
    """
    df = read_csv(path)
    df = validate_and_prepare(df, text_col=text_col, label_col=label_col, id_col=id_col)

    texts = df[text_col]
    y = encode_labels(df[label_col], label_map=label_map)

    return Dataset(df=df, texts=texts, y=y)
