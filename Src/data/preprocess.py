# Src/data/preprocess.py
from __future__ import annotations

import re
from datetime import datetime
from pathlib import Path
from typing import Optional, Dict, Any

import pandas as pd

import config
from Src.data.io import read_csv, validate_and_prepare, save_csv


# ----------------------------
# 1) 去除符号（用空格替换）
# ----------------------------
_non_alnum_re = re.compile(
    config.REGEX_NON_ALNUM if hasattr(config, "REGEX_NON_ALNUM") else r"[^A-Za-z0-9\s]"
)


def remove_symbols(text: str) -> str:
    """
    将所有非字母数字空白字符替换为一个空格。
    示例："(abc);def?" -> " abc  def "
    """
    return _non_alnum_re.sub(" ", text)


# ----------------------------
# 2) 压缩空白
# ----------------------------
_ws_re = re.compile(r"\s+")


def squeeze_whitespace(text: str) -> str:
    """将多个空白压缩为单个空格，并去首尾空格。"""
    return _ws_re.sub(" ", text).strip()


# ----------------------------
# 3) 词干化（依赖 NLTK）
# ----------------------------
def _get_stemmer(stemmer_type: str = "porter"):
    """
    延迟导入 NLTK，避免未安装时影响其他步骤。
    stemmer_type: 'porter' / 'snowball'
    """
    try:
        from nltk.stem import PorterStemmer, SnowballStemmer
    except Exception as e:
        raise RuntimeError(
            "NLTK is required for stemming but not available. "
            "Install it via: pip install nltk"
        ) from e

    stemmer_type = (stemmer_type or "porter").lower()
    if stemmer_type == "porter":
        return PorterStemmer()
    if stemmer_type == "snowball":
        return SnowballStemmer("english")
    raise ValueError(f"Unsupported stemmer_type: {stemmer_type}")


def stem_text(text: str, stemmer_type: str = "porter") -> str:
    """对文本按空格切词后进行词干化，再拼回。"""
    stemmer = _get_stemmer(stemmer_type)
    tokens = text.split()
    stemmed = [stemmer.stem(tok) for tok in tokens]
    return " ".join(stemmed)


# ----------------------------
# 4) 主预处理流程
# ----------------------------
def preprocess_text(
    text: str,
    *,
    do_lowercase: bool = True,
    remove_punct: bool = True,
    do_squeeze_whitespace: bool = True,
    do_stemming: bool = False,
    stemmer_type: str = "porter",
) -> str:
    """
    单条文本预处理：
    - 可选小写
    - 去符号（空格替换）
    - 压缩空白
    - 可选词干化
    """
    if text is None:
        text = ""
    text = str(text)

    if do_lowercase:
        text = text.lower()

    if remove_punct:
        text = remove_symbols(text)

    if do_squeeze_whitespace:
        text = squeeze_whitespace(text)

    if do_stemming:
        text = stem_text(text, stemmer_type=stemmer_type)
        if do_squeeze_whitespace:
            text = squeeze_whitespace(text)

    return text


def preprocess_dataframe(
    df: pd.DataFrame,
    text_col: str = config.TEXT_COL,
    *,
    output_col: Optional[str] = None,
    do_lowercase: bool = True,
    remove_punct: bool = True,
    do_squeeze_whitespace: bool = True,
    do_stemming: bool = False,
    stemmer_type: str = "porter",
) -> pd.DataFrame:
    """
    对 DataFrame 中某列文本做预处理。
    - output_col 为 None：覆盖 text_col
    - output_col 指定：保留原列并新增预处理结果列
    """
    out = df.copy()
    target_col = output_col or text_col

    out[target_col] = out[text_col].apply(
        lambda x: preprocess_text(
            x,
            do_lowercase=do_lowercase,
            remove_punct=remove_punct,
            do_squeeze_whitespace=do_squeeze_whitespace,
            do_stemming=do_stemming,
            stemmer_type=stemmer_type,
        )
    )
    return out


# ----------------------------
# 5) 测试入口：读数据→预处理→写入 Results/带时间戳目录
# ----------------------------
def _make_timestamped_dir(base_dir: Path, prefix: str = "preprocess") -> Path:
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = base_dir / f"{prefix}_{ts}"
    out_dir.mkdir(parents=True, exist_ok=True)
    return out_dir


def _write_preview(df_before: pd.DataFrame, df_after: pd.DataFrame, out_dir: Path) -> None:
    """
    写一个对照文件：展示处理前后前N行（仅文本列 + 标签列 + id列）
    """
    cols = [c for c in [config.ID_COL, config.TEXT_COL, config.LABEL_COL] if c in df_before.columns]
    before = df_before[cols].head(20).copy()
    after = df_after[cols].head(20).copy()

    # 重命名方便对照
    before = before.rename(columns={config.TEXT_COL: f"{config.TEXT_COL}_before"})
    after = after.rename(columns={config.TEXT_COL: f"{config.TEXT_COL}_after"})

    merged = pd.concat([before.reset_index(drop=True), after[[f"{config.TEXT_COL}_after"]]], axis=1)
    save_csv(merged, out_dir / "preview_before_after.csv", index=False)


def _write_log(info: Dict[str, Any], out_dir: Path) -> None:
    lines = []
    for k, v in info.items():
        lines.append(f"{k}: {v}")
    (out_dir / "run_log.txt").write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    # 你现在的 Title_Stemmed 已经 stemmed，因此默认 do_stemming=False
    # 如果你要复现“去符号 + 词干化”的完整流程，把 do_stemming=True
    DO_STEMMING = False
    STEMMER_TYPE = "porter"

    # 输出到 Results 下的时间戳目录
    out_dir = _make_timestamped_dir(config.RESULTS_DIR, prefix="preprocess")

    # 选择测试哪些文件：这里全部跑一遍，方便你对照
    inputs = {
        "train": config.RAW_TRAIN_FILE,
        "test": config.RAW_TEST_FILE,
        "stemmed_all": config.RAW_STEMMED_FILE,
    }

    # 记录信息
    run_info = {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "do_stemming": DO_STEMMING,
        "stemmer_type": STEMMER_TYPE,
        "regex_non_alnum": getattr(config, "REGEX_NON_ALNUM", r"[^A-Za-z0-9\s]"),
        "text_col": config.TEXT_COL,
        "label_col": config.LABEL_COL,
        "id_col": config.ID_COL,
        "output_dir": str(out_dir),
    }

    for name, path in inputs.items():
        df_raw = read_csv(path)
        df_raw = validate_and_prepare(df_raw, dropna=True)

        df_processed = preprocess_dataframe(
            df_raw,
            text_col=config.TEXT_COL,
            output_col=None,              # 覆盖原列；如要保留可改成 "Title_Processed"
            do_lowercase=True,
            remove_punct=True,
            do_squeeze_whitespace=True,
            do_stemming=DO_STEMMING,
            stemmer_type=STEMMER_TYPE,
        )

        # 输出文件
        save_csv(df_processed, out_dir / f"{name}_preprocessed.csv", index=False)

        # 对照样例（写一次即可，也可以每个文件都写）
        _write_preview(df_raw, df_processed, out_dir / f"preview_{name}")
        run_info[f"{name}_rows"] = len(df_processed)
        run_info[f"{name}_output"] = str(out_dir / f"{name}_preprocessed.csv")

    _write_log(run_info, out_dir)

    print(f"[OK] Preprocess test outputs written to: {out_dir}")
