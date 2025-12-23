# config.py 设置项目文件路径和超参数
from __future__ import annotations
from pathlib import Path


# ============================
# 1) 项目根目录与路径
# ============================
ROOT_DIR = Path(r"/workspace/user/Coding/jialin/StackoverflowClassification")

DATA_DIR = ROOT_DIR / "Data"
RAW_DIR = DATA_DIR / "Raw"
PROCESSED_DIR = DATA_DIR / "Processed"

RESULTS_DIR = ROOT_DIR / "Results"
FIGURES_DIR = RESULTS_DIR / "Figures"
METRICS_DIR = RESULTS_DIR / "Metrics"
MODELS_DIR = RESULTS_DIR / "Models"

SCRIPTS_DIR = ROOT_DIR / "Scripts"
SRC_DIR = ROOT_DIR / "Src"


# ============================
# 2) 数据文件名（Raw/Processed）
# ============================
RAW_STEMMED_FILE = RAW_DIR / "StemmedData_30000.csv"
RAW_TRAIN_FILE = RAW_DIR / "train_24000.csv"
RAW_TEST_FILE = RAW_DIR / "test_6000.csv"

PROCESSED_STEMMED_FILE = PROCESSED_DIR / "StemmedData_30000_clean.csv"
PROCESSED_TRAIN_FILE = PROCESSED_DIR / "train_24000_clean.csv"
PROCESSED_TEST_FILE = PROCESSED_DIR / "test_6000_clean.csv"


# ============================
# 3) 模型与结果文件保存路径
# ============================
VECTORIZER_PATH = MODELS_DIR / "vectorizer.joblib"
NB_MODEL_PATH = MODELS_DIR / "nb_model.joblib"
LABEL_MAP_PATH = MODELS_DIR / "label_map.json"

NB_METRICS_JSON = METRICS_DIR / "nb_metrics.json"
NB_REPORT_TXT = METRICS_DIR / "nb_classification_report.txt"
CONFUSION_MATRIX_PNG = FIGURES_DIR / "confusion_matrix.png"


# ============================
# 4) 数据字段与标签映射
# ============================
ID_COL = "Id"
TEXT_COL = "Title"
LABEL_COL = "Tags"

# 二分类标签
LABEL_MAP = {"python": 0, "html": 1}
INV_LABEL_MAP = {v: k for k, v in LABEL_MAP.items()}
ALLOWED_LABELS = tuple(LABEL_MAP.keys())


# ============================
# 5) 训练超参数
# ============================
RANDOM_SEED = 42
TEST_SIZE = 0.2
SHUFFLE = True
STRATIFY = True


# ============================
# 6) 向量化超参数
# ============================
VECTORIZER_PARAMS = {
    # 基础
    "lowercase": False,      # 若要 vectorizer 内部小写可设 True
    "analyzer": "word",      # word / char / char_wb
    "ngram_range": (1, 1),   # (1,1) 仅 unigram；可尝试 (1,2)
    "token_pattern": r"(?u)\b\w+\b",  # 默认也可；此处更宽松以保留数字/下划线等
    "strip_accents": None,   # None / 'ascii' / 'unicode'
    "stop_words": None,      # 若 REMOVE_STOPWORDS=True，可在代码里替换为英文停用词表

    # 词表过滤（调参重点）
    "min_df": 1,             # 低频词过滤；可试 2/3/5
    "max_df": 1.0,           # 高频词过滤；可试 0.9/0.95
    "max_features": None,    # 限制词表大小；如 20000/50000

    # 词频权重（CountVectorizer 仍是词频；binary=True 转为0/1）
    "binary": False,
}


# ============================
# 7) 模型超参数（Step 3/4：MultinomialNB）
# ============================
NB_PARAMS = {
    "alpha": 1.0,            # 拉普拉斯/狄利克雷平滑；常试 0.1/0.5/1.0
    "fit_prior": True,
    "class_prior": None,     # 可手动指定，如 [0.5, 0.5]
}


# ============================
# 8) 评估超参数（Step 5）
# ============================
POS_LABEL_NAME = "html"                  # 以 html 为正类（用于二分类解释）
POS_LABEL_ID = LABEL_MAP[POS_LABEL_NAME]

METRICS_AVERAGE = "binary"               # binary / macro / micro / weighted
ZERO_DIVISION = 0                        # sklearn classification_report 参数

SAVE_CONFUSION_MATRIX_FIG = True
CONFUSION_MATRIX_NORMALIZE = None   # None / "true" / "pred" / "all"