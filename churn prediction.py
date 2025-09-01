"""
Advanced Customer Churn Prediction Script (v4.2 – Stratified)
-------------------------------------------------------------
* Directly use StratifiedKFold(n_splits=5, shuffle=True)
* In-frame snapshot_date = the latest transaction in the training set to avoid time leakage
* If the validation set is of a single category → rollback for 30 days each time (up to 180 days), skip if it is still of a single category
"""

import pandas as pd, numpy as np, logging, matplotlib.pyplot as plt
import lightgbm as lgb
from lightgbm import LGBMClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
import shap, seaborn as sns, statsmodels.api as sm
from matplotlib.backends.backend_pdf import PdfPages
from scipy.stats import linregress

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s | %(levelname)s | %(message)s")
sns.set_theme(style="whitegrid")

# ---------- 1. read the original data set ----------
df = pd.read_csv("Lolly_Export.csv")
df["timestamp"] = pd.to_datetime(
    df["timestamp"], format="%d/%m/%Y %H:%M:%S",
    dayfirst=True, errors="coerce"
)
df = df.dropna(subset=["timestamp"]).sort_values("timestamp")

HORIZON = 30      # 30 days without consumption = loss

# ---------- 2. feature construction ----------
def build_features(txns: pd.DataFrame, snapshot: pd.Timestamp) -> pd.DataFrame:
    rec = []
    for cid, g in txns.groupby("customerIdentifier", sort=False):
        g = g.sort_values("timestamp")
        latest = g["timestamp"].iat[-1]
        recency = (snapshot - latest).days

        cut30, cut60, cut90 = latest - pd.Timedelta(days=30), \
                              latest - pd.Timedelta(days=60), \
                              latest - pd.Timedelta(days=90)
        last30 = g[g["timestamp"] >= cut30]
        prev30 = g[(g["timestamp"] < cut30) & (g["timestamp"] >= cut60)]
        last90 = g[g["timestamp"] >= cut90]

        rec.append({
            "customerIdentifier": cid,
            "last_txn_ts": latest,
            "Recency": recency,
            "amt_30": last30["totalAmount"].sum(),
            "txn_30": len(last30),
            "amt_90": last90["totalAmount"].sum(),
            "txn_90": len(last90),
            "amt_30_vs_prev30": (last30["totalAmount"].sum()
                                 / max(prev30["totalAmount"].sum(), 1) - 1),
            "txn_30_vs_prev30": (len(last30) / max(len(prev30), 1) - 1),
        })
    return pd.DataFrame(rec)

snapshot_global = df["timestamp"].max()
X = build_features(df, snapshot_global).sort_values("last_txn_ts").reset_index(drop=True)

# Global label (It will be recalculated each time it is folded later, so there is no need to change it here again)
X["churn"] = (X["Recency"] > HORIZON).astype(int)

feature_cols = [c for c in X.columns
                if c not in ("customerIdentifier", "churn", "last_txn_ts")]
model_cols   = [c for c in feature_cols if c != "Recency"]  # Actual input of the model

# ---------- 3. StratifiedKFold ----------
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

fold_aucs, shap_stack, val_indices = [], [], []
oof_pred = np.full(len(X), np.nan)        # Unverified rows maintain NaN

def run_cv(splitter):
    for fold, (tr_idx, val_idx) in enumerate(splitter.split(X, X["churn"])):
        snapshot_fold = X.loc[tr_idx, "last_txn_ts"].max()

        # === Construct in-fold data ===
        X_tr = X.loc[tr_idx, feature_cols].copy()
        X_val = X.loc[val_idx, feature_cols].copy()

        rec_tr  = (snapshot_fold - X.loc[tr_idx, "last_txn_ts"]).dt.days
        rec_val = (snapshot_fold - X.loc[val_idx, "last_txn_ts"]).dt.days
        y_tr  = (rec_tr  > HORIZON).astype(int).values
        y_val = (rec_val > HORIZON).astype(int).values
        X_tr["Recency"]  = rec_tr.values
        X_val["Recency"] = rec_val.values

        # —— If the validation set is still a single class, roll back by 30 days according to the time —— 
        back = 0
        while len(np.unique(y_val)) < 2 and back < 180:
            back += 30
            logging.warning("fold %d Verification set single class → Rollback %d days", fold, back)
            extra_idx = np.arange(val_idx[0] - back, val_idx[-1] + 1)
            extra_idx = extra_idx[(0 <= extra_idx) & (extra_idx < len(X))]
            X_val = X.loc[extra_idx, feature_cols].copy()
            rec_val = (snapshot_fold - X.loc[extra_idx, "last_txn_ts"]).dt.days
            y_val   = (rec_val > HORIZON).astype(int).values
            X_val["Recency"] = rec_val.values

        if len(np.unique(y_val)) < 2:
            logging.warning("fold %d Still a single class, skip", fold)
            continue

        # === train LightGBM ===
        model = LGBMClassifier(
            n_estimators      = 2000,
            learning_rate     = 0.1,
            num_leaves        = 63,
            max_depth         = -1,        # No limit on depth
            min_data_in_leaf  = 5,
            min_sum_hessian_in_leaf = 1e-3,
            subsample         = 0.8,
            colsample_bytree  = 0.8,
            objective         = "binary",
            class_weight      = "balanced",   # use the fine adjustment of {0:1, 1:2} instead
            random_state      = fold,
        )

        model.fit(
            X_tr[model_cols], y_tr,
            eval_set=[(X_val[model_cols], y_val)],
            eval_metric="auc",
            callbacks=[lgb.early_stopping(50)]
        )

        y_pred = model.predict_proba(X_val[model_cols])[:, 1]
        auc = roc_auc_score(y_val, y_pred)
        fold_aucs.append(auc)
        logging.info("fold %d: train=%d  val=%d  AUC=%.4f",
                     fold, len(tr_idx), len(X_val), auc)

        oof_pred[X_val.index] = y_pred      # Use the current index directly
        shap_stack.append(shap.TreeExplainer(model)(X_val[model_cols]).values)
        val_indices.append(X_val.index.values)

run_cv(skf)

mask = ~np.isnan(oof_pred)
logging.info("Fold AUCs %s | mean=%.4f | masked OOF=%.4f",
             [round(a,4) for a in fold_aucs],
             np.mean(fold_aucs),
             roc_auc_score(X["churn"][mask], oof_pred[mask]))

# ---------- 4. SHAP ----------
if shap_stack:
    shap.summary_plot(
        np.vstack(shap_stack),
        X.loc[np.concatenate(val_indices), model_cols],
        feature_names=model_cols,
        show=False,
    )
    plt.tight_layout(); plt.savefig("shap_beeswarm.png", dpi=300); plt.close()
    logging.info("shap_beeswarm.png is created")
else:
    logging.warning("There is no valid fold. Skip the SHAP drawing")

# ---------- 5. Export OOF ----------
cols_to_keep = [
    "customerIdentifier", "churn",
    "Recency", "txn_90", "amt_90", "txn_30_vs_prev30"
]
X_out = X[cols_to_keep].copy()
X_out["p_churn"] = oof_pred
X_out.to_csv("oof_predictions.csv", index=False)
logging.info("oof_predictions.csv 已写出")

# ---------- 6. Linear check (paper-ready, 8 figures) ----------
import os
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator, FuncFormatter
from scipy.stats import linregress
import statsmodels.api as sm

OUT_DIR = Path("paper_figs")
OUT_DIR.mkdir(parents=True, exist_ok=True)

plt.rcParams.update({
    "figure.dpi": 300,
    "savefig.dpi": 600,
    "font.size": 11,
    "axes.titlesize": 11,
    "axes.labelsize": 11,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "axes.linewidth": 0.8,
})

def _sanitize(name: str) -> str:
    return "".join(c if (c.isalnum() or c in "_-") else "_" for c in name)

def pretty_feature_name(feat: str) -> str:
    mapping = {
        "Recency": "Recency (days)",
        "txn_90": "Transactions (last 90d)",
        "amt_90": "Amount (last 90d)",
        "txn_30_vs_prev30": "Transactions 30d vs prior 30d (ratio−1)",
    }
    return mapping.get(feat, feat)

def format_axes(ax, xlabel: str):
    # 只留左/下边框；细主网格；整数 x 刻度（若可能）
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(True, which="major", linewidth=0.5, alpha=0.3)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("P(Churn within 30d)")
    ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:.0%}"))

def wilson_ci(k: int, n: int, z: float = 1.96):
    if n <= 0:
        return np.nan, np.nan
    phat = k / n
    denom = 1.0 + z**2 / n
    center = (phat + z**2 / (2*n)) / denom
    half = z * np.sqrt((phat*(1 - phat) + z**2/(4*n)) / n) / denom
    return center - half, center + half

def plot_bins(x: np.ndarray, y: np.ndarray, feat: str, out_path: Path):
    mask = np.isfinite(x) & np.isfinite(y)
    x, y = x[mask], y[mask].astype(int)
    # 使用 7 等分位（与原脚本保持一致）：0..1 的 8 个量点形成 7 个箱
    q = np.unique(np.quantile(x, np.linspace(0, 1, 8)))
    if len(q) < 3:
        logging.warning("特征 %s 分位点不足，跳过分箱图", feat)
        return
    # digitize 给出箱索引 0..len(q)-2
    bins = np.digitize(x, q[1:-1], right=False)

    xs, rates, los, his, ns = [], [], [], [], []
    for b in range(len(q) - 1):
        m = bins == b
        n = int(m.sum())
        if n == 0:
            continue
        xb = float(np.mean(x[m]))
        k = int(np.sum(y[m]))
        rate = k / n
        lo, hi = wilson_ci(k, n)
        xs.append(xb); rates.append(rate); los.append(lo); his.append(hi); ns.append(n)

    # 按 x 升序绘线
    order = np.argsort(xs)
    xs = np.asarray(xs)[order]
    rates = np.asarray(rates)[order]
    los = np.asarray(los)[order]
    his = np.asarray(his)[order]

    fig, ax = plt.subplots(figsize=(4.2, 3.2))
    ax.plot(xs, rates, marker="o", linewidth=1.7, markersize=4)
    ax.fill_between(xs, los, his, alpha=0.2, linewidth=0)
    format_axes(ax, pretty_feature_name(feat))
    # R^2（线性拟合 bins 均值 vs 率）
    if len(xs) >= 2:
        r2 = linregress(xs, rates).rvalue ** 2
        ax.text(0.02, 0.96, rf"$R^2={r2:.2f}$", transform=ax.transAxes,
                ha="left", va="top")
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)
    logging.info("Saved %s", out_path)

def plot_logit(x: np.ndarray, y: np.ndarray, feat: str, out_path: Path):
    mask = np.isfinite(x) & np.isfinite(y)
    x, y = x[mask], y[mask].astype(int)
    # 处理常数列/极端值：若方差几乎为 0，直接跳过
    if np.nanstd(x) < 1e-12:
        logging.warning("特征 %s 方差过小，跳过 Logit 图", feat)
        return

    # 适度截尾，避免极端点撑开可视范围（不改变标签）
    lo, hi = np.nanpercentile(x, [1, 99])
    x_vis = np.clip(x, lo, hi)

    # 拟合带常数项的单变量 Logit（与原脚本保持一致）
    try:
        logit = sm.Logit(y, sm.add_constant(x)).fit_regularized(disp=False)
        beta = float(logit.params[1])
        xs = np.linspace(np.nanmin(x_vis), np.nanmax(x_vis), 200)
        preds = logit.predict(sm.add_constant(xs))
    except Exception as e:
        logging.warning("%s Logit 拟合失败：%s，改用极大似然", feat, e)
        try:
            logit = sm.Logit(y, sm.add_constant(x)).fit(disp=False, method="lbfgs", maxiter=200)
            beta = float(logit.params[1])
            xs = np.linspace(np.nanmin(x_vis), np.nanmax(x_vis), 200)
            preds = logit.predict(sm.add_constant(xs))
        except Exception as e2:
            logging.warning("%s 仍失败：%s，跳过 Logit 图", feat, e2)
            return

    fig, ax = plt.subplots(figsize=(4.2, 3.2))
    # 0/1 标签增加轻微竖向抖动，避免遮挡
    rng = np.random.default_rng(12345)
    jitter = rng.uniform(-0.02, 0.02, size=len(y))
    ax.scatter(x_vis, y + jitter, s=8, alpha=0.25, linewidths=0)
    # 预测曲线
    ax.plot(xs, preds, linewidth=1.7)
    format_axes(ax, pretty_feature_name(feat))
    ax.text(0.02, 0.96, rf"$\beta={beta:.3f}$", transform=ax.transAxes,
            ha="left", va="top")
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)
    logging.info("Saved %s", out_path)

# === 从前面已写出的 X_out 取数 ===
y = X_out["churn"].to_numpy()
X_feat = X_out.drop(columns=["customerIdentifier", "churn"])

targets = ["Recency", "txn_90", "amt_90", "txn_30_vs_prev30"]
saved_paths = []

for feat in targets:
    if feat not in X_feat.columns:
        logging.warning("%s 缺失，跳过该特征的两张图", feat)
        continue
    x = X_feat[feat].to_numpy()

    p1 = OUT_DIR / f"bins_{_sanitize(feat)}.png"
    p2 = OUT_DIR / f"logit_{_sanitize(feat)}.png"

    plot_bins(x, y, feat, p1)
    plot_logit(x, y, feat, p2)

    saved_paths.extend([p1.as_posix(), p2.as_posix()])

logging.info("All paper-ready figures saved under: %s", OUT_DIR.resolve())
logging.info("Files: %s", saved_paths)