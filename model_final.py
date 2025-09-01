# -*- coding: utf-8 -*-
"""
Edge Trigger + Dynamic Points  (Reinforced & Annotated)
-------------------------------------------------------------------------------
- Baseline propensity model (no points)
- Effect model backbone: "hazard" (non-purchase days) or "classification"
- A: Exposure features (rule/campaign/lagpoints) -> expo_day/expo_7/expo_30
- B: Balance & progress (balance, gap_to_reward, pct_to_next, near_reward)
- C: Purchase hazard on non-purchase days (auto-select horizon); redemption hazard kept
- BP-score tilt; clip-to-zero prior; liability report; sensitivity heatmap

本版关键增强：
- [ADD] Edge Trigger 成本支持“条件发生”口径（更贴现实）：cost = p1_used × extra_points × VPP_OLD
- [ADD] 明确“额外点数”（BOOST_MULT），只对额外部分计成本/效应
- [ADD] 仅对“临近奖励”的人天投放（NEAR_REWARD_MIN）
- [ADD] 概率后校准（decile_map/全局缩放），筛选与汇报口径可分离；阈值可按校准缩放自适应
- [ADD] 预算切断 + 累计曲线；run_config/calibration_info 落盘
"""

import os, json, warnings
import pandas as pd, numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline

# ---------------- 0) Paths & globals ----------------
DATA_FILE = "filtered_UCL_data_with_scores.csv"
CAMPAIGN_FILE = "campaign_exposure.csv"  # 可选：若存在并含 gUniqueID,dateDateTime,exposed
SAVE_DIR  = "outputs"; os.makedirs(SAVE_DIR, exist_ok=True)

ID_COL   = "gUniqueID"
TIME_COL = "dateDateTime"
PENCE_COL= "iPenceIncVat"
EM1_COL  = "itemEmissionScore"
EM2_COL  = "emissionScore"
PTS_COL0 = "reward points"   # raw points (can be negative)

# ---------------- Value per point (cost) & margin assumptions ----------------
FACE_VALUE_PER_POINT = 10.0 / 100.0   # £0.10 per point
DISCOUNT_FACTOR = 0.40                 # δ
EXPECTED_REDEEM_RATE = 0.65            # r

CPP_REDEEM = FACE_VALUE_PER_POINT * DISCOUNT_FACTOR          # £/pt，被兑1点的净成本
VPP_OLD    = CPP_REDEEM * EXPECTED_REDEEM_RATE               # £/pt，每发出1点的期望成本（统一口径）

MARGIN  = 0.30
SEED = 42

# Trigger window
REC_WIN = (21, 60)
NO_ORDER_DAYS = 7

# -------- Dynamic points knobs --------
ALPHA = 0.003
BETA  = 0.5
GAMMA_LIST = [1.00, 1.15]

# === A/B/C 配置 ===
EFFECT_BACKBONE = "hazard"        # {"hazard","classification"}
EFFECT_SIGNAL   = "points"        # {"points","exposure","progress"}
EXPOSURE_MODE   = "rule"          # {"rule","campaign","lagpoints","none"}
EXPOSURE_WINDOW = 7
EXPO_OFFER_DELTA = 1.0

# 积分映射设置
PTS_FEATURE = "pts_lag30"
INCENTIVE_MODE = "precredit"
OVERLAP = 1.0

# Emission normalization
EM_NORMALIZE = True

# Effect sign handling
FORCE_NONNEG_POINTS_EFFECT = False

# Tilts
BP_TILT = 0.6
REC_TILT = 0.2

# Hazard modeling（赎回/购买）
REDEEM_HORIZON_DAYS = 30
BUY_HORIZONS = (7, 14, 30)
REDEEM_STEP_PTS = 1000

# Fast mode
FAST = False; MAX_USERS = 2000; DAYS_BACK = 365

# Policy knobs (edge trigger)
COOLDOWN_DAYS = 30

# === [ADD] 触发经济学与筛选/汇报口径 ===
OFFER_MODE = "upfront"         # {"conditional","upfront"} 成本口径：条件发生/预发
BOOST_MULT = 1.0                   # 额外倍数；1.0=再送一份（双倍积分）；只对“额外的那份”计成本/效应
NEAR_REWARD_MIN = None             # 仅对 pct_to_next >= 0.6 的人天候选；设 None 关闭

MAKE_CALIBRATION = True
USE_CALIBRATED_PROBS = False        # 计算 uplift/画汇报曲线/触发摘要用校准口径
CALIBRATION_MODE = "decile_map"    # {"decile_map","global_scale"}
CALIBRATION_BINS = 10
CALIBRATION_MIN_COUNT_PER_BIN = 50

SELECT_WITH_CALIBRATION = False     # True: 用校准ROI做筛选；False: 用未校准ROI
ADAPT_THRESHOLD_TO_CALIBRATION = True
MIN_ROI_THRESHOLD = 0.05            # 也可设 None 完全依赖预算切断
BUDGET_LIMIT_GBP = 50           # 例如 50/100 做小预算前缀验证

# ---------------- Helper ----------------
def safe_ratio(num, den, default=0.0):
    den = float(den)
    if not np.isfinite(den) or abs(den) < 1e-12: return float(default)
    val = float(num) / den
    if not np.isfinite(val): return float(default)
    return val

def clip01(x): return np.minimum(np.maximum(x, 0.0), 1.0)

sigmoid = lambda z: 1/(1+np.exp(-z))

# === 预处理&度量（用于审计对照） ===
def preprocess_naive(df_in):
    return df_in.copy(), {"dup_removed": 0, "posting_lag_mean_days": 0.0, "emission_imputed_cnt": 0}

def preprocess_adjusted(df_in):
    df = df_in.copy()
    audit = {}
    if TIME_COL in df.columns and ID_COL in df.columns:
        df["_time_rounded"] = pd.to_datetime(df[TIME_COL], errors="coerce").dt.floor("1min")
        df["_amt_round"] = pd.to_numeric(df.get(PENCE_COL, 0.0), errors="coerce").fillna(0)
        df = df.sort_values([ID_COL, TIME_COL])
        key = [ID_COL, "_amt_round", "_time_rounded"]
        dup_mask1 = df.duplicated(key, keep="first")
        df["_t_shift"] = df.groupby([ID_COL, "_amt_round"])[TIME_COL].shift(1)
        near_dup = (df["_t_shift"].notna() &
                    ((pd.to_datetime(df[TIME_COL], errors="coerce") - pd.to_datetime(df["_t_shift"], errors="coerce")) <= pd.Timedelta(minutes=5)))
        dup_mask = dup_mask1 | near_dup
        audit["dup_removed"] = int(dup_mask.sum())
        df = df.loc[~dup_mask].drop(columns=["_t_shift"])
        df.drop(columns=["_time_rounded", "_amt_round"], errors="ignore", inplace=True)
    else:
        audit["dup_removed"] = 0

    POSTED_COL = None
    for cand in ["postedDateTime", "postingDateTime", "settledDate", "settlementDateTime"]:
        if cand in df.columns:
            POSTED_COL = cand; break
    audit["posting_lag_mean_days"] = 0.0
    if POSTED_COL:
        t0 = pd.to_datetime(df[TIME_COL], errors="coerce")
        tp = pd.to_datetime(df[POSTED_COL], errors="coerce")
        lag = (tp - t0).dt.days
        audit["posting_lag_mean_days"] = float(pd.to_numeric(lag, errors="coerce").fillna(0).mean())
        df[TIME_COL] = np.where(t0.notna(), t0, tp)

    em = pd.to_numeric(df.get(EM1_COL, np.nan), errors="coerce")
    if em.isna().all() and (EM2_COL in df.columns):
        em = pd.to_numeric(df[EM2_COL], errors="coerce")
    df["emission"] = em
    miss0 = int(df["emission"].isna().sum())
    if miss0 > 0:
        med_user = df.groupby(ID_COL)["emission"].transform("median")
        df["emission"] = df["emission"].fillna(med_user)
        df["emission"] = df["emission"].fillna(df["emission"].median())
    audit["emission_imputed_cnt"] = miss0
    df["em_imputed"] = (em.isna()).astype("int8")

    return df, audit

def compute_metrics_side_by_side(daily_df, partner_vpp=None, low_carbon_cut=None):
    met = {}
    rev = float(daily_df["amt_day"].sum())
    if ("partner_id" in daily_df.columns) and (partner_vpp is not None):
        vmap = partner_vpp.set_index("partner_id")["VPP"].to_dict()
        vpp_eff = daily_df["partner_id"].map(vmap).fillna(VPP_OLD)
        cost = float((daily_df["pts_day"] * vpp_eff).sum())
        vpp_for_liab = float(vpp_eff.mean())
    else:
        cost = float((daily_df["pts_day"] * VPP_OLD).sum())
        vpp_for_liab = VPP_OLD
    met["B_rate"] = safe_ratio(cost, rev, default=0.0)

    emd = pd.to_numeric(daily_df["em_day"], errors="coerce")
    if low_carbon_cut is None and np.isfinite(emd).any():
        low_carbon_cut = float(np.nanpercentile(emd, 40))
    met["low_carbon_share"] = float(np.nanmean(emd <= low_carbon_cut)) if (low_carbon_cut is not None and np.isfinite(low_carbon_cut)) else np.nan

    bal_by_u = (daily_df.sort_values([ID_COL, TIME_COL])
                        .groupby(ID_COL)["balance"].last()
                        .fillna(0.0))
    end_bal = float(bal_by_u.sum())
    met["liability_£"] = end_bal * vpp_for_liab

    met["rows"] = int(len(daily_df))
    met["revenue_£"] = rev
    met["cost_£"] = cost
    met["vpp_used_mean"] = vpp_for_liab
    met["mean_p30"] = float(daily_df["p_base"].mean()) if "p_base" in daily_df.columns else np.nan
    return met

# ---------------- 1) Read & standardize ----------------
df = pd.read_csv(DATA_FILE, low_memory=False)
df.columns = [c.strip() for c in df.columns]
df[TIME_COL] = pd.to_datetime(df[TIME_COL], errors="coerce")
df = df.dropna(subset=[TIME_COL, ID_COL, PENCE_COL])

df["amount"] = pd.to_numeric(df[PENCE_COL], errors="coerce").fillna(0) / 100.0

if EM1_COL in df.columns and df[EM1_COL].notna().any():
    df["emission"] = pd.to_numeric(df[EM1_COL], errors="coerce")
else:
    df["emission"] = pd.to_numeric(df.get(EM2_COL, np.nan), errors="coerce")

df["pts_raw"]    = pd.to_numeric(df.get(PTS_COL0, 0.0), errors="coerce").fillna(0.0)
df["pts_issue"]  = df["pts_raw"].clip(lower=0.0)
df["pts_redeem"] = (-df["pts_raw"].clip(upper=0.0))

df = df.sort_values([ID_COL, TIME_COL])

if FAST:
    date_from = pd.Timestamp.today() - pd.Timedelta(days=DAYS_BACK)
    df = df[df[TIME_COL] >= date_from]
    top_users = (df.groupby(ID_COL)["amount"].sum().nlargest(MAX_USERS).index)
    df = df[df[ID_COL].isin(top_users)].copy()

# ---------------- 2) Daily panel & features ----------------
def build_daily(trans: pd.DataFrame) -> pd.DataFrame:
    assert np.issubdtype(trans[TIME_COL].dtype, np.datetime64)
    g0 = trans.set_index(TIME_COL).groupby(ID_COL)
    daily = (
        g0["amount"].resample("D").sum().rename("amt_day").to_frame()
        .join(g0["pts_issue"].resample("D").sum().rename("pts_day"))
        .join(g0["pts_redeem"].resample("D").sum().rename("pts_redeem_day"))
        .fillna(0.0).reset_index()
    )
    daily["ordered"] = (daily["amt_day"] > 0).astype("int8")

    emis = trans[[ID_COL, TIME_COL, "emission", "amount"]].copy()
    emis["w_em"] = emis["emission"] * emis["amount"]
    em_daily = (emis.set_index(TIME_COL).groupby(ID_COL).resample("D").sum())
    em_daily["em_day"] = np.where(em_daily["amount"] > 0,
                                  em_daily["w_em"] / em_daily["amount"], np.nan)
    em_daily["em_day"] = em_daily.groupby(level=0)["em_day"].ffill().bfill()
    em_daily = em_daily[["em_day"]].reset_index()
    daily = daily.merge(em_daily, on=[ID_COL, TIME_COL], how="left")

    END_DATE = trans[TIME_COL].max().normalize()
    daily = daily.set_index([ID_COL, TIME_COL]).sort_index()

    pads = []
    for uid, g in daily.groupby(level=0, sort=False):
        last_day = g.index.get_level_values(1).max().normalize()
        if last_day < END_DATE:
            rng = pd.date_range(last_day + pd.Timedelta(days=1), END_DATE, freq="D")
            if len(rng):
                pads.append(pd.DataFrame({
                    ID_COL: uid, TIME_COL: rng,
                    "amt_day": 0.0, "pts_day": 0.0, "pts_redeem_day": 0.0, "ordered": 0
                }).set_index([ID_COL, TIME_COL]))
    if pads:
        daily = pd.concat([daily, pd.concat(pads, axis=0)], axis=0).sort_index()

    daily["em_day"] = daily.groupby(level=0)["em_day"].ffill().bfill()
    daily = daily.reset_index()

    daily = daily.sort_values([ID_COL, TIME_COL]).set_index([ID_COL, TIME_COL])
    g = daily.groupby(level=0)
    daily["txn_90"] = (g["ordered"].rolling(90, min_periods=1).sum()).values.astype("int16")
    daily["amt_90"] = (g["amt_day"].rolling(90, min_periods=1).sum()).values
    daily["txn_30"] = (g["ordered"].rolling(30, min_periods=1).sum()).values.astype("int16")
    daily["txn_prev30"] = (g["ordered"].shift(30).rolling(30, min_periods=1).sum()).values
    daily["txn_30_vs_prev30"] = (daily["txn_30"] - daily["txn_prev30"]).fillna(0).astype("int16")

    tNp = g["ordered"].shift(1).rolling(NO_ORDER_DAYS, min_periods=1).sum()
    daily["txn_pastN"] = np.nan_to_num(tNp.values, nan=0.0).astype("int16")

    tLag30 = g["pts_day"].shift(1).rolling(30, min_periods=1).sum()
    daily["pts_lag30"] = np.nan_to_num(tLag30.values, nan=0.0)

    daily["cum_issue"]  = g["pts_day"].cumsum().values
    daily["cum_redeem"] = g["pts_redeem_day"].cumsum().values
    daily["balance"]    = (daily["cum_issue"] - daily["cum_redeem"]).values

    daily["earn_7"]  = (g["pts_day"].rolling(7, min_periods=1).sum()).values
    daily["spend_7"] = (g["amt_day"].rolling(7, min_periods=1).sum()).values
    daily = daily.reset_index()

    daily["order_date"] = daily[TIME_COL].where(daily["ordered"].eq(1))
    daily["last_order_date"] = daily.groupby(ID_COL)["order_date"].ffill()
    rec = (daily[TIME_COL] - daily["last_order_date"]).dt.days
    daily["recency"] = rec.fillna(9999).clip(lower=0).astype("int16")

    daily = daily.sort_values([ID_COL, TIME_COL])
    will = daily.groupby(ID_COL)["ordered"].shift(-1).rolling(30, min_periods=1).sum()
    daily["will_buy_30"] = will.fillna(0).gt(0).astype("int8")

    basket_med = trans.groupby(ID_COL)["amount"].median().rename("basket_med")
    sum_issue = trans.groupby(ID_COL)["pts_issue"].sum()
    sum_amt   = trans.groupby(ID_COL)["amount"].sum().replace(0, np.nan)
    ppp_cust  = (sum_issue / sum_amt).replace([np.inf,-np.inf], np.nan).fillna(0.0).rename("ppp_cust")
    daily = daily.merge(basket_med, on=ID_COL, how="left").merge(ppp_cust, on=ID_COL, how="left")
    global_ppp = safe_ratio(trans["pts_issue"].sum(), trans["amount"].sum(), default=0.0)
    daily["ppp_used"] = daily["ppp_cust"].fillna(global_ppp).clip(lower=0.0)
    daily["basket_med"] = daily["basket_med"].fillna(trans["amount"].median())

    daily["redeem_today"] = (daily["pts_redeem_day"] > 0).astype("int8")
    g2 = daily.groupby(ID_COL)
    fut = g2["redeem_today"].shift(-1).rolling(REDEEM_HORIZON_DAYS, min_periods=1).sum()
    daily["redeem_next_30"] = fut.fillna(0).gt(0).astype("int8")

    step = max(REDEEM_STEP_PTS, 1)
    daily["gap_to_reward"] = (step - (daily["balance"] % step)) % step
    daily["pct_to_next"]   = 1.0 - (daily["gap_to_reward"] / step)
    daily["pct_to_next"]   = clip01(daily["pct_to_next"].values)
    daily["near_reward"]   = (daily["gap_to_reward"] <= step * 0.2).astype("int8")

    edge_eligible = daily["recency"].between(REC_WIN[0], REC_WIN[1]) & (daily["txn_pastN"]==0)
    expo_day_rule = edge_eligible.astype("int8")
    expo_day = np.zeros(len(daily), dtype="int8")
    if EXPOSURE_MODE == "rule":
        expo_day = expo_day_rule.values
    elif EXPOSURE_MODE == "lagpoints":
        expo_day = daily.groupby(ID_COL)["pts_day"].shift(1).fillna(0).gt(0).astype("int8").values
    elif EXPOSURE_MODE == "campaign" and os.path.exists(CAMPAIGN_FILE):
        camp = pd.read_csv(CAMPAIGN_FILE)
        camp[TIME_COL] = pd.to_datetime(camp[TIME_COL], errors="coerce")
        camp = camp[[ID_COL, TIME_COL, "exposed"]].dropna()
        daily = daily.merge(camp, on=[ID_COL, TIME_COL], how="left")
        expo_day = daily["exposed"].fillna(0).astype("int8").values
        daily.drop(columns=["exposed"], inplace=True)
    daily["expo_day"] = expo_day
    daily = daily.sort_values([ID_COL, TIME_COL]).set_index([ID_COL, TIME_COL])
    g3 = daily.groupby(level=0)
    daily["expo_7"]  = (g3["expo_day"].rolling(EXPOSURE_WINDOW, min_periods=1).sum()).values
    daily["expo_30"] = (g3["expo_day"].rolling(30, min_periods=1).sum()).values
    daily = daily.reset_index()
    return daily

daily = build_daily(df)

# === 并行口径审计 ===
try:
    df_naive, audit_naive = preprocess_naive(df)
    df_adj, audit_adj = preprocess_adjusted(df)
    daily_naive = daily.copy()
    daily_adj = build_daily(df_adj)
    met_naive = compute_metrics_side_by_side(daily_naive)
    met_adj   = compute_metrics_side_by_side(daily_adj)
    comp_rows = []
    for k in sorted(set(met_naive.keys()) | set(met_adj.keys())):
        comp_rows.append({
            "metric": k,
            "naive": met_naive.get(k, np.nan),
            "adjusted": met_adj.get(k, np.nan),
            "diff": (met_adj.get(k, np.nan) - met_naive.get(k, np.nan))
                    if (isinstance(met_adj.get(k, np.nan),(int,float)) and isinstance(met_naive.get(k, np.nan),(int,float)))
                    else np.nan
        })
    pd.DataFrame(comp_rows).to_csv(os.path.join(SAVE_DIR, "naive_vs_adjusted_metrics.csv"), index=False)
    with open(os.path.join(SAVE_DIR, "adjust_audit_report.txt"), "w", encoding="utf-8") as f:
        f.write("=== Adjusted pipeline audit ===\n")
        for k,v in audit_adj.items(): f.write(f"{k}: {v}\n")
    print("Saved:", os.path.join(SAVE_DIR, "naive_vs_adjusted_metrics.csv"))
    print("Saved:", os.path.join(SAVE_DIR, "adjust_audit_report.txt"))
except Exception as e:
    print("[Parallel pipelines] skipped initial metrics due to:", e)

# ---------- Sanity check: pts_lag30 ----------
def leakage_sanity_check(daily_df):
    tmp = daily_df.sort_values([ID_COL, TIME_COL]).set_index([ID_COL, TIME_COL])
    g = tmp.groupby(level=0)
    chk = g["pts_day"].shift(1).rolling(30, min_periods=1).sum()
    diff_mean = np.nanmean(np.abs((chk.values - tmp["pts_lag30"].values)))
    if diff_mean > 1e-6:
        print(f"[Sanity] WARNING: pts_lag30 differs from recompute, mean abs diff={diff_mean:.6f}.")
    else:
        print("[Sanity] pts_lag30 matches definition; no look-ahead detected.")
leakage_sanity_check(daily)

# ---------------- 3) MODELS: Baseline vs Effect ----------------
feature_cols_base = ["recency", "txn_90", "amt_90", "txn_30_vs_prev30"]
train_base = daily.dropna(subset=feature_cols_base + ["will_buy_30"]).copy()
cut_time = train_base.sort_values(TIME_COL)[TIME_COL].quantile(0.8)
Xb_tr = train_base[train_base[TIME_COL] < cut_time][feature_cols_base]
yb_tr = train_base[train_base[TIME_COL] < cut_time]["will_buy_30"]
Xb_te = train_base[train_base[TIME_COL] >= cut_time][feature_cols_base]
yb_te = train_base[train_base[TIME_COL] >= cut_time]["will_buy_30"]

clf_base = make_pipeline(StandardScaler(with_mean=False),
                         LogisticRegression(max_iter=300, class_weight="balanced", random_state=SEED))
clf_base.fit(Xb_tr, yb_tr)
auc_base = roc_auc_score(yb_te, clf_base.predict_proba(Xb_te)[:,1])
print("Holdout AUC (baseline, no points):", round(auc_base, 4))

# === 校准器 ===
daily["p_base"] = clf_base.predict_proba(daily[feature_cols_base])[:,1]

def build_calibrator(daily_df, bins=10, mode="decile_map"):
    meta = {"mode": mode, "bins": bins}
    cal = daily_df[[TIME_COL, ID_COL, "p_base", "will_buy_30"]].dropna().copy()
    global_scale = safe_ratio(cal["will_buy_30"].mean(), cal["p_base"].mean(), default=1.0)
    meta["global_scale"] = float(global_scale)
    if mode != "decile_map":
        def _cal_fn(p): return clip01(p * global_scale)
        meta["edges"] = None; meta["act_rates"] = None
        return _cal_fn, meta
    qs = np.linspace(0, 1, bins+1)
    edges = np.unique(np.quantile(cal["p_base"], qs))
    if len(edges) <= 2:
        def _cal_fn(p): return clip01(p * global_scale)
        meta["edges"] = edges.tolist(); meta["act_rates"] = None
        return _cal_fn, meta
    cal["bin"] = np.clip(np.digitize(cal["p_base"], edges[1:-1], right=True), 0, len(edges)-2)
    grp = cal.groupby("bin").agg(avg_p=("p_base","mean"),
                                 act_rate=("will_buy_30","mean"),
                                 n=("will_buy_30","size")).reset_index()
    if (grp["n"] < CALIBRATION_MIN_COUNT_PER_BIN).any():
        def _cal_fn(p): return clip01(p * global_scale)
        meta["edges"] = edges.tolist(); meta["act_rates"] = None
        meta["note"] = "fallback_global_scale_due_to_small_bins"
        return _cal_fn, meta
    ratio = (grp["act_rate"] / grp["avg_p"]).replace([np.inf, -np.inf], np.nan).fillna(global_scale).values
    def _cal_fn(p):
        p = np.asarray(p, dtype=float)
        bins_idx = np.clip(np.digitize(p, edges[1:-1], right=True), 0, len(edges)-2)
        return clip01(p * ratio[bins_idx])
    meta["edges"] = edges.tolist()
    meta["act_rates"] = grp["act_rate"].round(8).tolist()
    meta["avg_p"] = grp["avg_p"].round(8).tolist()
    meta["ratio"] = ratio.round(8).tolist()
    return _cal_fn, meta

# 生成校准表 & 构建校准器
calibration_info = {}
if MAKE_CALIBRATION:
    calib = daily[[TIME_COL, ID_COL, "p_base", "will_buy_30"]].dropna().copy()
    try:
        calib["bucket"] = pd.qcut(calib["p_base"], q=CALIBRATION_BINS, labels=False, duplicates='drop')
        tab = calib.groupby("bucket").agg(avg_p=("p_base","mean"),
                                          act_rate=("will_buy_30","mean"),
                                          n=("will_buy_30","size")).reset_index()
        tab.to_csv(os.path.join(SAVE_DIR, "calibration_table.csv"), index=False)
    except Exception as e:
        print("Calibration table failed:", e)

calibrate_fn, calibration_info = build_calibrator(daily, bins=CALIBRATION_BINS, mode=CALIBRATION_MODE)
with open(os.path.join(SAVE_DIR, "calibration_info.json"), "w", encoding="utf-8") as f:
    json.dump(calibration_info, f, indent=2)

def apply_calibration_if_needed(p):
    return calibrate_fn(p) if USE_CALIBRATED_PROBS else p

# ---- Effect model ----
def train_effect_model(daily_df):
    if EFFECT_SIGNAL == "exposure":
        eff_col = "expo_7"
    elif EFFECT_SIGNAL == "progress":
        eff_col = "pct_to_next"
    else:
        eff_col = PTS_FEATURE
    if EFFECT_BACKBONE == "hazard":
        def _label_buy(df, h):
            g = df.groupby(ID_COL)
            fut = g["ordered"].shift(-1).rolling(h, min_periods=1).sum()
            return fut.fillna(0).gt(0).astype("int8")
        chosen = None; last_reason = ""
        for H in BUY_HORIZONS:
            df2 = daily_df[daily_df["ordered"]==0].copy()
            df2["buy_next"] = _label_buy(daily_df, H).loc[df2.index]
            pos = int(df2["buy_next"].sum()); rate = pos / max(len(df2),1)
            if pos < 100 or rate < 0.001:
                last_reason = f"h={H} insufficient positives ({pos}, {rate:.5f})"; continue
            cut = df2.sort_values(TIME_COL)[TIME_COL].quantile(0.8)
            tr, te = df2[df2[TIME_COL]<cut], df2[df2[TIME_COL]>=cut]
            feats = feature_cols_base + ["balance","pct_to_next","near_reward","expo_7"]
            X_tr = tr[feats + [eff_col]].fillna(0.0)
            y_tr = tr["buy_next"]; X_te = te[feats + [eff_col]].fillna(0.0); y_te = te["buy_next"]
            pipe = make_pipeline(StandardScaler(with_mean=False),
                                 LogisticRegression(max_iter=300, class_weight="balanced", random_state=SEED))
            pipe.fit(X_tr, y_tr)
            auc = roc_auc_score(y_te, pipe.predict_proba(X_te)[:,1])
            coef = pd.Series(pipe.named_steps["logisticregression"].coef_[0], index=feats+[eff_col])
            scaler = pipe.named_steps["standardscaler"]
            scale_map = dict(zip(feats+[eff_col], scaler.scale_))
            b_raw = float(coef[eff_col]); b_used = max(b_raw, 0.0) if FORCE_NONNEG_POINTS_EFFECT else b_raw
            s_eff = float(scale_map.get(eff_col, 1.0)) if np.isfinite(scale_map.get(eff_col,1.0)) else 1.0
            chosen = (b_raw, b_used, s_eff, auc, eff_col, "hazard"); break
        if chosen is None:
            print("[Effect] hazard fallback to classification:", last_reason)
            return train_effect_model_classif(daily_df, eff_col)
        with open(os.path.join(SAVE_DIR, "buy_hazard_auc.txt"), "w", encoding="utf-8") as f:
            f.write(f"Buy-hazard chosen AUC={chosen[3]:.6f}\n")
        return chosen
    else:
        return train_effect_model_classif(daily_df, None)

def train_effect_model_classif(daily_df, eff_col_override=None):
    eff_col = eff_col_override or ("expo_7" if EFFECT_SIGNAL=="exposure" else ("pct_to_next" if EFFECT_SIGNAL=="progress" else PTS_FEATURE))
    feats = feature_cols_base + [eff_col]
    train = daily_df.dropna(subset=feats + ["will_buy_30"]).copy()
    cut = train.sort_values(TIME_COL)[TIME_COL].quantile(0.8)
    Xe_tr = train[train[TIME_COL] < cut][feats]; ye_tr = train[train[TIME_COL] < cut]["will_buy_30"]
    Xe_te = train[train[TIME_COL] >= cut][feats]; ye_te = train[train[TIME_COL] >= cut]["will_buy_30"]
    pipe = make_pipeline(StandardScaler(with_mean=False),
                         LogisticRegression(max_iter=300, class_weight="balanced", random_state=SEED))
    pipe.fit(Xe_tr, ye_tr)
    auc = roc_auc_score(ye_te, pipe.predict_proba(Xe_te)[:,1])
    coef = pd.Series(pipe.named_steps["logisticregression"].coef_[0], index=feats)
    scaler = pipe.named_steps["standardscaler"]
    scale_map = dict(zip(feats, scaler.scale_))
    b_raw = float(coef[eff_col]); b_used = max(b_raw, 0.0) if FORCE_NONNEG_POINTS_EFFECT else b_raw
    s_eff = float(scale_map.get(eff_col, 1.0)) if np.isfinite(scale_map.get(eff_col,1.0)) else 1.0
    print("Holdout AUC (effect model):", round(auc,4), "| eff_col=", eff_col, "| b_raw=", round(b_raw,6))
    return (b_raw, b_used, s_eff, auc, eff_col, "classification")

b_eff_raw, b_eff_used, scale_eff, auc_eff, EFFECT_COL, EFFECT_BACKBONE_USED = train_effect_model(daily)

logit_base = clf_base.named_steps["logisticregression"]
scaler_base = clf_base.named_steps["standardscaler"]
scale_map_base = dict(zip(feature_cols_base, scaler_base.scale_))
def linpred_base(X_base: pd.DataFrame) -> np.ndarray:
    stds = np.array([float(scale_map_base.get(c,1.0) if np.isfinite(scale_map_base.get(c,1.0)) else 1.0)
                     for c in feature_cols_base])
    Z = X_base.values / np.maximum(stds, 1e-12)
    return Z @ logit_base.coef_[0] + logit_base.intercept_[0]

# ---------------- 4) Edge trigger ----------------
cand = daily.copy()
low, high = REC_WIN
mask = cand["recency"].between(low, high) & (cand["txn_pastN"] == 0)
cand = cand[mask].copy()
print(f"[debug] rows in recency window: {int(daily['recency'].between(low, high).sum())}")
print(f"[debug] rows with past{NO_ORDER_DAYS}=0   : {int((daily['txn_pastN']==0).sum())}")
print(f"[debug] candidates selected              : {len(cand)}")

# === [ADD] 仅对“临近奖励”的人天投放（可选）
if NEAR_REWARD_MIN is not None:
    cand = cand[cand["pct_to_next"] >= float(NEAR_REWARD_MIN)].copy()

if cand.empty:
    print("No candidates matched.")
    pd.DataFrame(columns=[ID_COL, TIME_COL, "p0","p1","p0_cal","p1_cal","basket_med",
                          "dpts_offer","cost_£","uplift_£","uplift_cal_£","roi_£","roi_cal_£"]) \
      .to_csv(os.path.join(SAVE_DIR, "offer_triggers.csv"), index=False)
else:
    # --- 基准下一次篮子点数 ---
    cand["base_pts_next"] = (cand["basket_med"] * cand["ppp_used"]).clip(lower=0.0)

    # === [ADD] 额外点数（只对“额外的一份”计成本/效应）
    cand["dpts_offer"] = float(BOOST_MULT) * cand["base_pts_next"]

    # --- p0/p1 ---
    cand["p0"] = clf_base.predict_proba(cand[feature_cols_base])[:,1]
    if EFFECT_SIGNAL == "points":
        if PTS_FEATURE == "pts_day":
            dfeat = cand["dpts_offer"]
        else:
            dfeat = cand["dpts_offer"] if INCENTIVE_MODE=="precredit" else OVERLAP * cand["dpts_offer"]
    elif EFFECT_SIGNAL == "exposure":
        dfeat = pd.Series(EXPO_OFFER_DELTA, index=cand.index)
    else:
        step = max(REDEEM_STEP_PTS,1)
        dfeat = np.minimum(cand["dpts_offer"], cand["gap_to_reward"]) / step

    delta_lin = (dfeat / max(scale_eff,1e-6)) * b_eff_used
    Xb0 = linpred_base(cand[feature_cols_base])
    cand["p1"] = sigmoid(Xb0 + delta_lin)

    # === [ADD] 校准后的 p0/p1
    cand["p0_cal"] = apply_calibration_if_needed(cand["p0"].values)
    cand["p1_cal"] = apply_calibration_if_needed(cand["p1"].values)

    # --- uplift（raw & cal）
    cand["uplift_£"]     = (cand["p1"]     - cand["p0"])     * cand["basket_med"] * MARGIN
    cand["uplift_cal_£"] = (cand["p1_cal"] - cand["p0_cal"]) * cand["basket_med"] * MARGIN

    # === [REPLACE] 成本口径：条件发生 vs 预发 ===
    if OFFER_MODE == "conditional":
        p_for_cost = cand["p1_cal"] if USE_CALIBRATED_PROBS else cand["p1"]
        cand["cost_£"] = (p_for_cost * cand["dpts_offer"] * VPP_OLD).clip(lower=0.0)
    else:
        cand["cost_£"] = (cand["dpts_offer"] * VPP_OLD).clip(lower=0.0)

    # --- ROI（与汇报口径一致）
    cand["roi_£"]     = cand["uplift_£"]     - cand["cost_£"]
    cand["roi_cal_£"] = cand["uplift_cal_£"] - cand["cost_£"]

    df_triggers = cand.sort_values("roi_cal_£" if USE_CALIBRATED_PROBS else "roi_£", ascending=False).copy()
    df_triggers_out = df_triggers[[ID_COL, TIME_COL, "p0","p1","p0_cal","p1_cal","basket_med",
                                   "dpts_offer","cost_£","uplift_£","uplift_cal_£","roi_£","roi_cal_£"]]
    df_triggers_out.to_csv(os.path.join(SAVE_DIR, "offer_triggers.csv"), index=False)
print("Saved:", os.path.join(SAVE_DIR, "offer_triggers.csv"))

# === [REPLACE] 策略、曲线与摘要（自适应阈值 + 选择/汇报分离） ===
if 'df_triggers' in locals():
    df_pol = df_triggers.copy()

    roi_col_for_report  = "roi_cal_£" if USE_CALIBRATED_PROBS else "roi_£"
    roi_col_for_select  = ("roi_cal_£" if (USE_CALIBRATED_PROBS and SELECT_WITH_CALIBRATION) else "roi_£")

    try:
        print("ROI raw quantiles:\n", df_triggers["roi_£"].quantile([0,.25,.5,.75,.9,.95,.99]).round(6))
        if "roi_cal_£" in df_triggers:
            print("ROI cal quantiles:\n", df_triggers["roi_cal_£"].quantile([0,.25,.5,.75,.9,.95,.99]).round(6))
        print("Count raw>0:", int((df_triggers["roi_£"]>0).sum()),
              "cal>0:", int((df_triggers.get("roi_cal_£", pd.Series([]))>0).sum()))
    except Exception:
        pass

    thr = float(MIN_ROI_THRESHOLD) if (MIN_ROI_THRESHOLD is not None) else None
    if (thr is not None) and USE_CALIBRATED_PROBS and ADAPT_THRESHOLD_TO_CALIBRATION:
        calib_scale = float(calibration_info.get("global_scale", 1.0))
        thr = thr * max(calib_scale, 1e-9)
        print(f"[threshold] adapted to calibration scale={calib_scale:.6f} -> thr={thr:.6f}")

    df_pol = df_pol.sort_values(roi_col_for_select, ascending=False).copy()
    if thr is not None:
        df_pol = df_pol[df_pol[roi_col_for_select] >= thr]

    if COOLDOWN_DAYS and COOLDOWN_DAYS > 0 and len(df_pol):
        keep_idx, last_date = [], {}
        for i, row in df_pol.iterrows():
            uid = row[ID_COL]; d = pd.to_datetime(row[TIME_COL]).normalize()
            prev = last_date.get(uid)
            if (prev is None) or ((d - prev).days >= COOLDOWN_DAYS):
                keep_idx.append(i); last_date[uid] = d
        df_pol = df_pol.loc[keep_idx]

    if len(df_pol):
        curve_tmp = df_pol[["cost_£", ("uplift_cal_£" if USE_CALIBRATED_PROBS else "uplift_£")]].cumsum()
        curve_tmp.columns = ["cost_£","uplift_sel_£"]; curve_tmp["net_£"] = curve_tmp["uplift_sel_£"] - curve_tmp["cost_£"]
        if (BUDGET_LIMIT_GBP is not None):
            within = np.where(curve_tmp["cost_£"].values <= BUDGET_LIMIT_GBP)[0]
            if len(within):
                df_pol = df_pol.iloc[:within[-1]+1]

    pol_path = os.path.join(SAVE_DIR, "offer_triggers_policy.csv")
    df_pol.to_csv(pol_path, index=False); print("Saved:", pol_path)

    uplift_col_report = "uplift_cal_£" if USE_CALIBRATED_PROBS else "uplift_£"
    curve = df_pol[["cost_£", uplift_col_report]].cumsum()
    curve.columns = ["cost_£","uplift_sel_£"]; curve["net_£"] = curve["uplift_sel_£"] - curve["cost_£"]
    idx_breakeven = int(np.argmax(curve["net_£"].values >= 0)) if (curve["net_£"] >= 0).any() else None

    plt.figure(figsize=(7,4), dpi=160)
    plt.plot(curve["cost_£"].values, curve["uplift_sel_£"].values)
    plt.xlabel("Cumulative cost (£)"); plt.ylabel("Cumulative uplift (£)")
    plt.title(f"Edge-trigger Cost–Benefit curve ({'calibrated' if USE_CALIBRATED_PROBS else 'raw'})")
    if idx_breakeven is not None and idx_breakeven < len(curve):
        x_be = curve.iloc[idx_breakeven]["cost_£"]; y_be = curve.iloc[idx_breakeven]["uplift_sel_£"]
        plt.axvline(x=x_be, linestyle="--"); plt.text(x_be, y_be, "  breakeven", va="bottom")
    if (BUDGET_LIMIT_GBP is not None) and len(curve):
        within = np.where(curve["cost_£"].values <= BUDGET_LIMIT_GBP)[0]
        if len(within):
            x_bd = curve.iloc[within[-1]]["cost_£"]; y_bd = curve.iloc[within[-1]]["uplift_sel_£"]
            plt.axvline(x=x_bd, linestyle=":"); plt.text(x_bd, y_bd, "  budget", va="bottom")
    plt.tight_layout(); plt.savefig(os.path.join(SAVE_DIR, "cost_benefit_curve_annotated.png")); plt.close()

    curve_raw = df_triggers.sort_values("roi_£", ascending=False)[["cost_£", "uplift_£"]].cumsum()
    plt.figure(figsize=(7,4), dpi=160)
    plt.plot(curve_raw["cost_£"].values, curve_raw["uplift_£"].values)
    plt.xlabel("Cumulative cost (£)"); plt.ylabel("Cumulative uplift (£)")
    plt.title("Edge-trigger Cost–Benefit curve (raw)")
    plt.tight_layout(); plt.savefig(os.path.join(SAVE_DIR, "cost_benefit_curve.png")); plt.close()

    tot_uplift_sel = curve["uplift_sel_£"].iloc[-1] if len(curve) else 0.0
    tot_cost = curve["cost_£"].iloc[-1] if len(curve) else 0.0
    tot_net  = tot_uplift_sel - tot_cost
    with open(os.path.join(SAVE_DIR, "trigger_summary.txt"), "w", encoding="utf-8") as f:
        f.write(f"触发候选(策略后)条数: {len(df_pol):,}\n")
        if len(df_pol):
            f.write(f"正向ROI条数: {(df_pol[roi_col_for_report] > 0).sum():,}（占比 {100 * (df_pol[roi_col_for_report] > 0).mean():.1f}%）\n")
        else:
            f.write("正向ROI条数: 0（占比 0.0%）\n")
        f.write(f"累计成本(£): {tot_cost:,.2f}\n累计毛利提升(£): {tot_uplift_sel:,.2f}\n累计净收益(£): {tot_net:,.2f}\n")

# ---------------- 5) Dynamic points（成本率守恒 + BP-tilt） ----------------
hist_cost = (daily["pts_day"] * VPP_OLD).sum()
hist_rev  = daily["amt_day"].sum()
B_rate    = safe_ratio(hist_cost, hist_rev, default=0.0)
print("Historical cost rate:", round(B_rate, 6))

def _tilting_weights_bp(em_norm, recency, pbase, basket_med, vpp_new, b_pts_like, scale_like):
    grad_prob = pbase * (1 - pbase) * (b_pts_like / max(scale_like,1e-6))
    bp = grad_prob * basket_med * MARGIN / max(vpp_new, 1e-9)
    bp = np.nan_to_num(bp, nan=0.0)
    norm_bp = bp / (np.mean(bp) + 1e-9)
    tilt_bp = 1 + BP_TILT * (norm_bp - 1) if BP_TILT > 0 else 1.0
    if REC_TILT > 0:
        L, H = REC_WIN; center = 0.5 * (L + H); width = max(H - L, 1)
        rec_score = np.exp(-((recency - center) / (0.6 * width))**2)
        rec_score = rec_score / (rec_score.mean() + 1e-9)
        tilt_rec = 1 + REC_TILT * (rec_score - 1)
    else:
        tilt_rec = 1.0
    return tilt_bp * tilt_rec, bp

def calibrate_and_predict(alpha, beta, VPP_NEW, out_dir_for_plots=None):
    em = daily["em_day"].fillna(daily["em_day"].median()).astype(float)
    if EM_NORMALIZE:
        em_min = np.nanmin(em.values); em_max = np.nanmax(em.values)
        denom_em = max(em_max - em_min, 1e-9)
        em_norm = (em - em_min) / denom_em; em_norm = clip01(em_norm.values)
    else:
        em_norm = em.values

    b_like = b_eff_used; scale_like = scale_eff
    tilt, bp = _tilting_weights_bp(em_norm, daily["recency"].values.astype(float),
                                   daily["p_base"].values.astype(float),
                                   daily["basket_med"].values.astype(float),
                                   VPP_NEW, b_like, scale_like)

    denom = (1 + alpha * em_norm)**beta
    denom_eff = denom / tilt

    sum_amt = daily["amt_day"].sum()
    sum_amt_over = (daily["amt_day"] / denom_eff).sum()
    k = safe_ratio(B_rate * sum_amt, VPP_NEW * max(sum_amt_over, 1e-9), default=0.0)
    pts_new = k * daily["amt_day"] / denom_eff

    if PTS_FEATURE == "pts_day":
        feat_new = pts_new.values
    else:
        tmp = daily[[ID_COL, TIME_COL]].copy()
        tmp["pts_new"] = pts_new
        tmp = tmp.sort_values([ID_COL, TIME_COL]).set_index([ID_COL, TIME_COL])
        feat_new = tmp.groupby(level=0)["pts_new"].shift(1).rolling(30, min_periods=1).sum().values
        feat_new = np.nan_to_num(feat_new, nan=0.0)

    Xb0 = linpred_base(daily[feature_cols_base])
    base_std_pts = daily[PTS_FEATURE].values / max(scale_like, 1e-6)
    new_std_pts  = feat_new / max(scale_like, 1e-6)
    delta_lin    = (new_std_pts - base_std_pts) * (b_like)
    p_dyn = 1 / (1 + np.exp(-(Xb0 + delta_lin)))

    out_dir = out_dir_for_plots or SAVE_DIR
    try:
        w = np.where(daily["amt_day"].values > 0, pts_new.values / (k * daily["amt_day"].values + 1e-12), 0.0)
        plt.figure(figsize=(6,4), dpi=160); plt.hist(w[np.isfinite(w) & (w>0)], bins=40)
        plt.xlabel("Allocation weight w = 1/denom_eff"); plt.ylabel("Count"); plt.title("Distribution of allocation weights")
        plt.tight_layout(); plt.savefig(os.path.join(out_dir, "allocation_weight_hist.png")); plt.close()

        idx = np.random.RandomState(0).choice(len(w), size=min(8000, len(w)), replace=False)
        plt.figure(figsize=(6,4), dpi=160); plt.scatter(em_norm[idx], w[idx], s=6, alpha=0.35)
        plt.xlabel("Normalized emission"); plt.ylabel("Allocation weight (sample)"); plt.title("Weight vs Emission")
        plt.tight_layout(); plt.savefig(os.path.join(out_dir, "weight_vs_emission.png")); plt.close()

        plt.figure(figsize=(6,4), dpi=160); plt.scatter(daily["recency"].values[idx], w[idx], s=6, alpha=0.35)
        plt.xlabel("Recency (days)"); plt.ylabel("Allocation weight (sample)"); plt.title("Weight vs Recency")
        plt.tight_layout(); plt.savefig(os.path.join(out_dir, "weight_vs_recency.png")); plt.close()

        tmp2 = pd.DataFrame({"p_base": daily["p_base"].values, "p_dyn": p_dyn}); tmp2["delta"] = tmp2["p_dyn"] - tmp2["p_base"]
        tmp2["q"] = pd.qcut(tmp2["p_base"], q=10, labels=False, duplicates="drop")
        dec = tmp2.groupby("q").agg(mean_p_base=("p_base","mean"), mean_delta=("delta","mean"), n=("delta","size")).reset_index()
        dec.to_csv(os.path.join(out_dir, "uplift_by_pbase_quantile.csv"), index=False)

        pd.DataFrame({"bp_score": bp}).to_csv(os.path.join(out_dir, "bp_score.csv"), index=False)
    except Exception as e:
        print("[Explainability] plot/table failed:", e)

    # 同步输出“校准后均值”，便于论文口径
    p_base_cal = calibrate_fn(daily["p_base"].values) if USE_CALIBRATED_PROBS else daily["p_base"].values
    p_dyn_cal  = calibrate_fn(p_dyn) if USE_CALIBRATED_PROBS else p_dyn
    return k, pts_new, p_dyn, p_base_cal, p_dyn_cal

def run_dynamic_for_gamma(gamma_value, out_dir_suffix):
    VPP_NEW = gamma_value * VPP_OLD
    alpha_grid = np.array([0.0, 0.0015, 0.003, 0.006, 0.01], dtype=float)
    beta_grid  = np.array([0.5], dtype=float)

    out_dir = os.path.join(SAVE_DIR, out_dir_suffix); os.makedirs(out_dir, exist_ok=True)

    base_mean = float(np.mean(daily["p_base"]))
    delta_mat = np.zeros((len(alpha_grid), len(beta_grid)), dtype=float)
    for i, a in enumerate(alpha_grid):
        for j, b in enumerate(beta_grid):
            _, _, p_dyn_ab, _, _ = calibrate_and_predict(a, b, VPP_NEW, out_dir_for_plots=out_dir if (i==0 and j==0) else None)
            delta_mat[i, j] = float(np.mean(p_dyn_ab) - base_mean)

    pd.DataFrame(delta_mat, index=[f"alpha={a}" for a in alpha_grid], columns=[f"beta={b}" for b in beta_grid]) \
      .to_csv(os.path.join(out_dir, "sensitivity_table.csv"), float_format="%.6f", index=True)

    fig, ax = plt.subplots(figsize=(6,5), dpi=160)
    im = ax.imshow(delta_mat, aspect="auto", origin="lower")
    ax.set_xticks(range(len(beta_grid))); ax.set_yticks(range(len(alpha_grid)))
    ax.set_xticklabels([str(x) for x in beta_grid]); ax.set_yticklabels([str(x) for x in alpha_grid])
    ax.set_xlabel("beta"); ax.set_ylabel("alpha")
    ax.set_title(f"Mean Δp (gamma={gamma_value}, signal={EFFECT_SIGNAL})")
    for i in range(len(alpha_grid)):
        for j in range(len(beta_grid)):
            ax.text(j, i, f"{delta_mat[i,j]:.3f}", ha="center", va="center")
    plt.tight_layout(); plt.savefig(os.path.join(out_dir, "sensitivity_heatmap.png")); plt.close()

    k_new, pts_new, p_dyn, p_base_cal, p_dyn_cal = calibrate_and_predict(ALPHA, BETA, VPP_NEW, out_dir_for_plots=out_dir)

    summary = {
        "gamma": float(gamma_value),
        "mean_prob_base": float(np.mean(daily["p_base"])),
        "mean_prob_dyn":  float(np.mean(p_dyn)),
        "delta_prob":     float(np.mean(p_dyn) - np.mean(daily["p_base"])),
        "mean_prob_base_cal": float(np.mean(p_base_cal)),
        "mean_prob_dyn_cal":  float(np.mean(p_dyn_cal)),
        "delta_prob_cal":     float(np.mean(p_dyn_cal) - np.mean(p_base_cal)),
        "cost_fixed_£":   float((daily["pts_day"] * VPP_OLD).sum()),
        "cost_dynamic_£": float((pts_new * VPP_NEW).sum()),
        "revenue_£":      float(daily["amt_day"].sum()),
        "k_new":          float(k_new),
        "vpp_old":        float(VPP_OLD),
        "vpp_new":        float(VPP_NEW),
        "alpha":          float(ALPHA),
        "beta":           float(BETA),
        "effect_col":     EFFECT_COL,
        "effect_backbone":EFFECT_BACKBONE_USED,
        "b_eff_fitted":   float(b_eff_raw),
        "b_eff_used":     float(b_eff_used),
        "bp_tilt":        float(BP_TILT),
        "rec_tilt":       float(REC_TILT),
        "auc_base":       float(auc_base),
        "auc_eff":        float(auc_eff),
        "discount_factor": float(DISCOUNT_FACTOR),
        "cpp_redeem":      float(CPP_REDEEM),
        "cpp_issued":      float(VPP_OLD),
        "face_value_per_point": float(FACE_VALUE_PER_POINT),
        "expected_redeem_rate": float(EXPECTED_REDEEM_RATE),
        "use_calibrated_probs": bool(USE_CALIBRATED_PROBS),
        "calibration_mode": CALIBRATION_MODE,
        "calibration_bins": int(CALIBRATION_BINS),
    }
    pd.DataFrame([summary]).to_csv(os.path.join(out_dir, "dynamic_points_summary.csv"), index=False)

    base_issue_by_user = daily.groupby(ID_COL)["pts_day"].sum()
    redeem_by_user     = daily.groupby(ID_COL)["pts_redeem_day"].sum()
    end_bal_base_by_u  = base_issue_by_user - redeem_by_user
    base_end_bal_sum   = float(end_bal_base_by_u.sum())
    tmp_dyn = daily[[ID_COL, TIME_COL]].copy(); tmp_dyn["pts_new"] = pts_new.values
    issue_dyn_by_user = tmp_dyn.groupby(ID_COL)["pts_new"].sum()
    end_bal_dyn_by_u  = issue_dyn_by_user - redeem_by_user
    dyn_end_bal_sum   = float(end_bal_dyn_by_u.sum())

    liability_base = base_end_bal_sum * VPP_NEW
    liability_dyn  = dyn_end_bal_sum  * VPP_NEW
    liab_df = pd.DataFrame([{
        "gamma": gamma_value,
        "vpp_issued_new": VPP_NEW,
        "ending_balance_pts_base": base_end_bal_sum,
        "ending_balance_pts_dynamic": dyn_end_bal_sum,
        "liability_est_base_£": liability_base,
        "liability_est_dynamic_£": liability_dyn,
        "delta_liability_£": liability_dyn - liability_base,
    }])
    liab_df.to_csv(os.path.join(out_dir, "liability_report.csv"), index=False)

for gam in GAMMA_LIST:
    tag = f"gamma_{gam:.2f}".replace(".", "_"); run_dynamic_for_gamma(gam, tag)

# ---------------- 6) Redemption hazard（保留回退） ----------------
def _has_two_classes(y):
    y = np.asarray(y); return len(np.unique(y)) >= 2

def _build_redeem_label(df_in: pd.DataFrame, horizon: int) -> pd.Series:
    df2 = df_in.sort_values([ID_COL, TIME_COL]).copy(); g = df2.groupby(ID_COL)
    fut = g["redeem_today"].shift(-1).rolling(horizon, min_periods=1).sum()
    return fut.fillna(0).gt(0).astype("int8")

def train_redeem_hazard_autosel(daily_df: pd.DataFrame, horizons=(30, 60, 90), min_pos=50, min_rate=0.001):
    hazard_feats = ["balance", "gap_to_reward", "pct_to_next", "near_reward", "recency", "earn_7", "spend_7", "txn_30", "txn_90"]
    chosen=None; last_reason="unknown"
    for H in horizons:
        df2 = daily_df.copy(); df2["redeem_next"] = _build_redeem_label(df2, H)
        pos_all = int(df2["redeem_next"].sum()); rate_all = pos_all / max(len(df2),1)
        if pos_all < min_pos or rate_all < min_rate:
            last_reason = f"h={H}: positives={pos_all}, rate={rate_all:.6f} too low"; continue
        cut = df2.sort_values(TIME_COL)[TIME_COL].quantile(0.8); tr = df2[df2[TIME_COL] < cut]; te = df2[df2[TIME_COL] >= cut]
        y_tr = tr["redeem_next"]; y_te = te["redeem_next"]
        if (not _has_two_classes(y_tr)) or (not _has_two_classes(y_te)):
            last_reason = f"h={H}: single class"; continue
        X_tr = tr[hazard_feats].fillna(0.0); X_te = te[hazard_feats].fillna(0.0)
        pipe = make_pipeline(StandardScaler(with_mean=False),
                             LogisticRegression(max_iter=300, class_weight="balanced", random_state=SEED))
        pipe.fit(X_tr, y_tr); auc = roc_auc_score(y_te, pipe.predict_proba(X_te)[:, 1])
        chosen = (H, pipe, auc, hazard_feats); print(f"[RedeemHazard] horizon={H}, AUC={auc:.4f}, positives={pos_all}"); break
    auc_path = os.path.join(SAVE_DIR, "redeem_hazard_auc.txt")
    if chosen is None:
        with open(auc_path, "w", encoding="utf-8") as f:
            f.write("Redeem hazard: insufficient positives; last_reason=" + last_reason + "\n")
        print("[RedeemHazard] skipped:", last_reason); return None
    H, model, auc, feats = chosen
    with open(auc_path, "w", encoding="utf-8") as f:
        f.write(f"Redeem-next-{H}-days AUC: {auc:.6f}\n")
    hz_all = daily_df[[TIME_COL] + feats].copy()
    hz_all["p_hazard"] = model.predict_proba(hz_all[feats].fillna(0.0))[:, 1]
    hz_all["bucket"] = pd.qcut(hz_all["p_hazard"], q=10, labels=False, duplicates="drop")
    dec_hz = hz_all.groupby("bucket").agg(avg_p=("p_hazard","mean"), n=("p_hazard","size")).reset_index()
    hz_all["redeem_next"] = _build_redeem_label(daily_df, H)
    dec_hz["act_rate"] = hz_all.groupby("bucket")["redeem_next"].mean().values
    dec_hz.to_csv(os.path.join(SAVE_DIR, "redeem_hazard_deciles.csv"), index=False)
    with open(os.path.join(SAVE_DIR, "redeem_hazard_chosen_horizon.txt"), "w", encoding="utf-8") as f:
        f.write(str(H) + "\n")
    return {"horizon": H, "auc": auc}

_ = train_redeem_hazard_autosel(daily, horizons=(30, 60, 90), min_pos=50, min_rate=0.001)

# ---------------- Integrity & run_config ----------------
with open(os.path.join(SAVE_DIR, "data_integrity_report.txt"), "w", encoding="utf-8") as f:
    f.write("=== Data integrity summary ===\n")
    f.write(f"rows: {len(df):,}\n")
    f.write(f"sum pts_raw: {df['pts_raw'].sum():,.2f}\n")
    f.write(f"sum pts_issue(>=0): {df['pts_issue'].sum():,.2f}\n")
    f.write(f"sum pts_redeem(>=0): {df['pts_redeem'].sum():,.2f}\n")
    f.write(f"amount sum(£): {df['amount'].sum():,.2f}\n")
    f.write(f"global ppp_used: {safe_ratio(df['pts_issue'].sum(), df['amount'].sum()):.6f}\n")

run_config = dict(
    FACE_VALUE_PER_POINT=FACE_VALUE_PER_POINT, DISCOUNT_FACTOR=DISCOUNT_FACTOR, EXPECTED_REDEEM_RATE=EXPECTED_REDEEM_RATE,
    CPP_REDEEM=CPP_REDEEM, VPP_OLD=VPP_OLD, MARGIN=MARGIN,
    REC_WIN=REC_WIN, NO_ORDER_DAYS=NO_ORDER_DAYS, ALPHA=ALPHA, BETA=BETA, GAMMA_LIST=GAMMA_LIST,
    EFFECT_BACKBONE=EFFECT_BACKBONE, EFFECT_SIGNAL=EFFECT_SIGNAL,
    PTS_FEATURE=PTS_FEATURE, INCENTIVE_MODE=INCENTIVE_MODE,
    EM_NORMALIZE=EM_NORMALIZE, FORCE_NONNEG_POINTS_EFFECT=FORCE_NONNEG_POINTS_EFFECT,
    BP_TILT=BP_TILT, REC_TILT=REC_TILT, BUY_HORIZONS=BUY_HORIZONS,
    REDEEM_HORIZON_DAYS=REDEEM_HORIZON_DAYS, REDEEM_STEP_PTS=REDEEM_STEP_PTS,
    FAST=FAST, MAX_USERS=MAX_USERS, DAYS_BACK=DAYS_BACK,
    COOLDOWN_DAYS=COOLDOWN_DAYS, MIN_ROI_THRESHOLD=MIN_ROI_THRESHOLD, BUDGET_LIMIT_GBP=BUDGET_LIMIT_GBP,
    OFFER_MODE=OFFER_MODE, BOOST_MULT=BOOST_MULT, NEAR_REWARD_MIN=NEAR_REWARD_MIN,
    MAKE_CALIBRATION=MAKE_CALIBRATION, USE_CALIBRATED_PROBS=USE_CALIBRATED_PROBS,
    CALIBRATION_MODE=CALIBRATION_MODE, CALIBRATION_BINS=CALIBRATION_BINS, CALIBRATION_MIN_COUNT_PER_BIN=CALIBRATION_MIN_COUNT_PER_BIN,
    SELECT_WITH_CALIBRATION=SELECT_WITH_CALIBRATION, ADAPT_THRESHOLD_TO_CALIBRATION=ADAPT_THRESHOLD_TO_CALIBRATION,
    SEED=SEED
)
with open(os.path.join(SAVE_DIR, "run_config.json"), "w", encoding="utf-8") as f:
    json.dump(run_config, f, indent=2)

print("All outputs ready under:", SAVE_DIR)
print("Dynamic summaries under:", [os.path.join(SAVE_DIR, f"gamma_{g:.2f}".replace('.','_')) for g in GAMMA_LIST])
print(f"[Info] EFFECT_BACKBONE={EFFECT_BACKBONE_USED}, EFFECT_SIGNAL={EFFECT_SIGNAL}, EFFECT_COL={EFFECT_COL}")
print(f"[Cost params] face_value={FACE_VALUE_PER_POINT:.4f}, delta={DISCOUNT_FACTOR:.2f}, CPP_red={CPP_REDEEM:.4f}, redeem_rate={EXPECTED_REDEEM_RATE:.2f}, CPP_iss={VPP_OLD:.4f}")
