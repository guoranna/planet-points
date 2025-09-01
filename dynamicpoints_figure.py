# -*- coding: utf-8 -*-
# 动态积分（成本率守恒）图表生成脚本
# 依赖：pandas, matplotlib
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

# ========== 1) 在这里填入 4 个 CSV 的路径（gamma=1.00 / 1.15 各一组） ==========
g100_paths = {
    "summary":  r"outputs/gamma_1_00/dynamic_points_summary.csv",
    "liability":r"outputs/gamma_1_00/liability_report.csv",
    "decile":   r"outputs/gamma_1_00/uplift_by_pbase_quantile.csv",
    "sens":     r"outputs/gamma_1_00/sensitivity_table.csv",
}
g115_paths = {
    "summary":  r"outputs/gamma_1_15/dynamic_points_summary.csv",
    "liability":r"outputs/gamma_1_15/liability_report.csv",
    "decile":   r"outputs/gamma_1_15/uplift_by_pbase_quantile.csv",
    "sens":     r"outputs/gamma_1_15/sensitivity_table.csv",
}

# 输出目录
outdir = Path("figs_dynamic_policy")
outdir.mkdir(parents=True, exist_ok=True)

# ========== 2) 读数的通用函数（容错列名） ==========
def read_dynamic_summary(path):
    df = pd.read_csv(path)
    # 取第一行（整体汇总）
    row = df.iloc[0]

    def pick(series_like, candidates, default=None):
        for c in candidates:
            if c in series_like.index:
                return series_like[c]
        return default

    mean_base = pick(row, ["mean_prob_base", "mean_p_base", "p_base_mean"])
    mean_dyn  = pick(row, ["mean_prob_dyn", "mean_p_dyn", "p_dyn_mean"])
    delta_prob = pick(row, ["delta_prob", "mean_delta_prob", "delta_p"])
    k_new = pick(row, ["k_new", "k_scale", "scale"])
    auc_base = pick(row, ["auc_base", "base_auc"])
    auc_eff  = pick(row, ["auc_eff", "effect_auc"])
    cost_dev_bp = pick(row, ["cost_rate_deviation_bp", "cost_dev_bp", "cost_rate_bp"], 0.0)

    return {
        "mean_prob_base": float(mean_base),
        "mean_prob_dyn":  float(mean_dyn),
        "delta_prob":     float(delta_prob),   # 概率单位，后续 ×100 报 pp
        "k_new":          float(k_new),
        "auc_base":       float(auc_base),
        "auc_eff":        float(auc_eff),
        "cost_dev_bp":    float(cost_dev_bp),
    }

def read_liability(path):
    df = pd.read_csv(path)
    row = df.iloc[0]
    def pick(series_like, candidates, default=None):
        for c in candidates:
            if c in series_like.index:
                return series_like[c]
        return default
    end_base = pick(row, ["ending_balance_base", "ending_pts_base", "balance_base"])
    end_dyn  = pick(row, ["ending_balance_dyn", "ending_pts_dyn", "balance_dyn"])
    d_liab  = pick(row, ["delta_liability", "delta_liability_gbp", "liability_delta_gbp", "liability_change_gbp"], 0.0)
    vpp_old = pick(row, ["vpp_old", "VPP_old"], None)
    vpp_new = pick(row, ["vpp_new", "VPP_new"], None)
    return {
        "ending_balance_base": float(end_base),
        "ending_balance_dyn":  float(end_dyn),
        "delta_liability_gbp": float(d_liab),
        "vpp_old": vpp_old if vpp_old is None else float(vpp_old),
        "vpp_new": vpp_new if vpp_new is None else float(vpp_new),
    }

def read_decile(path):
    df = pd.read_csv(path)
    # 统一列名
    col_decile = [c for c in df.columns if c.lower() in {"decile","q","quantile"}][0]
    col_pbase  = [c for c in df.columns if "mean" in c.lower() and "base" in c.lower()][0]
    col_dprob  = [c for c in df.columns if ("delta" in c.lower() and "prob" in c.lower()) or c.lower()=="mean_delta"][0]
    col_n =     [c for c in df.columns if c.lower() in {"n","count","size"}][0]
    out = df[[col_decile, col_pbase, col_dprob, col_n]].copy()
    out.columns = ["decile","mean_p_base","mean_delta_prob","n"]
    # 转换为百分比点（pp）
    out["delta_pp"] = out["mean_delta_prob"] * 100.0
    out["pbase_pct"] = out["mean_p_base"] * 100.0
    return out.sort_values("decile")

def read_sensitivity(path, beta_target=0.5):
    df = pd.read_csv(path)
    # 兼容列名
    col_alpha = [c for c in df.columns if c.lower()=="alpha"][0]
    col_beta  = [c for c in df.columns if c.lower()=="beta"][0]
    col_mean  = [c for c in df.columns if ("mean" in c.lower() and "prob" in c.lower()) or c.lower()=="mean_delta"][0]
    d = df[df[col_beta]==beta_target].copy()
    d = d[[col_alpha, col_mean]].copy()
    d.columns = ["alpha","mean_delta_prob"]
    d["delta_pp"] = d["mean_delta_prob"] * 100.0
    return d.sort_values("alpha")

# ========== 3) 读取两组数据 ==========
g100_sum = read_dynamic_summary(g100_paths["summary"])
g115_sum = read_dynamic_summary(g115_paths["summary"])
g100_liab = read_liability(g100_paths["liability"])
g115_liab = read_liability(g115_paths["liability"])
g100_dec = read_decile(g100_paths["decile"])
g115_dec = read_decile(g115_paths["decile"])
g100_sens = read_sensitivity(g100_paths["sens"], beta_target=0.5)
g115_sens = read_sensitivity(g115_paths["sens"], beta_target=0.5)

# 便捷数值（pp 和 负债变化）
g100_pp = g100_sum["delta_prob"] * 100.0
g115_pp = g115_sum["delta_prob"] * 100.0
g100_dliab = g100_liab["delta_liability_gbp"]
g115_dliab = g115_liab["delta_liability_gbp"]

# ========== 4) 图1：参与度—负债的权衡曲线（两点散点） ==========
fig1 = plt.figure()
plt.scatter([g100_dliab, g115_dliab], [g100_pp, g115_pp])
for x, y, label in [(g100_dliab, g100_pp, "γ=1.00"),
                    (g115_dliab, g115_pp, "γ=1.15")]:
    plt.annotate(label, (x, y), xytext=(5, 5), textcoords="offset points")
plt.axhline(0, linewidth=1)
plt.xlabel("Δ liability (GBP)")
plt.ylabel("Δ purchase probability (pp)")
plt.title("Participation–Liability trade-off (cost-rate conserved)")
plt.tight_layout()
fig1.savefig(outdir / "tradeoff_deltaP_vs_deltaLiability.png", dpi=200)

# ========== 5) 图2：十分位异质性（两条折线对照） ==========
fig2 = plt.figure()
plt.plot(g100_dec["decile"], g100_dec["delta_pp"], label="γ=1.00")
plt.plot(g115_dec["decile"], g115_dec["delta_pp"], label="γ=1.15")
plt.axhline(0, linewidth=1)
plt.xlabel("Baseline-propensity decile (q)")
plt.ylabel("Δp (pp)")
plt.title("Decile heterogeneity of lift")
plt.legend()
plt.tight_layout()
fig2.savefig(outdir / "decile_heterogeneity_gamma100_vs_gamma115.png", dpi=200)

# 若你更喜欢分别出图，可改用以下两段（各自一张图）：
# fig2a = plt.figure(); plt.bar(g100_dec["decile"], g100_dec["delta_pp"]); ...
# fig2b = plt.figure(); plt.bar(g115_dec["decile"], g115_dec["delta_pp"]); ...

# ========== 6) 图3：负债桥/结余对比（γ=1.15） ==========
# 使用条形图展示 base vs dyn 的结余（pts）与负债变化（GBP）。
# 这里做两张单独图：一张“结余”，一张“负债变化”，各自一个 Figure。
# 3A 期末积分结余
fig3a = plt.figure()
plt.bar(["Base", "Dynamic"], [g115_liab["ending_balance_base"], g115_liab["ending_balance_dyn"]])
plt.xlabel("State")
plt.ylabel("Ending point balance (pts)")
plt.title("Ending balance (γ=1.15)")
plt.tight_layout()
fig3a.savefig(outdir / "ending_balance_gamma115.png", dpi=200)

# 3B 账面负债变化（GBP）
fig3b = plt.figure()
plt.bar(["Δ liability"], [g115_liab["delta_liability_gbp"]])
plt.xlabel("Metric")
plt.ylabel("GBP")
plt.title("Book liability change (γ=1.15)")
plt.tight_layout()
fig3b.savefig(outdir / "delta_liability_gamma115.png", dpi=200)

# （可选）γ=1.00 的对照图（会显示 0 变化）
fig3c = plt.figure()
plt.bar(["Base", "Dynamic"], [g100_liab["ending_balance_base"], g100_liab["ending_balance_dyn"]])
plt.xlabel("State")
plt.ylabel("Ending point balance (pts)")
plt.title("Ending balance (γ=1.00)")
plt.tight_layout()
fig3c.savefig(outdir / "ending_balance_gamma100.png", dpi=200)

# ========== 7) 图4：稳健性（α-敏感性，β=0.5） ==========
# 两条折线：γ=1.00 与 γ=1.15 的 mean Δp(pp) 随 α 的变化
fig4 = plt.figure()
plt.plot(g100_sens["alpha"], g100_sens["delta_pp"], label="γ=1.00")
plt.plot(g115_sens["alpha"], g115_sens["delta_pp"], label="γ=1.15")
plt.xlabel("alpha (low-carbon tilt strength)")
plt.ylabel("Mean Δp (pp) at β=0.5")
plt.title("Sensitivity over α (β=0.5)")
plt.legend()
plt.tight_layout()
fig4.savefig(outdir / "sensitivity_alpha_beta05.png", dpi=200)

print("Saved figures to:", outdir.resolve())
