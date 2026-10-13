"""figures and statistics for the report .tex

python src/plot.py [runs_dir]      (default: v2/runs)

Writes learning_curve, ablation, bias_vs_return (.pdf/.png), results_table.tex and stats.txt / stats.tex
to v2/figures/ and prints the statistical tests
"""
import os
import sys
import glob
import itertools
import math
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))   # v2/
RUNS_DIR = sys.argv[1] if len(sys.argv) > 1 else os.path.join(ROOT, "runs")
FIG_DIR = os.path.join(ROOT, "figures")
FINAL_FRAC = 0.2 # summary statistics use the last 20% of training
N_GRID = 300 # points of the common step grid
SMOOTH_WIN = 10 # moving average window over training episodes applied to each run before averaging
SMOOTH_WIN_EVAL = 3 # same for evaluations (one point every 10k steps)
RNG = np.random.default_rng(0)

# one fixed colour per configuration, used in every figure
CONFIGS = {
    "DDPG":            "#1f77b4",
    "DDPG+LN":         "#17becf",
    "TD3":             "#d62728",
    "TD3+LN":          "#ff7f0e",
    "TD3+LN (3 HL)":   "#2ca02c",
    "TD3+LN (UTD 2)":  "#9467bd",
}

plt.rcParams.update({"font.size": 8, "axes.titlesize": 9, "axes.labelsize": 8,
                     "legend.fontsize": 7, "xtick.labelsize": 7, "ytick.labelsize": 7})


def get_config_name(folder_name):
    # LunarLanderContinuous-v3__td3__rl_exp__HL2__LNTrue__UTD1__1__1727783307
    parts = folder_name.split('__')
    if len(parts) < 7:
        return None
    name = parts[1].upper()
    if parts[4] == "LNTrue":
        name += "+LN"
    if parts[3] != "HL2":
        name += f" ({parts[3][2:]} HL)"
    if parts[5] != "UTD1":
        name += f" (UTD {parts[5][3:]})"
    return name


def load_runs():
    runs = {}
    for run in sorted(glob.glob(os.path.join(RUNS_DIR, "*"))):
        cfg = get_config_name(os.path.basename(run))
        if cfg is None:
            continue
        ea = EventAccumulator(run, size_guidance={"scalars": 0})
        ea.Reload()
        data = {}
        for tag in ea.Tags()["scalars"]:
            ev = ea.Scalars(tag)
            data[tag] = (np.array([e.step for e in ev]), np.array([e.value for e in ev]))
        runs.setdefault(cfg, []).append(data)
    return runs


def pick(data, *tags):
    """first available tag (evaluation metrics exist only for runs made with the current main.py)"""
    for t in tags:
        if t in data and len(data[t][0]) > 1:
            return t, data[t]
    return None, None


def moving_average(x, w):
    if len(x) < w:
        return x
    c = np.cumsum(np.insert(x, 0, 0.0))
    head = c[1:w] / np.arange(1, w) # growing window for the first points
    return np.concatenate([head, (c[w:] - c[:-w]) / w])


def bootstrap_ci(x, n_boot=5000, alpha=0.05):
    """percentile bootstrap confidence interval of the mean along axis 0"""
    x = np.asarray(x)
    if len(x) < 2:
        return x.mean(0), x.mean(0)
    idx = RNG.integers(0, len(x), size=(n_boot, len(x)))
    means = x[idx].mean(1)
    return np.percentile(means, 100 * alpha / 2, axis=0), np.percentile(means, 100 * (1 - alpha / 2), axis=0)


def welch_test(a, b):
    """two sided t-test (p-value by numerical integration of the student law pdf)"""
    a, b = np.asarray(a, float), np.asarray(b, float)
    a, b = a[np.isfinite(a)], b[np.isfinite(b)]
    if len(a) < 2 or len(b) < 2:
        return np.nan, np.nan
    va, vb = a.var(ddof=1) / len(a), b.var(ddof=1) / len(b)
    if va + vb == 0:
        return np.nan, np.nan
    t = (a.mean() - b.mean()) / math.sqrt(va + vb)
    df = (va + vb) ** 2 / (va ** 2 / (len(a) - 1) + vb ** 2 / (len(b) - 1))
    xs = np.linspace(0, abs(t), 20001)
    logc = math.lgamma((df + 1) / 2) - math.lgamma(df / 2) - 0.5 * math.log(df * math.pi)
    pdf = np.exp(logc) * (1 + xs ** 2 / df) ** (-(df + 1) / 2)
    return t, max(0.0, 1 - 2 * np.trapezoid(pdf, xs))


def curves(runs_data, tags):
    """smoothing each run and interpolate on a common grid; returns grid, (n_runs, N_GRID) array, tag used"""
    picked = [pick(d, *tags) for d in runs_data]
    picked = [(t, sv) for t, sv in picked if t is not None]
    if not picked:
        return None, None, None
    max_step = min(sv[0].max() for _, sv in picked)
    grid = np.linspace(0, max_step, N_GRID)
    ys = []
    for _, (s, v) in picked:
        o = np.argsort(s)
        w = SMOOTH_WIN_EVAL if picked[0][0].startswith("eval/") else SMOOTH_WIN
        ys.append(np.interp(grid, s[o], moving_average(v[o], w)))
    return grid, np.array(ys), picked[0][0]


def plot_panel(ax, runs, cfgs, tags, ylabel, title):
    used = None
    for cfg in cfgs:
        if cfg not in runs:
            continue
        grid, ys, used = curves(runs[cfg], tags)
        if grid is None:
            continue
        lo, hi = bootstrap_ci(ys)
        ax.plot(grid / 1e3, ys.mean(0), color=CONFIGS[cfg], lw=1.3, label=f"{cfg} (n={len(ys)})")
        ax.fill_between(grid / 1e3, lo, hi, color=CONFIGS[cfg], alpha=0.2, lw=0)
    ax.set_title(title)
    ax.set_xlabel("Environment steps (×1000)")
    ax.set_ylabel(ylabel)
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best", frameon=False)
    return used


def two_panel_figure(runs, cfgs, filename, ret_title, bias_title, compact=False):
    """return + bias on training episodes + bias on evaluation episodes
    compact=True: only return and evaluation bias (smaller figure, used for the ablations)."""
    has_eval = any("eval/overestimation_bias" in d for c in cfgs if c in runs for d in runs[c])
    panels = [("ret", ["eval/episodic_return", "charts/episodic_return"], "Return", ret_title)]
    if not (compact and has_eval):
        panels.append(("bias", ["charts/overestimation_bias"], r"Bias  $\overline{Q(s,a)-G}$",
                       bias_title + (" (training)" if has_eval else "")))
    if has_eval:
        panels.append(("bias", ["eval/overestimation_bias"], r"Bias  $\overline{Q(s,a)-G}$", bias_title + " (eval.)"))
    size = (4.6, 2.3) if compact else ((7.2, 2.5) if len(panels) == 3 else (7, 2.6))
    fig, axs = plt.subplots(1, len(panels), figsize=size)
    used = []
    for i, (kind, tags, ylabel, title) in enumerate(panels):
        used.append(plot_panel(axs[i], runs, cfgs, tags, ylabel, title))
        if kind == "bias":
            axs[i].axhline(0, color="k", lw=0.6, ls="--")
        if i > 0:
            axs[i].get_legend().remove()
    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, f"{filename}.pdf"))
    fig.savefig(os.path.join(FIG_DIR, f"{filename}.png"), dpi=200)
    plt.close(fig)
    print(f"Saved {filename}  (metrics: {used})")


def run_summary(data, max_step):
    """mean of each metric over the last FINAL_FRAC of training for one run"""
    out = {}
    for key, tags in {"ret": ["eval/episodic_return", "charts/episodic_return"],
                      "train_ret": ["charts/episodic_return"],
                      "bias": ["charts/overestimation_bias"],
                      "eval_bias": ["eval/overestimation_bias"]}.items():
        t, sv = pick(data, *tags)
        if t is None:
            out[key] = np.nan
            continue
        s, v = sv
        m = (s >= (1 - FINAL_FRAC) * max_step) & np.isfinite(v)
        out[key] = v[m].mean() if m.any() else np.nan
    return out


def fmt(x, lo=None, hi=None):
    if np.isnan(x):
        return "--"
    if lo is None or np.isclose(lo, hi):
        return f"{x:.0f}"
    return f"{x:.0f} [{lo:.0f}, {hi:.0f}]"


def fmt_p(p):
    return "--" if np.isnan(p) else (r"$<$0.001" if p < 1e-3 else f"{p:.3f}")


def summaries(runs):
    max_step = max(max(sv[0].max() for sv in d.values()) for rs in runs.values() for d in rs)
    return {cfg: [run_summary(d, max_step) for d in rs] for cfg, rs in runs.items()}


def results_table(summ):
    lines = [r"\begin{tabular}{@{}lcccc@{}}", r"\toprule",
             r"Configuration & $n$ & Final return & Final bias (train) & Final bias (eval) \\ \midrule"]
    for cfg in CONFIGS:
        if cfg not in summ:
            continue
        cells = []
        for k in ["ret", "bias", "eval_bias"]:
            x = np.array([r[k] for r in summ[cfg]])
            x = x[np.isfinite(x)]
            if len(x) == 0:
                cells.append("--")
                continue
            lo, hi = bootstrap_ci(x)
            cells.append(fmt(x.mean(), lo, hi))
        lines.append(f"{cfg} & {len(summ[cfg])} & " + " & ".join(cells) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}"]

    comparisons = [("DDPG", "TD3"), ("DDPG", "DDPG+LN"), ("TD3", "TD3+LN"),
                   ("TD3+LN", "TD3+LN (3 HL)"), ("TD3+LN", "TD3+LN (UTD 2)")]
    lines += [r"\\[4pt]", r"\begin{tabular}{@{}lcccc@{}}", r"\toprule",
              r"Comparison (A vs B) & $\Delta$ return (B$-$A) & $p$ & $\Delta$ bias (B$-$A) & $p$ \\ \midrule"]
    print("\nWelch t-tests (two-sided), last %d%% of training:" % (100 * FINAL_FRAC))
    for a, b in comparisons:
        if a not in summ or b not in summ:
            continue
        row = [f"{a} vs {b}"]
        for k in ["ret", "bias"]:
            xa = np.array([r[k] for r in summ[a]]); xb = np.array([r[k] for r in summ[b]])
            t, p = welch_test(xb, xa)
            diff = np.nanmean(xb) - np.nanmean(xa)
            row += [f"{diff:+.0f}", fmt_p(p)]
            print(f"  {a:>8} vs {b:<15} {k:5s}: diff={diff:+7.1f}  t={t:6.2f}  p={p:.4f}")
        lines.append(" & ".join(row) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    with open(os.path.join(FIG_DIR, "results_table.tex"), "w") as f:
        f.write("\n".join(lines) + "\n")
    print("Saved results_table.tex")


def spearman(x, y, n_perm=20000):
    ok = np.isfinite(x) & np.isfinite(y)
    x, y = x[ok], y[ok]
    rx, ry = np.argsort(np.argsort(x)), np.argsort(np.argsort(y))
    rho = np.corrcoef(rx, ry)[0, 1]
    perm = np.array([np.corrcoef(rx, RNG.permutation(ry))[0, 1] for _ in range(n_perm)])
    return rho, (np.sum(np.abs(perm) >= abs(rho)) + 1) / (n_perm + 1)


def bias_vs_return(summ):
    fig, ax = plt.subplots(figsize=(2.6, 2.3))
    xs, ys = [], []
    for cfg in [c for c in CONFIGS if c in summ]:
        rs = summ[cfg]
        b = [r["bias"] for r in rs]; R = [r["ret"] for r in rs]
        ax.scatter(b, R, color=CONFIGS[cfg], label=cfg, s=18, edgecolor="k", lw=0.4)
        xs += b; ys += R
    ax.axvline(0, color="k", lw=0.6, ls="--")
    ax.set_xlabel(r"Final bias  $\overline{Q(s,a)-G}$")
    ax.set_ylabel("Final return")
    ax.grid(True, alpha=0.3)
    ax.legend(frameon=False, fontsize=6)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, "bias_vs_return.pdf"))
    fig.savefig(os.path.join(FIG_DIR, "bias_vs_return.png"), dpi=200)
    plt.close(fig)
    rho, p = spearman(np.array(xs), np.array(ys))
    print(f"\nSpearman(final bias, final return) over {int(np.sum(np.isfinite(xs) & np.isfinite(ys)))} runs: rho={rho:.2f}, permutation p={p:.4f}")
    print("Saved bias_vs_return")


#statistics used in the report text
TARGET_RETURN = 200 # landing score used to measure how fast each run learns


def group_spearman(summ, cfgs):
    """rank correlation between final bias and final return over all runs of the given configurations"""
    b = np.array([r["bias"] for c in cfgs if c in summ for r in summ[c]])
    R = np.array([r["ret"] for c in cfgs if c in summ for r in summ[c]])
    rho, p = spearman(b, R)
    return rho, p, int(np.sum(np.isfinite(b) & np.isfinite(R)))


def steps_to_target(runs, cfg):
    """for each run: first evaluation step in thousands at which the smoothed evaluation return
    reaches TARGET_RETURN (or nan if never reached)"""
    out = []
    for d in runs.get(cfg, []):
        if "eval/episodic_return" not in d:
            out.append(np.nan)
            continue
        s, v = d["eval/episodic_return"]
        o = np.argsort(s)
        s, v = s[o], moving_average(v[o], SMOOTH_WIN_EVAL)
        hit = np.where(v >= TARGET_RETURN)[0]
        out.append(s[hit[0]] / 1000 if len(hit) else np.nan)
    return np.array(out)


def fmt_p_eq(p):
    """p-value with its relation sign, for use in math mode: 'p=0.010' or 'p<0.001'."""
    return "p=\\text{--}" if np.isnan(p) else ("p<0.001" if p < 1e-3 else f"p={p:.3f}")


def report_stats(runs, summ):
    """computes the statistics quoted in the text of the report and saves them in
    figures/stats.txt and figures/stats.tex (as LaTeX macros, can be imported with \\input by the report)"""
    groups = {"All": list(CONFIGS),
              "DDPG": ["DDPG", "DDPG+LN"],
              "TDthree": ["TD3", "TD3+LN", "TD3+LN (3 HL)", "TD3+LN (UTD 2)"]}
    corr = {name: group_spearman(summ, cfgs) for name, cfgs in groups.items()}

    steps = {cfg: steps_to_target(runs, cfg) for cfg in CONFIGS}
    step_tests = {"LN": ("TD3", "TD3+LN"), "HLthree": ("TD3+LN", "TD3+LN (3 HL)"), "UTDtwo": ("TD3+LN", "TD3+LN (UTD 2)")}
    step_p = {name: welch_test(steps[a], steps[b])[1] for name, (a, b) in step_tests.items()}
    macro_cfg = {"TD3": "TDthree", "TD3+LN": "TDthreeLN", "TD3+LN (3 HL)": "HLthree", "TD3+LN (UTD 2)": "UTDtwo",
                 "DDPG": "DDPG", "DDPG+LN": "DDPGLN"}

    txt = ["Statistics quoted in the report (generated by src/plot.py)", ""]
    tex = ["% Generated by src/plot.py -- do not edit by hand. Statistics quoted in the text of the report.",
           "% p-value macros include the relation sign and are used in math mode, e.g. $\\rho=\\rhoAll$, $\\pRhoAll$."]
    txt.append("Rank correlation between final bias and final return (permutation p-value):")
    for name, (rho, p, n) in corr.items():
        txt.append(f"  {name:8s} n={n:2d}  rho={rho:+.2f}  p={p:.4f}   ({', '.join(groups[name])})")
        tex.append(f"\\newcommand{{\\rho{name}}}{{{rho:.2f}}}")
        tex.append(f"\\newcommand{{\\pRho{name}}}{{{fmt_p_eq(p)}}}")
        tex.append(f"\\newcommand{{\\nRho{name}}}{{{n}}}")
    txt += ["", f"First evaluation step (k) at which the smoothed evaluation return reaches {TARGET_RETURN}:"]
    for cfg in CONFIGS:
        if cfg not in runs:
            continue
        st = steps[cfg]
        med = np.nanmedian(st) if np.isfinite(st).any() else np.nan
        reached = int(np.isfinite(st).sum())
        med_txt = "--" if np.isnan(med) else f"{med:.0f}k"
        txt.append(f"  {cfg:16s} per seed={[None if np.isnan(x) else int(x) for x in st]}  median={med_txt}  reached by {reached}/{len(st)}")
        tex.append(f"\\newcommand{{\\steps{macro_cfg[cfg]}}}{{{'--' if np.isnan(med) else f'{med:.0f}k'}}}")
    txt += ["", "Welch t-tests on these steps:"]
    for name, (a, b) in step_tests.items():
        txt.append(f"  {a} vs {b}: p={step_p[name]:.4f}")
        tex.append(f"\\newcommand{{\\pSteps{name}}}{{{fmt_p_eq(step_p[name])}}}")

    with open(os.path.join(FIG_DIR, "stats.txt"), "w") as f:
        f.write("\n".join(txt) + "\n")
    with open(os.path.join(FIG_DIR, "stats.tex"), "w") as f:
        f.write("\n".join(tex) + "\n")
    print("\n" + "\n".join(txt))
    print("Saved stats.txt and stats.tex")


if __name__ == "__main__":
    os.makedirs(FIG_DIR, exist_ok=True)
    if not os.path.exists(RUNS_DIR):
        sys.exit(f"No {RUNS_DIR} folder found. Run experiments first.")
    runs = load_runs()
    two_panel_figure(runs, ["DDPG", "DDPG+LN", "TD3", "TD3+LN"], "learning_curve",
                     "Return", "Overestimation bias")
    two_panel_figure(runs, ["TD3+LN", "TD3+LN (3 HL)", "TD3+LN (UTD 2)"], "ablation",
                     "Return", "Bias", compact=True)
    summ = summaries(runs)
    results_table(summ)
    bias_vs_return(summ)
    report_stats(runs, summ)
    for cfg, rs in summ.items():
        print(f"{cfg:16s} " + "  ".join(f"{k}={np.nanmean([r[k] for r in rs]):7.1f}" for k in rs[0]))
