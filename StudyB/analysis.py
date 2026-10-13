import argparse
import glob
import json
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats


def load_results(results_dir):
    frames = []
    for path in sorted(glob.glob(os.path.join(results_dir, "*.json"))):
        with open(path) as f:
            meta = json.load(f)
        csv_path = os.path.join(results_dir, f"{meta['run_name']}.eval.csv")
        if not meta.get("finished") or not os.path.exists(csv_path):
            continue
        df = pd.read_csv(csv_path)
        for k in ["algo", "layernorm", "seed", "n_hidden", "utd", "policy_delay"]:
            df[k] = meta[k]
        frames.append(df)
    if not frames:
        raise SystemExit(f"No finished runs in {results_dir}")
    return pd.concat(frames, ignore_index=True)


def boot_ci(x, rng, n_boot=10_000, level=0.95):
    x = np.asarray(x, dtype=float)
    x = x[~np.isnan(x)]
    if len(x) < 2:
        return (np.nan, np.nan)
    means = rng.choice(x, size=(n_boot, len(x)), replace=True).mean(axis=1)
    a = (1 - level) / 2
    return tuple(np.quantile(means, [a, 1 - a]))


def boot_diff_ci(x, y, rng, n_boot=10_000, level=0.95):
    x, y = np.asarray(x, float), np.asarray(y, float)
    d = rng.choice(x, (n_boot, len(x))).mean(1) - rng.choice(y, (n_boot, len(y))).mean(1)
    a = (1 - level) / 2
    return tuple(np.quantile(d, [a, 1 - a]))


def holm(pvals):
    p = np.asarray(pvals, float)
    adj = np.empty_like(p)
    running = 0.0
    for rank, i in enumerate(np.argsort(p)):
        running = max(running, (len(p) - rank) * p[i])
        adj[i] = min(1.0, running)
    return adj


def welch(x, y):
    t = stats.ttest_ind(x, y, equal_var=False)
    return np.mean(x) - np.mean(y), t.statistic, t.df, t.pvalue


def spearman_perm(x, y, rng, n_perm=20_000):
    rho = stats.spearmanr(x, y).statistic
    rx, ry = stats.rankdata(x), stats.rankdata(y)
    perm = np.array([np.corrcoef(rx, rng.permutation(ry))[0, 1] for _ in range(n_perm)])
    return rho, (np.sum(np.abs(perm) >= abs(rho)) + 1) / (n_perm + 1)

FINAL_FROM = 240_000
ORDER = ["DDPG", "DDPG+LN", "TD3", "TD3+LN", "DDPG (d=2)", "DDPG+LN (d=2)"]
FAMILIES = {
    "main": [("DDPG", "TD3"), ("DDPG", "DDPG+LN"), ("TD3", "TD3+LN")],
    "delay": [("DDPG", "DDPG (d=2)"), ("DDPG+LN", "DDPG+LN (d=2)"), ("DDPG (d=2)", "DDPG+LN (d=2)")],
}
GROUPS = {"All": ["DDPG", "DDPG+LN", "TD3", "TD3+LN"], "DDPG": ["DDPG", "DDPG+LN"], "TDthree": ["TD3", "TD3+LN"]}
BLUE, ORANGE, VIOLET = "#2a78d6", "#eb6834", "#4a3aa7"
STYLE = {"DDPG": (BLUE, "-"), "DDPG+LN": (BLUE, "--"), "TD3": (ORANGE, "-"), "TD3+LN": (ORANGE, "--"),
         "DDPG (d=2)": (VIOLET, "-"), "DDPG+LN (d=2)": (VIOLET, "--")}
plt.rcParams.update({"font.size": 8, "axes.titlesize": 8.5, "axes.labelsize": 8, "legend.fontsize": 6.5,
                     "xtick.labelsize": 7, "ytick.labelsize": 7, "axes.spines.top": False,
                     "axes.spines.right": False, "axes.grid": True, "grid.color": "#d9d9d6", "grid.linewidth": 0.6,
                     "axes.edgecolor": "#52514e", "axes.labelcolor": "#0b0b0b", "xtick.color": "#52514e",
                     "ytick.color": "#52514e", "pdf.fonttype": 42, "savefig.bbox": "tight"})


def config_name(algo, ln, d):
    s = {"ddpg": "DDPG", "td3": "TD3"}[algo] + ("+LN" if ln else "")
    return s + (" (d=2)" if algo == "ddpg" and d == 2 else "")


def signed(v):
    return f"{v:+.0f}".replace("-", "$-$").replace("+", "$+$")


def with_ci(m, ci):
    return f"{m:.0f} [{ci[0]:.0f}, {ci[1]:.0f}]".replace("-", "$-$")


def p_fmt(p):
    return r"$<$0.001" if p < 1e-3 else f"{p:.3f}"


def load(results_dir):
    df = load_results(results_dir)
    df = df[(df.n_hidden == 2) & (df.utd == 1)].copy()
    df = df[~((df.algo == "td3") & (df.policy_delay != 2))]
    df["bias"] = np.where(df.algo == "td3", df.bias_min, df.bias_q1)
    df["cfg"] = [config_name(a, l, d) for a, l, d in zip(df.algo, df.layernorm, df.policy_delay)]
    return df


def run_tests(get, cfgs):
    tests = []
    for fam, pairs in FAMILIES.items():
        rows = []
        for a, b in pairs:
            if a in cfgs and b in cfgs:
                for k in ["return_mean", "bias"]:
                    x, y = get(b, k), get(a, k)
                    diff, t, dof, p = welch(x, y)
                    ci = boot_diff_ci(x, y, np.random.default_rng(0))
                    rows.append(dict(family=fam, a=a, b=b, metric=k, diff=diff, ci=ci, t=t, df=dof, p=p))
        for r, pa in zip(rows, holm([r["p"] for r in rows]) if rows else []):
            r["p_holm"] = pa
        tests += rows
    return tests


def write_table(path, summary, tests):
    tab = [r"\begin{tabular}{@{}lccc@{}}", r"\toprule",
           r"Configuration (study B) & $n$ & Final return & Final bias (eval) \\ \midrule"]
    tab += [f"{c} & {n} & {with_ci(rm, rci)} & {with_ci(bm, bci)} \\\\" for c, n, rm, rci, bm, bci in summary]
    tab += [r"\bottomrule", r"\end{tabular}", r"\\[4pt]", r"\begin{tabular}{@{}lcccc@{}}", r"\toprule",
            r"Comparison (X vs Y) & $\Delta$ return (Y$-$X) & $p_{\text{Holm}}$ & $\Delta$ bias, eval (Y$-$X) & $p_{\text{Holm}}$ \\ \midrule"]
    for fam, pairs in FAMILIES.items():
        for a, b in pairs:
            rr = [r for r in tests if r["family"] == fam and r["a"] == a and r["b"] == b]
            if len(rr) == 2:
                ret, bia = rr
                tab.append(f"{a} vs {b} & {signed(ret['diff'])} & {p_fmt(ret['p_holm'])} & "
                           f"{signed(bia['diff'])} & {p_fmt(bia['p_holm'])} \\\\")
        if fam == "main":
            tab.append(r"\midrule")
    tab += [r"\bottomrule", r"\end{tabular}"]
    with open(path, "w") as f:
        f.write("% generated by analysis.py (study B)\n" + "\n".join(tab) + "\n")


def plot(df, cfgs, rng, path):
    def curve(c, col):
        piv = df[df.cfg == c].pivot_table(index="step", columns="seed", values=col).dropna()
        cis = np.array([boot_ci(piv.loc[s].values, rng, n_boot=2000) for s in piv.index])
        return piv.index.values / 1e3, piv.mean(axis=1).values, cis

    fig, axes = plt.subplots(1, 3, figsize=(7.2, 2.5))
    panels = [(axes[0], GROUPS["All"], "return_mean", "Return", "(a) Return (eval.)"),
              (axes[1], GROUPS["All"], "bias", r"Bias  $\overline{Q(s,a)-G}$", "(b) Bias (eval.)"),
              (axes[2], ["DDPG", "DDPG+LN", "DDPG (d=2)", "DDPG+LN (d=2)"], "bias", r"Bias  $\overline{Q(s,a)-G}$",
               "(c) DDPG bias: delay $d$=1 vs 2")]
    for ax, cs, col, ylab, title in panels:
        for c in [c for c in cs if c in cfgs]:
            x, m, ci = curve(c, col)
            color, ls = STYLE[c]
            ax.plot(x, m, color=color, ls=ls, lw=1.3, label=c)
            ax.fill_between(x, ci[:, 0], ci[:, 1], color=color, alpha=0.15, lw=0)
        ax.set(title=title, xlabel="Environment steps (×1000)", ylabel=ylab)
        ax.grid(True, alpha=0.3)
        if col == "bias":
            ax.axhline(0, color="k", lw=0.6, ls="--")
    handles = {}
    for ax in axes:
        for h, l in zip(*ax.get_legend_handles_labels()):
            handles.setdefault(l, h)
    fig.legend(list(handles.values()), list(handles), loc="upper center", ncol=len(handles), frameon=False,
               bbox_to_anchor=(0.5, 1.01), fontsize=7)
    fig.tight_layout(rect=(0, 0, 1, 0.9))
    for ext in ["pdf", "png"]:
        fig.savefig(f"{path}.{ext}", dpi=200)
    plt.close(fig)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--results", default="results")
    p.add_argument("--out", default="report_b")
    args = p.parse_args()
    os.makedirs(args.out, exist_ok=True)
    rng = np.random.default_rng(0)

    df = load(args.results)
    fin = df[df.step >= FINAL_FROM].groupby(["cfg", "seed"])[["return_mean", "bias"]].mean().reset_index()
    cfgs = [c for c in ORDER if c in set(fin.cfg)]

    def get(c, k):
        return fin[fin.cfg == c][k].values

    summary = []
    for c in cfgs:
        r, b = get(c, "return_mean"), get(c, "bias")
        summary.append((c, len(r), r.mean(), boot_ci(r, rng), b.mean(), boot_ci(b, rng)))
    tests = run_tests(get, cfgs)

    txt = ["Per configuration (mean over seeds, 95% bootstrap CI), last 20% of training:"]
    txt += [f"  {c:15s} n={n:2d}  return {rm:7.1f} [{rci[0]:7.1f}, {rci[1]:7.1f}]"
            f"  bias {bm:7.1f} [{bci[0]:7.1f}, {bci[1]:7.1f}]" for c, n, rm, rci, bm, bci in summary]
    txt.append("\nWelch t-tests (Y minus X), Holm within each family, 95% bootstrap CI of the difference:")
    txt += [f"  [{r['family']}] {r['a']:13s} vs {r['b']:15s} {r['metric']:11s}: diff={r['diff']:+7.1f}"
            f" [{r['ci'][0]:+7.1f}, {r['ci'][1]:+7.1f}]  t={r['t']:6.2f} (df {r['df']:4.1f})"
            f"  p={r['p']:.4f}  p_Holm={r['p_holm']:.4f}" for r in tests]

    txt.append("\nSpearman(final bias, final return), permutation p:")
    macros = []
    for g, cs in GROUPS.items():
        sub = fin[fin.cfg.isin(cs)]
        rho, pv = spearman_perm(sub.bias.values, sub.return_mean.values, rng)
        txt.append(f"  {g:8s} n={len(sub):2d}  rho={rho:+.2f}  p={pv:.4f}")
        macros += [rf"\newcommand{{\BrhoS{g}}}{{{rho:.2f}}}",
                   rf"\newcommand{{\BpRho{g}}}{{{'p<0.001' if pv < 1e-3 else f'p={pv:.3f}'}}}",
                   rf"\newcommand{{\BnRho{g}}}{{{len(sub)}}}"]

    write_table(os.path.join(args.out, "table_b.tex"), summary, tests)
    with open(os.path.join(args.out, "macros_b.tex"), "w") as f:
        f.write("% generated by analysis.py (study B)\n" + "\n".join(macros) + "\n")
    with open(os.path.join(args.out, "stats_b.txt"), "w") as f:
        f.write("\n".join(txt) + "\n")
    print("\n".join(txt))
    plot(df, cfgs, rng, os.path.join(args.out, "study_b"))


if __name__ == "__main__":
    main()
