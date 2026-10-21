"""Audit aggregate XGBoost results and generate publication review plots.

Uses four *aggregate* CSV files only. It cannot read a prepared dataset,
row-level predictions, stored joblib models or the March-April holdout.
Existing outputs are never replaced; a written manifest permits hash checks.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd

BUDGETS = (0.001, 0.005, 0.01)
MONTHS = {"A_january": (186_097, 281), "B_february": (167_990, 202)}
CANDIDATES = ("depth2_120", "depth2_240", "depth3_240")
SERIES = {
    "A_january": ("logistic", "xgboost"),
    "B_february": tuple(f"{f}_{m}" for f in ("logistic", "xgboost")
                        for m in ("uncalibrated", "sigmoid", "isotonic")),
}
FEATURES = (
    "amount", "oldbalanceOrg", "hour", "day_of_week", "day_of_month", "month",
    "type", "category", "metadata.payment_method", "metadata.device_info.os",
    "metadata.device_info.type", "metadata.location.city", "metadata.location.state",
)
FILES = ("december_candidates.csv", "metrics.csv",
         "february_shared_cohort_bins.csv", "stage_b_feature_importance.csv")
NAVY, ORANGE, TEAL, PURPLE, DARK = "#183750", "#C45B14", "#28738A", "#7F4A84", "#252525"


def sha256(path: Path) -> str:
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def require(condition: bool, message: str):
    if not condition:
        raise ValueError(message)


def load_and_validate(folder: Path) -> dict[str, pd.DataFrame]:
    tables = {name: pd.read_csv(folder / name) for name in FILES}
    c, m, b, g = (tables[name] for name in FILES)
    require(list(c.columns) == ["candidate", "max_depth", "n_estimators", "average_precision", "log_loss"],
            "December candidate columns changed")
    require(tuple(c.candidate) == CANDIDATES, "Candidate identities or order changed")
    require(list(c.max_depth) == [2, 2, 3] and list(c.n_estimators) == [120, 240, 240],
            "Locked December candidate configurations changed")
    require(np.isfinite(c[["average_precision", "log_loss"]].to_numpy(dtype=float)).all()
            and c.average_precision.between(0, 1).all() and c.log_loss.ge(0).all(),
            "Invalid December metrics")
    best_ap = float(c.average_precision.max())
    near = c.loc[c.average_precision >= best_ap - 1e-6]
    best_ll = float(near.log_loss.min())
    chosen = near.loc[near.log_loss <= best_ll + 1e-10, "candidate"].iloc[0]
    require(chosen == "depth3_240", "December choice changed: open a newly versioned protocol")

    expected_columns = {"stage", "series", "average_precision", "log_loss", "brier_score",
                        "review_fraction", "selected", "actual_review_fraction", "true_positives",
                        "false_positives", "false_negatives", "true_negatives", "precision", "recall",
                        "cutoff_score", "cutoff_tie_count", "selected_from_cutoff_tie", "tie_crosses_cutoff"}
    require(set(m.columns) == expected_columns and len(m) == 24, "Metric table columns or size changed")
    require(set(m.stage) == set(MONTHS), "Unexpected stage: March-April must stay out of scope")
    for stage, (n, p) in MONTHS.items():
        part = m.loc[m.stage.eq(stage)]
        require(set(part.series) == set(SERIES[stage]), f"Unexpected model series in {stage}")
        for series in SERIES[stage]:
            s = part.loc[part.series.eq(series)].sort_values("review_fraction")
            require(len(s) == 3 and np.allclose(s.review_fraction, BUDGETS, rtol=0, atol=1e-12),
                    f"Review budgets invalid for {series}")
            require(s[["average_precision", "log_loss", "brier_score"]].nunique().eq(1).all(),
                    f"Repeated model metrics disagree for {series}")
            require(np.isfinite(s.select_dtypes(include="number")).all().all(),
                    f"Nonfinite metrics for {series}")
            for row in s.itertuples():
                k = math.ceil(n * row.review_fraction)
                require(row.selected == k and row.true_positives + row.false_positives == k,
                        f"Invalid selected count for {series}")
                require(row.true_positives + row.false_negatives == p
                        and row.false_positives + row.true_negatives == n - p,
                        f"Confusion totals do not reconcile for {series}")
                require(np.isclose(row.precision, row.true_positives / k, atol=1e-12)
                        and np.isclose(row.recall, row.true_positives / p, atol=1e-12)
                        and np.isclose(row.actual_review_fraction, k / n, atol=1e-12),
                        f"Precision/recall/budget mismatch for {series}")
                require(0 <= row.cutoff_score <= 1 and row.cutoff_tie_count >= row.selected_from_cutoff_tie,
                        f"Invalid score/tie boundary for {series}")
                require(0 <= row.average_precision <= 1 and row.log_loss >= 0 and row.brier_score >= 0,
                        f"Invalid model scores for {series}")

    require(set(b.columns) == {"reference_decile", "series", "rows", "positives",
                               "observed_rate", "mean_probability"} and len(b) == 60,
            "Shared cohort table columns or size changed")
    require(set(b.series) == set(SERIES["B_february"]), "Shared cohort series changed")
    require(set(b.reference_decile) == set(range(1, 11)), "Exactly ten matched deciles required")
    for series in SERIES["B_february"]:
        s = b.loc[b.series.eq(series)]
        require(len(s) == 10 and not s.reference_decile.duplicated().any()
                and int(s.rows.sum()) == 167_990 and int(s.positives.sum()) == 202,
                f"Matched cohort sums disagree for {series}")
    for decile, s in b.groupby("reference_decile"):
        require(s.rows.nunique() == s.positives.nunique() == 1 and s.rows.iloc[0] == 16_799,
                f"Cohort {decile} no longer contains identical rows")
    require(np.isfinite(b[["observed_rate", "mean_probability"]]).all().all()
            and b.mean_probability.between(0, 1).all()
            and np.allclose(b.observed_rate, b.positives / b.rows, atol=1e-12),
            "Invalid observed/predicted bin rates")
    require(list(g.columns) == ["feature", "normalized_gain"] and len(g) == 13
            and set(g.feature) == set(FEATURES) and not g.feature.duplicated().any(),
            "Original feature-group count changed")
    require(g.normalized_gain.ge(0).all() and np.isfinite(g.normalized_gain).all()
            and np.isclose(g.normalized_gain.sum(), 1.0, atol=1e-6),
            "Importance weights do not form a normalized distribution")
    return tables


def setup():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.size": 10, "axes.titlesize": 12, "axes.labelsize": 10,
                         "axes.spines.right": False, "axes.spines.top": False,
                         "savefig.facecolor": "white"})
    return plt


def save(fig, path):
    fig.savefig(path, dpi=170, bbox_inches="tight")
    import matplotlib.pyplot as plt
    plt.close(fig)


def plot_december(c, output, plt):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.6))
    labels = ["Depth 2\n120 trees", "Depth 2\n240 trees", "Depth 3\n240 trees"]
    for ax, col, heading, fmt in [
        (axes[0], "average_precision", "December average precision ↑", ".4f"),
        (axes[1], "log_loss", "December log loss ↓", ".5f")]:
        vals = c[col].to_numpy(float)
        bars = ax.bar(range(3), vals, color=["#698497", "#698497", ORANGE],
                      edgecolor=DARK, linewidth=.7, width=.67)
        for bar, v in zip(bars, vals):
            ax.annotate(f"{v:{fmt}}", (bar.get_x()+bar.get_width()/2, v),
                        xytext=(0, 5), textcoords="offset points", ha="center", fontsize=10)
        ax.set_xticks(range(3), labels)
        ax.set_ylim(0, max(vals)*1.18)
        ax.grid(axis="y", alpha=.18)
        ax.set_title(heading)
    fig.suptitle("December configuration selection · October–November fitting; AP chosen in advance")
    fig.text(.5, .005, "Orange = selected configuration; differences are historical observations, not significance tests",
             ha="center", fontsize=9, color="#444444")
    fig.tight_layout(rect=(0, .04, 1, .93))
    save(fig, output / "xgb_december_candidate_comparison.png")


def plot_budgets(m, output, plt):
    s=m.loc[m.stage.eq("B_february") & m.series.isin(["logistic_uncalibrated","xgboost_uncalibrated"])].copy()
    budgets=np.array(BUDGETS)
    lr=s.loc[s.series.eq("logistic_uncalibrated")].sort_values("review_fraction")
    xg=s.loc[s.series.eq("xgboost_uncalibrated")].sort_values("review_fraction")
    a,b=lr.true_positives.to_numpy(int),xg.true_positives.to_numpy(int)
    fig, ax=plt.subplots(1,2,figsize=(11.8,4.7),gridspec_kw={"width_ratios":[1.55,1]})
    x=np.arange(3)
    bars1=ax[0].bar(x-.19,a,.38,color=NAVY,edgecolor=DARK,label="Logistic")
    bars2=ax[0].bar(x+.19,b,.38,color=ORANGE,hatch="//",edgecolor=DARK,label="XGBoost")
    for bar in list(bars1)+list(bars2):
        ax[0].text(bar.get_x()+bar.get_width()/2,bar.get_height()+2,str(int(bar.get_height())),ha="center")
    ax[0].set_ylim(0,225)
    ax[0].set_xticks(x,["0.1%\n168 reviews","0.5%\n840 reviews","1.0%\n1,680 reviews"])
    ax[0].set_ylabel("Positive-labelled transactions caught (of 202)")
    ax[0].grid(axis="y",alpha=.2)
    ax[0].legend(frameon=False)
    diff=b-a
    ax[1].bar(x,diff,color=ORANGE,edgecolor=DARK,hatch="//",width=.55)
    ax[1].set_xticks(x,["0.1%","0.5%","1.0%"])
    ax[1].set_ylim(0,max(diff)+1.4)
    ax[1].set_ylabel("Additional positives caught by XGBoost")
    ax[1].grid(axis="y",alpha=.2)
    for i,v in enumerate(diff):ax[1].text(i,float(v)+.1,f"+{v}",ha="center",weight="bold")
    fig.suptitle("February retrospective review budgets · uncalibrated scores · same 202 positives")
    fig.tight_layout(rect=(0,0,1,.93))
    save(fig,output/"xgb_february_review_budget_differences.png")


def plot_losses(m,output,plt):
    s=m.loc[m.stage.eq("B_february") & np.isclose(m.review_fraction, .01)].set_index("series")
    methods=["uncalibrated","sigmoid","isotonic"]
    fig, ax=plt.subplots(1,2,figsize=(12.5,4.8))
    for panel,col,factor,title in [(ax[0],"log_loss",1000,"Log loss × 1,000 (lower better)"),
                                    (ax[1],"brier_score",10000,"Brier score × 10,000 (lower better)")]:
        x=np.arange(3)
        for offset,fam,color in [(-.19,"logistic",NAVY),(.19,"xgboost",ORANGE)]:
            vals=np.array([s.loc[f"{fam}_{method}",col] for method in methods])*factor
            bars=panel.bar(x+offset,vals,.37,color=color,label=fam.title(),edgecolor=DARK,
                           hatch="//" if fam=="xgboost" else "")
            for bar,val in zip(bars,vals):
                panel.text(bar.get_x()+bar.get_width()/2,bar.get_height()+max(vals)*.025,
                           f"{val:.3f}",ha="center",fontsize=8,rotation=0)
        maxv=max(float(s.loc[f"{fam}_{method}",col])*factor for fam in ("logistic","xgboost") for method in methods)
        panel.set_ylim(0,maxv*1.20)
        panel.set_xticks(x,["Raw","Sigmoid","Isotonic"])
        panel.set_title(title)
        panel.grid(axis="y",alpha=.2)
    ax[0].legend(frameon=False,ncol=2)
    fig.suptitle("February historical probability diagnostics · calibration maps fitted on January")
    fig.tight_layout(rect=(0,0,1,.92))
    save(fig,output/"xgb_february_loss_and_brier_comparison.png")


def plot_shared_bins(b,output,plt):
    names=[("logistic_uncalibrated",NAVY,"-","o"),
           ("logistic_sigmoid",TEAL,"--","o"),
           ("logistic_isotonic","#398461",":","o"),
           ("xgboost_uncalibrated",ORANGE,"-","s"),
           ("xgboost_sigmoid",PURPLE,"--","s"),
           ("xgboost_isotonic","#9B782A",":","s")]
    fig,ax=plt.subplots(1,2,figsize=(15,5.5),gridspec_kw={"width_ratios":[1.7,1]})
    for name,color,style,marker in names:
        s=b.loc[b.series.eq(name)].sort_values("reference_decile")
        ax[0].plot(s.reference_decile.iloc[:9],s.mean_probability.iloc[:9]*1e4,
                   label=name.replace("_"," / "),color=color,linestyle=style,marker=marker,
                   linewidth=1.4,markersize=4)
    obs=b.loc[b.series.eq("logistic_uncalibrated")].sort_values("reference_decile")
    ax[0].plot(obs.reference_decile.iloc[:9],obs.observed_rate.iloc[:9]*1e4,color=DARK,
               linewidth=2.5,linestyle="-",marker="D",label="Observed labels")
    ax[0].set_xticks(range(1,10))
    ax[0].set_xlabel("Fixed logistic reference decile (1–9)")
    ax[0].set_ylabel("Predicted or observed positive rate (basis points)")
    ax[0].set_title("Lower nine deciles · 1 basis point = 0.01%")
    ax[0].grid(alpha=.17)
    ax[0].legend(ncol=2,fontsize=7.8,loc="upper left")
    # Separate the high-risk group to avoid compressing the lower-risk rates.
    top=b.loc[b.reference_decile.eq(10)].set_index("series")
    labels=["LR raw","LR sig","LR iso","XGB raw","XGB sig","XGB iso"]
    x=np.arange(6)
    cols=[NAVY,TEAL,"#398461",ORANGE,PURPLE,"#9B782A"]
    vals=[100*top.loc[name,"mean_probability"] for name,_,_,_ in names]
    ax[1].bar(x,vals,color=cols,edgecolor=DARK,width=.65)
    ax[1].axhline(float(obs.observed_rate.iloc[-1])*100,color=DARK,
                  linestyle="--",linewidth=2,label=f"Observed {obs.observed_rate.iloc[-1]*100:.3f}% (199/16,799)")
    ax[1].set_xticks(x,labels,rotation=35,ha="right")
    ax[1].set_ylim(0,max(vals)*1.27)
    ax[1].set_ylabel("Predicted positive rate (%)")
    ax[1].set_title("Highest decile · 16,799 transactions")
    ax[1].grid(axis="y",alpha=.17)
    ax[1].legend(fontsize=8)
    fig.suptitle("February reliability: identical logistic-defined transaction cohorts for every series")
    fig.text(.5,.002,"Observed positives by decile: 0, 1, 0, 0, 0, 0, 0, 0, 2, 199. This is descriptive grouping, not a per-row calibration test.",
             fontsize=9,ha="center",color="#424242")
    fig.tight_layout(rect=(0,.035,1,.92))
    save(fig,output/"xgb_february_shared_cohort_split_scale.png")


def plot_gain(g,output,plt):
    g=g.sort_values("normalized_gain",ascending=False)
    fig,ax=plt.subplots(figsize=(10.7,6.4))
    y=np.arange(len(g))
    cols=[ORANGE if name in ("category","type") else TEAL for name in g.feature]
    ax.barh(y,g.normalized_gain*100,color=cols,edgecolor=DARK,height=.74)
    ax.set_yticks(y,g.feature)
    ax.invert_yaxis()
    ax.set_xlim(0,max(g.normalized_gain*100)*1.20)
    ax.set_xlabel("Summed normalized per-encoded-feature average gain (%)")
    ax.set_title("Stage B: source-feature group gain diagnostic (not causal or total gain)")
    ax.grid(axis="x",alpha=.18)
    for i,v in enumerate(g.normalized_gain):
        ax.text(100*v+.55,i,f"{100*v:.2f}%",va="center",fontsize=8.7)
    fig.tight_layout()
    save(fig,output/"xgb_grouped_gain_with_metric_caveat.png")


def produce(input_dir: Path, output_dir: Path):
    if output_dir.exists():
        raise FileExistsError(f"Refusing to overwrite versioned output: {output_dir}")
    tables=load_and_validate(input_dir)
    output_dir.parent.mkdir(parents=True,exist_ok=True)
    output_dir.mkdir(exist_ok=False)
    plt=setup()
    c,m,b,g=(tables[f] for f in FILES)
    plot_december(c,output_dir,plt)
    plot_budgets(m,output_dir,plt)
    plot_losses(m,output_dir,plt)
    plot_shared_bins(b,output_dir,plt)
    plot_gain(g,output_dir,plt)
    manifest={"source_tables_sha256": {name:sha256(input_dir/name) for name in FILES},
              "figures_sha256": {p.name:sha256(p) for p in sorted(output_dir.glob("*.png"))},
              "source_scope": "Aggregates only: December, January, February; no test/holdout rows loaded",
              "selected_december_configuration": "depth3_240",
              "complete":True}
    (output_dir/"audit_manifest.json").write_text(json.dumps(manifest,indent=2)+"\n",encoding="utf-8")
    print(f"PASS: four aggregate CSVs reconciled, five figures created in {output_dir}")


def check_only(input_dir:Path, output_dir:Path):
    load_and_validate(input_dir)
    manifest=json.loads((output_dir/"audit_manifest.json").read_text(encoding="utf-8"))
    require(manifest.get("complete") is True,"Incomplete output manifest")
    for name,digest in manifest["source_tables_sha256"].items():
        require(name in FILES and sha256(input_dir/name)==digest,f"Input changed: {name}")
    require(len(manifest["source_tables_sha256"])==4,"Missing input hash")
    require(len(manifest["figures_sha256"])==5,"Expected five figure hashes")
    for name,digest in manifest["figures_sha256"].items():
        require(name.endswith(".png") and sha256(output_dir/name)==digest,f"Figure changed: {name}")
    print("PASS: aggregate arithmetic, source hashes, and saved figure hashes verified; nothing rewritten")


def main():
    root=Path(__file__).resolve().parents[1]
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input",type=Path,default=root/"results/tables/xgboost_benchmark_v1")
    parser.add_argument("--output",type=Path,default=root/"models/xgboost_aggregate_review_v2")
    parser.add_argument("--check-only",action="store_true")
    args=parser.parse_args()
    if args.check_only:check_only(args.input,args.output)
    else:produce(args.input,args.output)

if __name__=="__main__":main()
