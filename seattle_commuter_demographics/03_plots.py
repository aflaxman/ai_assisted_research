"""Plots summarizing the analysis."""
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

OUT = Path(__file__).parent / "results"
OUT.mkdir(exist_ok=True)

plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False})


def fig1_topline_inflow() -> None:
    df = pd.read_csv(OUT / "03_seattle_workers_by_home.csv", index_col=0)
    order = ["Seattle", "King Co. (outside Seattle)",
             "WA (outside King Co.)", "Out of state"]
    counts = df.loc[order, "total"]
    fig, ax = plt.subplots(figsize=(7.5, 4))
    colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728"]
    ax.barh(range(len(counts)), counts, color=colors)
    for i, (lbl, v) in enumerate(zip(counts.index, counts.values)):
        ax.text(v + 5000, i, f"{v:,} ({v/counts.sum()*100:.1f}%)", va="center")
    ax.set_yticks(range(len(counts))); ax.set_yticklabels(counts.index)
    ax.invert_yaxis()
    ax.set_xlabel("Workers")
    ax.set_xlim(0, counts.max() * 1.25)
    ax.set_title(f"Where Seattle's {counts.sum():,} workers live (LODES 2023)")
    fig.tight_layout(); fig.savefig(OUT / "fig1_inflow_topline.png", dpi=140)


def fig2_industry_commute() -> None:
    home = pd.read_csv(OUT / "05_industry_home_origin.csv")
    bucket_order = ["Seattle", "King Co. (outside Seattle)",
                    "WA (outside King Co.)", "Out of state"]
    ind_order = ["Educational Services (CNS15 / NAICS 61)",
                 "Health Care & Social Assistance (CNS16 / NAICS 62)",
                 "Public Administration (CNS20 / NAICS 92)"]
    pv = home[home["home_origin"].isin(bucket_order)] \
        .pivot(index="industry", columns="home_origin", values="share_pct") \
        .reindex(ind_order)[bucket_order]
    short = {"Educational Services (CNS15 / NAICS 61)": "Education\n(48k workers)",
             "Health Care & Social Assistance (CNS16 / NAICS 62)": "Health Care\n(88k workers)",
             "Public Administration (CNS20 / NAICS 92)": "Government\n(21k workers)"}
    pv.index = [short[i] for i in pv.index]

    fig, ax = plt.subplots(figsize=(8, 4))
    left = np.zeros(len(pv))
    colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728"]
    for col, c in zip(bucket_order, colors):
        ax.barh(pv.index, pv[col], left=left, color=c, label=col)
        for i, (val, lo) in enumerate(zip(pv[col], left)):
            if val >= 4:
                ax.text(lo + val/2, i, f"{val:.0f}%", ha="center", va="center",
                        color="white", fontsize=9, fontweight="bold")
        left += pv[col].values
    ax.set_xlim(0, 100); ax.set_xlabel("Share of workers (%)")
    ax.set_title("Where Seattle workers live, by industry")
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, -0.25),
              ncols=4, frameon=False, fontsize=9)
    fig.tight_layout(); fig.savefig(OUT / "fig2_industry_home_origin.png", dpi=140)


def fig3_demographics() -> None:
    d = pd.read_csv(OUT / "04_industry_demographics.csv").set_index("industry")
    ind_order = ["All Seattle workers (baseline)",
                 "Health Care & Social Assistance (CNS16 / NAICS 62)",
                 "Educational Services (CNS15 / NAICS 61)",
                 "Public Administration (CNS20 / NAICS 92)"]
    short = {"All Seattle workers (baseline)": "All Seattle\nworkers",
             "Health Care & Social Assistance (CNS16 / NAICS 62)": "Health\nCare",
             "Educational Services (CNS15 / NAICS 61)": "Education",
             "Public Administration (CNS20 / NAICS 92)": "Government"}
    d = d.reindex(ind_order)
    panels = [
        ("Female", "CS02", "% Female"),
        ("Earn >$3.33k/mo", "CE03", "% Higher earnings"),
        ("Age 55+", "CA03", "% Workers age 55+"),
        ("Bachelor's+ (age 30+)", "CD04", "% with Bachelor's+ (age 30+)"),
    ]
    fig, axes = plt.subplots(1, 4, figsize=(13, 3.5), sharey=False)
    colors = ["#888888", "#d62728", "#1f77b4", "#2ca02c"]
    for ax, (title, col, ylab) in zip(axes, panels):
        ax.bar([short[i] for i in d.index], d[col], color=colors)
        for i, v in enumerate(d[col]):
            ax.text(i, v + 0.5, f"{v:.0f}%", ha="center")
        ax.set_title(title); ax.set_ylabel(ylab)
        ax.set_ylim(0, max(d[col]) * 1.18)
    fig.suptitle("Workers in Seattle: demographic comparison across institutional sectors")
    fig.tight_layout(); fig.savefig(OUT / "fig3_demographics.png", dpi=140)


if __name__ == "__main__":
    fig1_topline_inflow()
    fig2_industry_commute()
    fig3_demographics()
    print("plots written to", OUT)
