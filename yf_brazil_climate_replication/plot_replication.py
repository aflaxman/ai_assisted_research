"""Four-panel summary figure for the YF replication."""
import json, pathlib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = pathlib.Path(__file__).resolve().parent
B = json.load(open(ROOT / "results" / "replication_bundle.json"))

fig, ax = plt.subplots(2, 2, figsize=(12, 8.5))
obs = B["observed"]

# A. published parameters vs observed
a = ax[0, 0]
a.plot(obs["weeks"], obs["cases"], "o--", color="#C44536", ms=5, label="observed 2017/18")
pub = B["published_run"]["series"]
a.plot(pub["weeks"][:18], pub["cases"][:18], "-", color="#1B4965", lw=2,
       label="published parameters")
a.set_yscale("log"); a.set_xlim(0, 18)
a.set_xlabel("week"); a.set_ylabel("symptomatic cases (log scale)")
a.set_title("A. The published parameter set overshoots 24-fold", loc="left", fontsize=11)
a.legend(fontsize=8, frameon=False)

# B. refits vs observed
b = ax[0, 1]
b.plot(obs["weeks"], obs["cases"], "o--", color="#C44536", ms=5, label="observed 2017/18")
for key, col, lab in (("fit_table1", "#1B4965", "refit, Table 1 mosquito params"),
                      ("fit_t23", "#5F8C62", "refit, Table 3 polys at 23 C")):
    s = B[key]["series"]
    b.plot(s["weeks"][:18], s["cases"][:18], "-", color=col, lw=2, label=lab)
b.set_xlim(0, 18); b.set_xlabel("week"); b.set_ylabel("symptomatic cases")
b.set_title("B. Refitting recovers the observed curve", loc="left", fontsize=11)
b.legend(fontsize=8, frameon=False)

# C. climate scenarios, consistent parameterisation
c = ax[1, 0]
cols = {23.0: "#222222", 25.9: "#1B4965", 26.5: "#3E7CB1", 26.9: "#7FB2D6",
        26.0: "#8C5F8C", 27.0: "#C06C84", 28.3: "#C44536"}
for r in [r for r in B["table4"] if r["variant"] == "B_consistent"]:
    lab = "2017/18 baseline (23 C)" if r["rcp"] == "baseline" else f"{r['rcp']} {r['temp']} C"
    c.plot(range(1, len(r["series"]) + 1), r["series"], "-",
           color=cols[r["temp"]], lw=2.2 if r["rcp"] == "baseline" else 1.4, label=lab)
c.set_xlim(0, 25); c.set_xlabel("week"); c.set_ylabel("symptomatic cases")
c.set_title("C. Warming shortens the outbreak but not its peak", loc="left", fontsize=11)
c.legend(fontsize=7, frameon=False)

# D. cumulative incidence: mine vs paper
d = ax[1, 1]
paper = B["paper"]["table4"]
temps, mine, theirs = [], [], []
for r in [r for r in B["table4"] if r["variant"] == "B_consistent" and r["rcp"] != "baseline"]:
    temps.append(f"{r['rcp'][3:]}\n{r['temp']}C")
    mine.append(r["d_cum"])
    theirs.append(100 * (paper[f"('{r['rcp']}', {r['temp']})"]["cum"] - 1254) / 1254)
x = range(len(temps))
d.bar([i - 0.2 for i in x], theirs, 0.4, color="#C44536", label="paper as published")
d.bar([i + 0.2 for i in x], mine, 0.4, color="#1B4965", label="this replication")
d.set_xticks(list(x)); d.set_xticklabels(temps, fontsize=7)
d.axhline(0, color="#888", lw=0.8)
d.set_ylabel("change in cumulative incidence (%)")
d.set_title("D. The cumulative-incidence decline replicates", loc="left", fontsize=11)
d.legend(fontsize=8, frameon=False)

for a_ in ax.flat:
    a_.spines[["top", "right"]].set_visible(False)
fig.tight_layout()
fig.savefig(ROOT / "results" / "yf_replication.png", dpi=150)
print("wrote results/yf_replication.png")
