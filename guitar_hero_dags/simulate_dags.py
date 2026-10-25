"""Simulate the competing DAGs behind the Guitar Hero demo-station story.

Every scenario is calibrated so a cross-sectional look at store-weeks
shows demo stores selling roughly 9x more Guitar Hero than non-demo
stores -- the contrast Walmart noticed (18 vs 2 units/week). The
scenarios differ only in the causal structure generating that contrast,
so the same observed table hides wildly different answers to "what
happens if we mandate demos chain-wide?"

Scenarios:
- H1  demos_work:      demo -> sales, demos assigned at random
- H2  superstar_mgr:   manager enthusiasm -> demo AND -> GH sales (GH-specific)
- H2b general_hustle:  as H2, but hustle also lifts other games somewhat
- H3  busy_stores:     store traffic -> demo AND -> ALL game sales
- H4  sales_to_demo:   a hot early week -> manager installs demo (reverse)
- H5  noticed:         demos truly work (3x) but the contrast is built by
                       noticing high-outcome stores first, Walmart-style

Each scenario reports:
- observed RR: mean GH sales, demo vs no-demo stores (the anecdote's 9x)
- Madden RR:  the same contrast for an unrelated game (negative control)
- rollout:    chain mean sales with demos mandated everywhere, divided by
              the status-quo chain mean (what acting on the belief delivers)
- for H4/H5:  within-store week 1 -> week 2 change among demo stores

Run: python simulate_dags.py
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

OUT = Path(__file__).parent / "outputs"
LAM_GH = 2.0  # baseline Guitar Hero units/week
LAM_MADDEN = 5.0  # baseline Madden units/week (negative-control outcome)
N = 400_000  # store-weeks per scenario

# Validated categorical palette, slots 1-2 (matches evalue_examples figure).
BLUE, ORANGE = "#2a78d6", "#eb6834"


def rr(outcome, exposed):
    return outcome[exposed].mean() / outcome[~exposed].mean()


def max_fakeable_rr(e):
    """Largest observed RR a zero-effect world can show when a single
    binary confounder has strength e on BOTH arms (Ding & VanderWeele
    bounding factor at RR_EU = RR_UD = e)."""
    return e * e / (2 * e - 1)


def sim_equal_strength_boundary(e, n, rng):
    """A zero-effect world saturating the bound at equal strength e:
    every demo store has the confounder, 1/e of non-demo stores do,
    and the confounder multiplies sales by e."""
    demo = rng.random(n) < 0.10
    m = rng.random(n) < np.where(demo, 1.0, 1.0 / e)
    gh = rng.poisson(LAM_GH * np.where(m, e, 1.0))
    return rr(gh, demo)


def h1_demos_work(n, rng, effect=9.0):
    demo = rng.random(n) < 0.10
    gh = rng.poisson(LAM_GH * np.where(demo, effect, 1.0))
    madden = rng.poisson(LAM_MADDEN, n)
    rollout = LAM_GH * effect / (LAM_GH * np.where(demo, effect, 1.0)).mean()
    return dict(name="H1 demos work (9x)", true_rr=effect,
                observed=rr(gh, demo), madden=rr(madden, demo),
                rollout=rollout)


def _confounded(n, rng, prev, q_conf, q_plain, k_gh, k_madden, name):
    """Shared machinery for H2/H2b/H3: confounder -> demo, -> sales."""
    conf = rng.random(n) < prev
    demo = rng.random(n) < np.where(conf, q_conf, q_plain)
    gh_rate = LAM_GH * np.where(conf, k_gh, 1.0)  # no demo arrow
    gh = rng.poisson(gh_rate)
    madden = rng.poisson(LAM_MADDEN * np.where(conf, k_madden, 1.0))
    p1, p0 = conf[demo].mean(), conf[~demo].mean()
    return dict(name=name, true_rr=1.0, observed=rr(gh, demo),
                madden=rr(madden, demo), rollout=1.0,
                rr_eu=p1 / p0, rr_ud=k_gh)


def h2_superstar_manager(n, rng):
    # Solved so the observed RR is ~9 with zero causal effect: superstar
    # managers (10% of stores) sell 9.84x more GH and almost always set
    # up demos. GH-specific enthusiasm: Madden untouched.
    return _confounded(n, rng, prev=0.10, q_conf=0.95, q_plain=0.005,
                       k_gh=9.84, k_madden=1.0,
                       name="H2 superstar manager (GH-specific)")


def h2b_general_hustle(n, rng):
    # Same manager, but hustle lifts every game somewhat.
    return _confounded(n, rng, prev=0.10, q_conf=0.95, q_plain=0.005,
                       k_gh=9.84, k_madden=2.5,
                       name="H2b general hustle")


def h3_busy_stores(n, rng):
    # High-traffic stores (25%) sell 12x more of EVERYTHING and almost
    # all of them have demos.
    return _confounded(n, rng, prev=0.25, q_conf=0.90, q_plain=0.002,
                       k_gh=12.0, k_madden=12.0, name="H3 busy stores")


def h4_sales_to_demo(n, rng):
    # Persistent store-level GH popularity (heavy-tailed), no demo effect.
    # Managers install demos after a hot week 1; week 2 has the same rates.
    g = rng.gamma(shape=0.25, scale=4.0, size=n)  # mean 1, heavy tail
    week1 = rng.poisson(LAM_GH * g)
    demo = week1 >= np.quantile(week1, 0.95)
    week2 = rng.poisson(LAM_GH * g)
    madden = rng.poisson(LAM_MADDEN, n)
    return dict(name="H4 sales -> demo (reverse)", true_rr=1.0,
                observed=rr(week2, demo), madden=rr(madden, demo),
                rollout=1.0,
                within_store=week2[demo].mean() - week1[demo].mean())


def h5_noticed(n, rng, effect=3.0, notice_at=15):
    # Demos truly work (3x, randomly adopted by 10% of stores), with
    # store-level heterogeneity. Corporate scans week 1, notices stores
    # selling >= notice_at, finds they have demos, and quotes THEIR sales
    # against everyone else's -- the anecdote's arithmetic.
    g = rng.gamma(shape=0.5, scale=2.0, size=n)  # mean 1
    demo = rng.random(n) < 0.10
    rate = LAM_GH * g * np.where(demo, effect, 1.0)
    week1 = rng.poisson(rate)
    week2 = rng.poisson(rate)
    noticed = (week1 >= notice_at) & demo
    reported = week1[noticed].mean() / week1[~noticed].mean()
    fair = rr(week1, demo)  # what a proper cohort contrast would report
    madden = rng.poisson(LAM_MADDEN, n)
    rollout = (LAM_GH * g * effect).mean() / rate.mean()
    return dict(name="H5 noticed (true effect 3x)", true_rr=effect,
                observed=reported, madden=rr(madden, demo), rollout=rollout,
                fair_rr=fair,
                within_store=week2[noticed].mean() - week1[noticed].mean())


def run_all(n=N, seed=20260821):
    rng = np.random.default_rng(seed)
    return [h1_demos_work(n, rng), h2_superstar_manager(n, rng),
            h2b_general_hustle(n, rng), h3_busy_stores(n, rng),
            h4_sales_to_demo(n, rng), h5_noticed(n, rng)]


def fig_evalue_curve(rng):
    """How an observed RR of 9 demands confounder strength 17.5."""
    es = np.linspace(1.001, 25, 400)
    fig, ax = plt.subplots(figsize=(7.6, 5.0), dpi=150)
    fig.patch.set_facecolor("#fcfcfb")
    ax.set_facecolor("#fcfcfb")

    ax.plot(es, max_fakeable_rr(es), color=BLUE, lw=2,
            label="largest fakeable RR: $E^2/(2E-1)$")
    sim_es = [3, 6, 9, 12, 17.49, 22]
    sims = [sim_equal_strength_boundary(e, 300_000, rng) for e in sim_es]
    ax.scatter(sim_es, sims, s=70, color=BLUE, edgecolor="#fcfcfb",
               linewidth=1.5, zorder=3, label="simulated zero-effect worlds")

    ax.axhline(9, color="#c3c2b7", lw=1.2, ls="--", zorder=0)
    ax.annotate("observed RR 9 (Walmart, 18 vs 2)", (1.5, 9.25),
                fontsize=9, color="#5c5b53")
    ax.scatter([17.49], [9], s=90, color=ORANGE, edgecolor="#fcfcfb",
               linewidth=1.5, zorder=4)
    ax.annotate("E-value: strength 17.5 on both\narms just reaches RR 9",
                (17.49, 9), xytext=(8, -66), textcoords="offset points",
                fontsize=9, color="#5c5b53",
                arrowprops=dict(arrowstyle="-", color="#c3c2b7"))
    e9 = max_fakeable_rr(9.0)
    ax.scatter([9], [e9], s=90, color=ORANGE, edgecolor="#fcfcfb",
               linewidth=1.5, zorder=4)
    ax.annotate(f"a confounder as strong as the claimed\n"
                f"effect (9 on both arms) fakes only RR {e9:.1f}",
                (9, e9), xytext=(14, -34), textcoords="offset points",
                fontsize=9, color="#5c5b53",
                arrowprops=dict(arrowstyle="-", color="#c3c2b7"))

    ax.set_xlabel("confounder strength E on both arms "
                  "(RR with exposure = RR with outcome)",
                  fontsize=9, color="#5c5b53")
    ax.set_ylabel("largest observed RR a zero-effect world can show",
                  fontsize=9, color="#5c5b53")
    ax.set_xlim(0, 25.5)
    ax.set_ylim(0, 14)
    ax.tick_params(colors="#5c5b53", labelsize=9)
    ax.grid(color="#e8e7e0", lw=0.8, zorder=0)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color("#c3c2b7")
    ax.legend(loc="upper left", frameon=False, fontsize=9)
    ax.set_title("Why faking a 9x association takes a 17.5x confounder",
                 fontsize=11, color="#1a1a19", loc="left", pad=10)
    fig.tight_layout()
    fig.savefig(OUT / "evalue_curve.png", bbox_inches="tight")
    print(f"Wrote {OUT / 'evalue_curve.png'}")


def fig_observed_vs_rollout(results):
    """Same observed association, very different rollout payoffs."""
    fig, ax = plt.subplots(figsize=(8.6, 5.2), dpi=150)
    fig.patch.set_facecolor("#fcfcfb")
    ax.set_facecolor("#fcfcfb")

    short = ["H1\ndemos\nwork", "H2\nsuperstar\nmanager",
             "H2b\ngeneral\nhustle", "H3\nbusy\nstores",
             "H4\nsales\n→ demo", "H5\nnoticed\n(true 3×)"]
    xs = np.arange(len(results), dtype=float)
    dot = dict(s=80, zorder=3, edgecolor="#fcfcfb", linewidth=1.5)
    for x, res in zip(xs, results):
        obs, roll = res["observed"], res["rollout"]
        ax.plot([x, x], [obs, roll], color="#c3c2b7", lw=2, zorder=1)
        ax.scatter([x], [obs], color=ORANGE, **dot)
        ax.scatter([x], [roll], color=BLUE, **dot)
        ax.annotate(f"{obs:.1f}", (x, obs), xytext=(10, -4),
                    textcoords="offset points", fontsize=9, color="#5c5b53")
        ax.annotate(f"{roll:.1f}", (x, roll), xytext=(10, -4),
                    textcoords="offset points", fontsize=9, color="#5c5b53")

    ax.set_yscale("log")
    ax.set_yticks([1, 2, 3, 5, 9, 12])
    ax.set_yticklabels(["1×", "2×", "3×", "5×", "9×", "12×"])
    ax.set_ylim(0.8, 26)
    ax.set_xticks(xs)
    ax.set_xticklabels(short, fontsize=9)
    ax.set_ylabel("ratio (log scale)", fontsize=9, color="#5c5b53")
    ax.tick_params(colors="#5c5b53", labelsize=9)
    ax.grid(axis="y", color="#e8e7e0", lw=0.8, zorder=0)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color("#c3c2b7")
    ax.scatter([], [], color=ORANGE, s=80,
               label="association you observe (demo vs no-demo sales)")
    ax.scatter([], [], color=BLUE, s=80,
               label="what chain-wide rollout delivers (× chain sales)")
    ax.legend(loc="upper left", frameon=False, fontsize=9)
    ax.set_title("Six worlds, one anecdote: every scenario shows demo stores "
                 "selling ~9×,\nbut mandating demos pays off in only two",
                 fontsize=11, color="#1a1a19", loc="left", pad=10)
    h5 = results[5]
    fig.text(0.01, 0.005,
             "Simulated store-weeks calibrated to the VICE oral-history "
             f"numbers (18 vs 2 units/week). H5's observed "
             f"{h5['observed']:.1f}× is the noticed-stores contrast; its "
             f"fair cohort RR is {h5['fair_rr']:.1f}×.",
             fontsize=7.5, color="#8a897e")
    fig.tight_layout()
    fig.savefig(OUT / "observed_vs_rollout.png", bbox_inches="tight")
    print(f"Wrote {OUT / 'observed_vs_rollout.png'}")


def main():
    OUT.mkdir(exist_ok=True)
    results = run_all()

    print(f"{'scenario':<36}{'true RR':>8}{'observed':>10}"
          f"{'Madden':>8}{'rollout':>9}")
    for res in results:
        print(f"{res['name']:<36}{res['true_rr']:>8.1f}"
              f"{res['observed']:>10.1f}{res['madden']:>8.1f}"
              f"{res['rollout']:>9.1f}")

    h2, h5 = results[1], results[5]
    print()
    print(f"H2 confounder strength: RR with demo {h2['rr_eu']:.0f}, "
          f"RR with sales {h2['rr_ud']:.1f} "
          "(both exceed 9 -- they must, to fake RR 9)")
    print(f"H4 within-store change after demo: "
          f"{results[4]['within_store']:+.1f} units/week (regression to the mean)")
    print(f"H5 fair cohort RR: {h5['fair_rr']:.1f} "
          f"(true 3.0), but the noticed-stores contrast reports "
          f"{h5['observed']:.1f}; noticed stores' next week: "
          f"{h5['within_store']:+.1f} units")

    rng = np.random.default_rng(1)
    fig_evalue_curve(rng)
    fig_observed_vs_rollout(results)


if __name__ == "__main__":
    main()
