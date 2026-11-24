"""Generate the policy-signal covariate and the ground-truth parameter file.

The policy is a single intervention: intensity 0.85 switching on at day 25 and
held to the end. The Python model applies DELAYED ADAPTATION (a geometric lag
rho) on top of that step; here we pre-bake the resulting exponential ramp into
the covariate itself, so camdl's `policy(t)` already carries the delay. The
model then reads this tsv through its `forcing { policy : interpolated ... }`
block.
"""

from pathlib import Path

HERE = Path(__file__).parent
N_DAYS = 120
POLICY_START = 25
POLICY_INTENSITY = 0.85
ADAPT_RATE = 0.3  # rho: geometric adaptation toward the target each day

# Ground-truth parameters used to simulate the synthetic "observed" series.
TRUTH = {
    "beta": 0.55,
    "gamma": 0.2,
    "endog_delta": 0.5,
    "endog_x0": 1200.0,
    "endog_nu": 3.0,
    "policy_weight": 0.7,
    "sigma_c2": 0.02,
    "i0": 10.0,
}


def policy_signal():
    """Exponentially-smoothed step -> the delay-adapted policy pressure."""
    rows, adapted = [], 0.0
    for t in range(N_DAYS + 1):
        target = POLICY_INTENSITY if t >= POLICY_START else 0.0
        adapted = (1 - ADAPT_RATE) * adapted + ADAPT_RATE * target
        rows.append((t, adapted))
    return rows


def write_policy_tsv():
    path = HERE / "data" / "policy_signal.tsv"
    with path.open("w") as f:
        f.write("time\tpolicy\n")
        for t, p in policy_signal():
            f.write(f"{t}\t{p:.6f}\n")
    print(f"wrote {path}")


def write_truth_toml():
    path = HERE / "params" / "truth.toml"
    with path.open("w") as f:
        f.write("# Ground-truth parameters for the camdl policy-behavioral SIR.\n")
        f.write("# Used by `camdl simulate` to generate the synthetic case series\n")
        f.write("# that the fit then tries to recover.\n\n")
        for k, v in TRUTH.items():
            f.write(f"{k} = {v}\n")
    print(f"wrote {path}")


if __name__ == "__main__":
    write_policy_tsv()
    write_truth_toml()
