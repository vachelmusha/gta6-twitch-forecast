"""Leave-one-out backtest over past launches, cut at the end of hour 1 of release day.

Usage: python3 backtest/backtest.py [kit_input_dir] [variant,variant,...]

kit_input_dir defaults to $MAKEFUN_KIT_INPUT, else the first gta/gta6-twitch-ccu-forecast-01-kit/input
found beside this repository or any directory above it (a git worktree sits deeper than the checkout).

Each launch in the union of the practice cases' comparables becomes a pseudo-case:
hour 1's count is known, hours 2..24 are the answer. Every candidate is fitted on the
other launches only, and its spread comes from residuals that also exclude the target.
Losses are the competition's pinball loss on the log scale.
"""
import glob
import json
import math
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import main  # noqa: E402

KIT_INPUT_TAIL = os.path.join("gta", "gta6-twitch-ccu-forecast-01-kit", "input")


def find_kit_input():
    env = os.environ.get("MAKEFUN_KIT_INPUT")
    if env:
        return env
    d = os.path.dirname(os.path.abspath(__file__))
    while True:
        cand = os.path.join(d, KIT_INPUT_TAIL)
        if glob.glob(os.path.join(cand, "*.in")):
            return cand
        parent = os.path.dirname(d)
        if parent == d:
            return os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", KIT_INPUT_TAIL)
        d = parent


DEFAULT_KIT_INPUT = find_kit_input()


def load_comparables(kit_input):
    """The union of the practice cases' comparables, by name."""
    seen = {}
    for path in sorted(glob.glob(os.path.join(kit_input, "*.in"))):
        with open(path, encoding="utf-8") as f:
            for comp in json.load(f)["comparables"]:
                seen.setdefault(comp["name"], comp)
    return list(seen.values())


def load_launches(kit_input):
    launches = []
    for comp in load_comparables(kit_input):
        L = main.launch_from_comparable(comp)
        if L:
            L["p"] = comp_p(comp)
            launches.append(L)
    return launches


def comp_p(comp):
    day0 = main.hour_index(comp["day_start"])
    return max(main.series_of(comp["hourly"], day0)[0], 1.0)


def pinball(y, qs):
    ly = math.log(max(y, 1.0))
    total = 0.0
    for tau, q in zip(main.LEVELS, qs):
        d = ly - math.log(q)
        total += 2 * max(tau * d, (tau - 1) * d)
    return total / len(qs)


def score(launch, fc):
    """Mean item loss and coverage hits over the launch's printed hours 2..24."""
    p = launch["p"]
    rows = {r["hour"]: r["q"] for r in main.finalize(fc, p)["hours"]}
    losses, hits = [], []
    for h, lr in launch["logratio"].items():
        y = p * math.exp(lr)
        losses.append(pinball(y, rows[h]))
        hits.append((rows[h][0] <= y <= rows[h][4], rows[h][1] <= y <= rows[h][3], h))
    return losses, hits


def flat_weights(launches):
    return [dict(L, weight=1.0) for L in launches]


BASE = dict(main.CONFIG, center="mean", feature=None, pool=2, spread="empirical", wpow=1.0, size_bw=1.5,
            align="none", shrink=0.0, inner=1.0)
US = dict(BASE, align="us")

VARIANTS = {
    "house_baseline": None,
    "chosen (main.CONFIG)": dict(main.CONFIG),
    "old size1.5": dict(BASE),
    "us": US,
    "us_ref": dict(BASE, align="us_ref"),
    "us_inner0.75": dict(US, inner=0.75),
    "us_shrink0.25": dict(US, shrink=0.25),
    "us_i.75_s.25_median": dict(US, inner=0.75, shrink=0.25, center="median"),
    "us_i.75_s.25_jump": dict(US, inner=0.75, shrink=0.25, feature="jump"),
    "us_i.75_s.25_over_eve": dict(US, inner=0.75, shrink=0.25, feature="over_eve"),
    "us_i.75_s.5": dict(US, inner=0.75, shrink=0.5),
    "us_i.75_s.25_size2": dict(US, inner=0.75, shrink=0.25, size_bw=2.0),
}


def evaluate(launches, cfg):
    per = []
    pool = flat_weights(launches) if cfg and cfg.get("flat") else launches
    for i, L in enumerate(launches):
        if cfg is None:
            fc = main.flat_forecast(L["p"])
        else:
            train = pool[:i] + pool[i + 1:]
            fc = main.model_forecast(train, L["p"], L, cfg)
        per.append((L, *score(L, fc)))
    return per


def summarize(per):
    def agg(rows, weighted):
        num = den = cov = n = 0.0
        for L, losses, hits in rows:
            w = L["weight"] if weighted else 1.0
            if not losses or w <= 0:
                continue
            num += w * sum(losses) / len(losses)
            cov += w * sum(x[0] for x in hits) / len(hits)
            den += w
        return num / den, cov / den
    rolling = [r for r in per if r[0]["weight"] >= 0.7]
    all_l, all_c = agg(per, False)
    w_l, w_c = agg(per, True)
    r_l, r_c = agg(rolling, False)
    return all_l, all_c, w_l, w_c, r_l, r_c, len(rolling)


def calibration(per):
    """Coverage of the 5-95 and 25-75 bands by horizon block, all launches and the GTA-like subset."""
    blocks = ((2, 5), (6, 9), (10, 16), (17, 24))
    for label, rows in (("all", per), ("gta", [r for r in per if r[0]["weight"] >= 0.7])):
        cells = []
        for lo, hi in blocks:
            hs = [x for _, _, hits in rows for x in hits if lo <= x[2] <= hi]
            cells.append(f"h{lo}-{hi} 90:{sum(x[0] for x in hs) / len(hs):.2f} 50:{sum(x[1] for x in hs) / len(hs):.2f}")
        print(f"    {label:4s} " + "  ".join(cells))


def main_cli():
    kit_input = sys.argv[1] if len(sys.argv) > 1 and sys.argv[1] else DEFAULT_KIT_INPUT
    launches = load_launches(kit_input)
    if not launches:
        sys.exit(f"0 launches read from {kit_input}: pass the kit's input/ or set MAKEFUN_KIT_INPUT")
    names = sys.argv[2].split(",") if len(sys.argv) > 2 else list(VARIANTS)
    print(f"{len(launches)} launches; GTA-like subset = prior weight >= 0.7")
    print(f"{'variant':28s} {'loss_all':>9s} {'cov_all':>8s} {'loss_wtd':>9s} {'cov_wtd':>8s} {'loss_gta':>9s} {'cov_gta':>8s}")
    for name in names:
        per = evaluate(launches, VARIANTS[name])
        s = summarize(per)
        if os.environ.get("PER_LAUNCH"):
            for L, losses, hits in per:
                if L["weight"] >= 0.7:
                    print(f"    {L['name'][:30]:30s} {sum(losses) / len(losses):.3f} cov {sum(x[0] for x in hits) / len(hits):.2f}")
        if os.environ.get("CALIBRATION"):
            calibration(per)
        print(f"{name:28s} {s[0]:9.4f} {s[1]:8.3f} {s[2]:9.4f} {s[3]:8.3f} {s[4]:9.4f} {s[5]:8.3f}  (n_gta={s[6]})")


if __name__ == "__main__":
    main_cli()
