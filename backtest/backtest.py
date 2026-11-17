"""Leave-one-out backtest over past launches, cut at the end of hour 1 of release day.

Usage: python3 backtest/backtest.py [kit_input_dir] [variant,variant,...]

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

DEFAULT_KIT_INPUT = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "..", "..", "gta", "gta6-twitch-ccu-forecast-01-kit", "input")


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
        hits.append(rows[h][0] <= y <= rows[h][4])
    return losses, hits


def flat_weights(launches):
    return [dict(L, weight=1.0) for L in launches]


BASE = dict(main.CONFIG, center="mean", feature=None, ridge=1.0, pool=2, spread="empirical", wpow=1.0, size_bw=None)

VARIANTS = {
    "house_baseline": None,
    "house_like_ratio_spread": dict(BASE, spread="normal", flat=True),
    "chosen (main.CONFIG)": dict(main.CONFIG),
    "prior_weights": dict(BASE),
    "prior_median": dict(BASE, center="median"),
    "prior_normal": dict(BASE, spread="normal"),
    "prior_reg_jump": dict(BASE, feature="jump"),
    "prior_reg_over_eve": dict(BASE, feature="over_eve"),
    "prior_reg_eve_growth": dict(BASE, feature="eve_growth"),
    "prior_reg_over_peak": dict(BASE, feature="over_peak"),
    "prior_reg_over_peak48": dict(BASE, feature="over_peak48"),
    "prior2": dict(BASE, wpow=2.0),
    "prior2_reg_over_peak": dict(BASE, wpow=2.0, feature="over_peak"),
    "size1": dict(BASE, size_bw=1.0),
    "size1.5": dict(BASE, size_bw=1.5),
    "size2": dict(BASE, size_bw=2.0),
    "size2.5": dict(BASE, size_bw=2.5),
    "size1.5_flatprior": dict(BASE, size_bw=1.5, flat=True),
    "size1.5_prior2": dict(BASE, size_bw=1.5, wpow=2.0),
    "size1.5_median": dict(BASE, size_bw=1.5, center="median"),
    "size1.5_normal": dict(BASE, size_bw=1.5, spread="normal"),
    "size1.5_pool1": dict(BASE, size_bw=1.5, pool=1),
    "size1.5_pool3": dict(BASE, size_bw=1.5, pool=3),
    "size1.5_reg_jump": dict(BASE, size_bw=1.5, feature="jump"),
    "size1.5_reg_over_eve": dict(BASE, size_bw=1.5, feature="over_eve"),
    "size1.5_reg_over_peak": dict(BASE, size_bw=1.5, feature="over_peak"),
}


def evaluate(launches, cfg):
    per = []
    pool = flat_weights(launches) if cfg and cfg.get("flat") else launches
    for i, L in enumerate(launches):
        if cfg is None:
            fc = main.flat_forecast(L["p"])
        else:
            train = pool[:i] + pool[i + 1:]
            fc = main.model_forecast(train, L["p"], L["feats"], cfg)
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
            cov += w * sum(hits) / len(hits)
            den += w
        return num / den, cov / den
    rolling = [r for r in per if r[0]["weight"] >= 0.7]
    all_l, all_c = agg(per, False)
    w_l, w_c = agg(per, True)
    r_l, r_c = agg(rolling, False)
    return all_l, all_c, w_l, w_c, r_l, r_c, len(rolling)


def main_cli():
    kit_input = sys.argv[1] if len(sys.argv) > 1 and sys.argv[1] else DEFAULT_KIT_INPUT
    launches = load_launches(kit_input)
    names = sys.argv[2].split(",") if len(sys.argv) > 2 else list(VARIANTS)
    print(f"{len(launches)} launches; GTA-like subset = prior weight >= 0.7")
    print(f"{'variant':28s} {'loss_all':>9s} {'cov_all':>8s} {'loss_wtd':>9s} {'cov_wtd':>8s} {'loss_gta':>9s} {'cov_gta':>8s}")
    for name in names:
        per = evaluate(launches, VARIANTS[name])
        s = summarize(per)
        if os.environ.get("PER_LAUNCH"):
            for L, losses, hits in per:
                if L["weight"] >= 0.7:
                    print(f"    {L['name'][:30]:30s} {sum(losses) / len(losses):.3f} cov {sum(hits) / len(hits):.2f}")
        print(f"{name:28s} {s[0]:9.4f} {s[1]:8.3f} {s[2]:9.4f} {s[3]:8.3f} {s[4]:9.4f} {s[5]:8.3f}  (n_gta={s[6]})")


if __name__ == "__main__":
    main_cli()
