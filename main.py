"""Forecast a launch day's Twitch category viewers for hours 2..24 from hour 1.

Each comparable launch becomes a curve of log(viewers at hour h / viewers at hour 1).
Curves are shifted so every launch's US-east midnight unlock (04:00Z under US daylight time,
05:00Z otherwise) lands on the target's: the Americas arrive in the same forecast hour.
The forecast centre for each hour is a similarity-weighted average of those curves,
and the quantiles come from the leave-one-out residuals of that same centre.
Similarity is the unlock style and tier the notes give, times a kernel on hour 1's size.
When the case carries minute reads and live streams (the final), they nudge the first hours.

CONFIG is the backtest's winner; its other settings are the variants backtest/ compares.
"""
import json
import math
import os
from datetime import date, datetime, timedelta

LEVELS = (0.05, 0.25, 0.5, 0.75, 0.95)
Z = (-1.6448536, -0.6744898, 0.0, 0.6744898, 1.6448536)
HOURS = range(2, 25)
MAX_Q = 1e9

CONFIG = {
    "center": "mean",
    "feature": None,
    "ridge": 1.0,
    "pool": 2,
    "spread": "empirical",
    "wpow": 1.0,
    "size_bw": 1.5,
    "align": "us",
    "shrink": 0.25,
    "inner": 0.75,
}


def parse_time(stamp):
    return datetime.fromisoformat(stamp.replace("Z", "+00:00"))


def hour_index(stamp):
    return int(parse_time(stamp).timestamp() // 3600)


def nth_sunday(year, month, n):
    first = date(year, month, 1)
    return first + timedelta(days=(6 - first.weekday()) % 7 + 7 * (n - 1))


def us_unlock_index(day_start):
    """Hour index (0 = hour 1) of US-east midnight on the release day: 04:00Z under daylight time, else 05:00Z."""
    d = parse_time(day_start).date()
    dst = nth_sunday(d.year, 3, 2) <= d < nth_sunday(d.year, 11, 1)
    return 4 if dst else 5


def prior_weight(comp):
    """Similarity to GTA VI read from the unlock note, tier and platforms."""
    text = (comp.get("unlock") or "").lower()
    platforms = (comp.get("platforms") or "").lower()
    if "not a launch" in text:
        return 0.0
    has_rolling = "midnight local" in text
    has_simultaneous = any(k in text for k in ("simultaneous", "single global", "single time"))
    if has_rolling and has_simultaneous:
        w = 0.7
    elif has_rolling:
        w = 1.0
    elif has_simultaneous:
        w = 0.2
    else:
        w = 0.5
    if "early-access" in text or ("early access" in text and "no early access" not in text):
        w *= 0.3
    if "tier: weak" in text or "free-to-play" in platforms:
        w *= 0.3
    if "pc" not in platforms:
        w *= 1.5
    return w


def series_of(rows, day0):
    out = {}
    for r in rows or []:
        try:
            v = r.get("viewers")
            if isinstance(v, (int, float)) and v >= 0 and math.isfinite(v):
                out[hour_index(r["from"]) - day0] = float(v)
        except (AttributeError, KeyError, TypeError, ValueError):
            continue
    return out


def mean_over(pre, lo, hi):
    vals = [pre[k] for k in range(lo, hi + 1) if k in pre]
    return sum(vals) / len(vals) if vals else None


def features(pre, p):
    """Pre-release trajectory signals, each a log ratio; None when the history is missing."""
    lp = math.log(max(p, 1.0))
    last = next((pre[k] for k in range(-1, -4, -1) if k in pre), None)
    eve1 = mean_over(pre, -12, -1)
    eve2 = mean_over(pre, -36, -25)
    recent = [pre[k] for k in range(-48, 0) if k in pre]
    return {
        "jump": lp - math.log(max(last, 1.0)) if last is not None else None,
        "over_eve": lp - math.log(max(eve1, 1.0)) if eve1 is not None else None,
        "eve_growth": math.log(max(eve1, 1.0) / max(eve2, 1.0)) if eve1 and eve2 else None,
        "over_peak": lp - math.log(max(max(pre.values()), 1.0)) if pre else None,
        "over_peak48": lp - math.log(max(max(recent), 1.0)) if recent else None,
        "logp": lp,
    }


def launch_from_comparable(comp):
    try:
        day0 = hour_index(comp["day_start"])
        series = series_of(comp.get("hourly"), day0)
        p = series.get(0)
        if p is None or p <= 0:
            return None
        rel = {k: math.log(max(v, 1.0) / p) for k, v in series.items()}
        return {
            "name": comp.get("name", ""),
            "weight": prior_weight(comp),
            "feats": features({k: v for k, v in series.items() if k < 0}, p),
            "us": us_unlock_index(comp["day_start"]),
            "rel": rel,
            "logratio": {h: rel[h - 1] for h in HOURS if (h - 1) in rel},
        }
    except (AttributeError, KeyError, TypeError, ValueError):
        return None


def curve_value(L, h, target, cfg):
    """Launch L's log ratio for the target's hour h, shifted so both launches' US unlock coincide."""
    k = target["us"] - L["us"] if cfg["align"] != "none" else 0
    y = L["rel"].get(h - 1 - k)
    if y is None:
        return None
    if cfg["align"] == "us_ref":
        ref = L["rel"].get(-k)
        return None if ref is None else y - ref
    return y


def weighted_quantile(pairs, tau):
    pairs = sorted(pairs)
    total = sum(w for _, w in pairs)
    if total <= 0:
        return 0.0
    acc = 0.0
    for i, (v, w) in enumerate(pairs):
        mid = acc + w / 2
        if mid >= tau * total:
            if i == 0:
                return v
            pv, pw = pairs[i - 1]
            pmid = acc - pw / 2
            t = (tau * total - pmid) / (mid - pmid) if mid > pmid else 1.0
            return pv + t * (v - pv)
        acc += w
    return pairs[-1][0]


def affinity(L, feats, size_range, cfg):
    """Prior similarity sharpened by wpow, times a kernel on hour-1 size (target clipped to the training range)."""
    w = L["weight"] ** cfg["wpow"]
    if cfg["size_bw"] and w > 0:
        lo, hi = size_range
        d = (min(max(feats["logp"], lo), hi) - L["feats"]["logp"]) / cfg["size_bw"]
        w *= math.exp(-0.5 * d * d)
    return w


def size_range(train):
    sizes = [L["feats"]["logp"] for L in train if L["weight"] > 0]
    return (min(sizes), max(sizes)) if sizes else (0.0, 30.0)


def center(train, h, target, cfg):
    feats = target["feats"]
    rng = size_range(train)
    pts = []
    for L in train:
        if L["weight"] <= 0:
            continue
        y = curve_value(L, h, target, cfg)
        w = affinity(L, feats, rng, cfg)
        if y is not None and w > 0:
            pts.append((y, w, L["feats"].get(cfg["feature"]) if cfg["feature"] else None))
    if not pts:
        return 0.0
    if cfg["center"] == "median":
        return weighted_quantile([(y, w) for y, w, _ in pts], 0.5)
    sw = sum(w for _, w, _ in pts)
    ybar = sum(y * w for y, w, _ in pts) / sw
    x0 = feats.get(cfg["feature"]) if cfg["feature"] else None
    usable = [(y, w, x) for y, w, x in pts if x is not None]
    if x0 is None or len(usable) < 5:
        return ybar
    uw = sum(w for _, w, _ in usable)
    xbar = sum(x * w for _, w, x in usable) / uw
    yb = sum(y * w for y, w, _ in usable) / uw
    sxy = sum(w * (x - xbar) * (y - yb) for y, w, x in usable)
    sxx = sum(w * (x - xbar) ** 2 for _, w, x in usable)
    slope = sxy / (sxx + cfg["ridge"] * uw)
    lo = min(x for _, _, x in usable)
    hi = max(x for _, _, x in usable)
    return yb + slope * (min(max(x0, lo), hi) - xbar)


def centers(train, target, cfg):
    return {h: center(train, h, target, cfg) for h in HOURS}


def residual_quantiles(train, target, cfg):
    """Per-hour quantiles of each training launch's error when it is left out of the fit, weighted by
    that launch's similarity to the target, shrunk toward equal weights by cfg["shrink"]. The 25-75 band
    is narrowed by cfg["inner"]: unnarrowed, it covered about two thirds of GTA-like launches' hours."""
    resid = {h: [] for h in HOURS}
    rng = size_range(train)
    live = [(i, L, affinity(L, target["feats"], rng, cfg)) for i, L in enumerate(train) if L["weight"] > 0]
    total = sum(a for _, _, a in live) or 1.0
    for i, L, a in live:
        w = (1 - cfg["shrink"]) * a / total + cfg["shrink"] / len(live)
        if w <= 0:
            continue
        mu = centers(train[:i] + train[i + 1:], L, cfg)
        for h, y in L["logratio"].items():
            resid[h].append((y - mu[h], w))
    out = {}
    for h in HOURS:
        pooled = [r for k in range(h - cfg["pool"], h + cfg["pool"] + 1) for r in resid.get(k, [])]
        if not pooled:
            out[h] = [0.15 * math.sqrt(h - 1) * z for z in Z]
        elif cfg["spread"] == "normal":
            sigma = math.sqrt(sum(r * r * w for r, w in pooled) / sum(w for _, w in pooled))
            out[h] = [sigma * z for z in Z]
        else:
            qs = [weighted_quantile(pooled, t) for t in LEVELS]
            gain = (1.0, cfg["inner"], 0.0, cfg["inner"], 1.0)
            out[h] = [qs[2] + g * (q - qs[2]) for g, q in zip(gain, qs)]
    return out


def model_forecast(train, p, target, cfg=CONFIG):
    mu = centers(train, target, cfg)
    rq = residual_quantiles(train, target, cfg)
    return {h: [p * math.exp(mu[h] + r) for r in rq[h]] for h in HOURS}


def flat_forecast(p):
    return {h: [p * math.exp(z * 0.15 * math.sqrt(h - 1)) for z in Z] for h in HOURS}


def finalize(fc, p):
    rows = []
    for h in HOURS:
        qs = fc.get(h) if fc else None
        if not qs or len(qs) != 5 or not all(isinstance(q, float) and math.isfinite(q) for q in qs):
            qs = flat_forecast(p)[h]
        clean, prev = [], 1e-3
        for q in qs:
            prev = min(max(q, prev, 1e-3), MAX_Q)
            clean.append(round(prev, 2) if prev >= 1 else prev)
        rows.append({"hour": h, "q": clean})
    return {"hours": rows}


def read_time(row):
    try:
        return parse_time(row["at"]).timestamp()
    except (KeyError, TypeError, ValueError, AttributeError):
        return None


def minute_reads(case):
    out = []
    for r in case.get("history") or []:
        t = read_time(r)
        v = r.get("viewers") if isinstance(r, dict) else None
        if t is not None and isinstance(v, (int, float)) and math.isfinite(v) and v >= 0:
            out.append((t, float(v)))
    return out


def hour1_print(case):
    """Hour 1's print, else the mean of the history inside hour 1, else the last history point."""
    try:
        v = float(case["known"][0]["viewers"])
        if math.isfinite(v):
            return max(v, 1.0)
    except (KeyError, IndexError, TypeError, ValueError):
        pass
    pts = minute_reads(case)
    try:
        day0 = hour_index(case["day_start"]) * 3600
        inside = [v for t, v in pts if day0 <= t < day0 + 3600]
        if inside:
            return max(sum(inside) / len(inside), 1.0)
    except (KeyError, TypeError, ValueError, AttributeError):
        pass
    return max(pts[-1][1], 1.0) if pts else 1.0


def pre_release(case, day0):
    """Hourly viewers before hour 1: the house's own hourly means first, then Twitch's top-categories
    snapshots, then the target's own pre-release category history among the comparables."""
    name = (case.get("game") or {}).get("name") or ""
    pre = {}
    for comp in case.get("comparables") or []:
        if name and isinstance(comp, dict) and (comp.get("name") or "").startswith(name) and prior_weight(comp) == 0.0:
            pre.update(series_of(comp.get("hourly"), day0))
    for snap in case.get("categories_hourly") or []:
        try:
            for cat in snap.get("top") or []:
                v = cat.get("viewers")
                if cat.get("name") == name and isinstance(v, (int, float)) and math.isfinite(v):
                    pre[hour_index(snap["from"]) - day0] = float(v)
        except (AttributeError, KeyError, TypeError, ValueError):
            continue
    pre.update(series_of(case.get("history_hourly"), day0))
    return {k: v for k, v in pre.items() if k < 0}


def minute_momentum(case, p):
    """log(level over hour 1's last 10 minutes / hour 1's print), from minute reads; 0 without them."""
    end = hour_index(case["checkpoint"]) * 3600
    tail = [v for t, v in minute_reads(case) if end - 600 <= t < end and v > 0]
    if len(tail) < 5:
        return 0.0
    return max(-0.3, min(0.3, math.log(sum(tail) / len(tail) / p)))


def top_stream_share(case, p):
    viewers = [s.get("viewers") for s in case.get("streams") or [] if isinstance(s, dict)]
    viewers = [v for v in viewers if isinstance(v, (int, float)) and v > 0]
    return min(max(viewers) / p, 0.9) if viewers else 0.0


def live_adjust(fc, momentum, share):
    """Shift the first hours by half the in-hour momentum (halving each hour) and lower their bottom
    quantiles by the top stream's share, which one streamer going offline would take away."""
    out = {}
    for h, qs in fc.items():
        shift = math.exp(0.5 * momentum * 0.5 ** (h - 2))
        drop = 1.0 - share * 0.5 ** (h - 2)
        out[h] = [q * shift * (drop if i < 2 else 1.0) for i, q in enumerate(qs)]
    return out


def safe(fn, default):
    try:
        return fn()
    except Exception:
        return default


def forecast_case(case):
    p = hour1_print(case)
    day0 = hour_index(case["day_start"])
    pre = safe(lambda: pre_release(case, day0), {})
    train = [L for L in map(launch_from_comparable, case.get("comparables") or []) if L]
    target = {"feats": features(pre, p), "us": us_unlock_index(case["day_start"])}
    fc = model_forecast(train, p, target)
    momentum = safe(lambda: minute_momentum(case, p), 0.0)
    share = safe(lambda: top_stream_share(case, p), 0.0)
    return live_adjust(fc, momentum, share), p


def run_case(case):
    p = safe(lambda: hour1_print(case), 1.0)
    try:
        fc, p = forecast_case(case)
        return finalize(fc, p)
    except Exception:
        return finalize(flat_forecast(p), p)


def main():
    src = os.environ.get("MAKEFUN_INPUT_DIR") or "/input/"
    dst = os.environ.get("MAKEFUN_OUTPUT_DIR") or "/output/"
    os.makedirs(dst, exist_ok=True)
    for name in sorted(os.listdir(src)):
        if not name.endswith(".in"):
            continue
        try:
            with open(os.path.join(src, name), encoding="utf-8") as f:
                out = run_case(json.load(f))
        except Exception:
            out = finalize(flat_forecast(1.0), 1.0)
        with open(os.path.join(dst, name[:-3] + ".out"), "w", encoding="utf-8") as f:
            json.dump(out, f, separators=(",", ":"))


if __name__ == "__main__":
    main()
