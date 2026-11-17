"""Write final-shaped test cases into an output directory.

Usage: python3 backtest/make_synthetic.py <out_dir> [kit_input_dir]

h00.in replays a rolling-midnight console launch (left out of its own comparables) in the final's
shape: minute reads from two days before the release day, hourly means of them, the top 20 streams,
Twitch's top categories hourly and the launch's pre-release category history among the comparables.
h00.answer holds that launch's real hours 2..24. x01.in and x02.in are hostile shapes: no
comparables with null history, and a missing hour-1 print.
"""
import json
import math
import os
import sys
from datetime import datetime, timedelta, timezone

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import main  # noqa: E402
import backtest  # noqa: E402

LAUNCH = "The Last of Us Part II"


def iso(t):
    return t.strftime("%Y-%m-%dT%H:%M:%SZ")


def wobble(i):
    return 1.0 + 0.04 * math.sin(i * 0.7) + 0.02 * math.sin(i * 2.3)


def build(comps):
    comp = next(c for c in comps if c["name"] == LAUNCH)
    day0 = datetime.fromisoformat(comp["day_start"].replace("Z", "+00:00"))
    hourly = {datetime.fromisoformat(r["from"].replace("Z", "+00:00")): r for r in comp["hourly"]}
    start, checkpoint = day0 - timedelta(days=2), day0 + timedelta(hours=1)

    history, t, i = [], start, 0
    while t < checkpoint:
        hour = t.replace(minute=0)
        here, nxt = hourly.get(hour), hourly.get(hour + timedelta(hours=1))
        if here:
            frac = t.minute / 60
            base = here["viewers"] * (1 - frac / 2) + (nxt or here)["viewers"] * frac / 2
            history.append({"at": iso(t), "viewers": round(base * wobble(i)), "channels": here["channels"]})
        t += timedelta(minutes=1)
        i += 1

    by_hour = {}
    for r in history:
        by_hour.setdefault(r["at"][:13], []).append(r)
    history_hourly = [{"from": k + ":00:00Z", "viewers": round(sum(r["viewers"] for r in v) / len(v)),
                       "channels": round(sum(r["channels"] for r in v) / len(v)), "reads": len(v)}
                      for k, v in sorted(by_hour.items())]
    hour1 = history_hourly[-1]
    p = hour1["viewers"]

    streams = [{"login": f"streamer{k:02d}", "viewers": round(p * 0.18 / (k + 1)), "language": "en" if k % 3 else "de",
                "started_at": iso(day0 - timedelta(hours=k % 5))} for k in range(20)]
    categories = [{"from": r["from"], "top": [{"name": LAUNCH, "viewers": r["viewers"], "channels": r["channels"]}]
                   + [{"name": f"Other {k}", "viewers": 50000 - 2000 * k, "channels": 900} for k in range(19)]}
                  for r in history_hourly]

    pre_cut = start + timedelta(hours=2)
    target_history = {"name": f"{LAUNCH} category before release", "day_start": comp["day_start"],
                      "unlock": "Not a launch: the target's own category before release.", "platforms": comp["platforms"],
                      "source": comp["source"],
                      "hourly": [r for r in comp["hourly"] if datetime.fromisoformat(r["from"].replace("Z", "+00:00")) < pre_cut]}
    others = [c for c in comps if c["name"] != LAUNCH and "Not a launch" not in (c.get("unlock") or "")]

    case = {
        "schema": "makefun/twitch-ccu-case/v1", "id": "h00",
        "game": {"name": LAUNCH, "twitch_id": "0"},
        "day_start": comp["day_start"], "checkpoint": iso(checkpoint), "levels": list(main.LEVELS),
        "known": [{"hour": 1, "from": iso(day0), "until": iso(checkpoint), "viewers": p,
                   "channels": hour1["channels"], "reads": hour1["reads"]}],
        "forecast": [{"hour": h, "from": iso(day0 + timedelta(hours=h - 1)), "until": iso(day0 + timedelta(hours=h))}
                     for h in main.HOURS],
        "history": history, "history_hourly": history_hourly[:-1] + [hour1],
        "streams": streams, "categories_hourly": categories,
        "comparables": others + [target_history],
        "notes": f"Synthetic final-shaped case replaying {LAUNCH}.",
    }
    answer = {h: hourly[day0 + timedelta(hours=h - 1)]["viewers"] for h in main.HOURS
              if day0 + timedelta(hours=h - 1) in hourly}
    return case, answer


def main_cli():
    out = sys.argv[1]
    kit_input = sys.argv[2] if len(sys.argv) > 2 else backtest.DEFAULT_KIT_INPUT
    case, answer = build(backtest.load_comparables(kit_input))
    os.makedirs(out, exist_ok=True)
    hostile_empty = dict(case, id="x01", comparables=[], history=None, history_hourly=None, streams=None,
                         categories_hourly=None)
    hostile_noprint = dict(case, id="x02", known=[{"hour": 1, "viewers": None}])
    for name, obj in (("h00.in", case), ("x01.in", hostile_empty), ("x02.in", hostile_noprint),
                      ("h00.answer", answer)):
        with open(os.path.join(out, name), "w", encoding="utf-8") as f:
            json.dump(obj, f)
    print(f"wrote h00 (hour 1 = {case['known'][0]['viewers']}, {len(case['history'])} minute reads), x01, x02 to {out}")


if __name__ == "__main__":
    main_cli()
