"""Check every forecast against the rules' output format; score it when an answer file sits beside the case.

Usage: python3 backtest/validate.py <input_dir> <output_dir>
Exits 1 when any case's output is invalid.
"""
import json
import math
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from backtest import pinball  # noqa: E402

HOURS = set(range(2, 25))


def problems(path):
    if not os.path.isfile(path) or os.path.islink(path):
        return ["missing or not a regular file"]
    if os.path.getsize(path) > 65536:
        return ["over 65,536 bytes"]
    try:
        with open(path, encoding="utf-8") as f:
            doc = json.load(f)
    except (UnicodeDecodeError, json.JSONDecodeError) as e:
        return [f"not JSON: {e}"]
    if not isinstance(doc, dict) or set(doc) != {"hours"} or not isinstance(doc["hours"], list):
        return ["not an object with exactly one key, hours"]
    errs, seen = [], []
    if len(doc["hours"]) != 23:
        errs.append(f"{len(doc['hours'])} rows, not 23")
    for row in doc["hours"]:
        if not isinstance(row, dict) or set(row) != {"hour", "q"}:
            errs.append(f"row {row!r} keys are not exactly hour, q")
            continue
        h, q = row["hour"], row["q"]
        seen.append(h)
        ok = (isinstance(q, list) and len(q) == 5
              and all(isinstance(x, (int, float)) and not isinstance(x, bool) and math.isfinite(x) and 0 < x <= 1e9 for x in q)
              and all(a <= b for a, b in zip(q, q[1:])))
        if not ok:
            errs.append(f"hour {h}: quantiles {q!r} not five finite positive non-decreasing numbers <= 1e9")
    if sorted(seen) != sorted(HOURS):
        errs.append(f"hours are {sorted(seen)}, not 2..24 each once")
    return errs


def main():
    src, dst = sys.argv[1], sys.argv[2]
    bad = 0
    for name in sorted(os.listdir(src)):
        if not name.endswith(".in"):
            continue
        cid = name[:-3]
        errs = problems(os.path.join(dst, cid + ".out"))
        bad += bool(errs)
        line = f"{cid}: {'INVALID ' + '; '.join(errs) if errs else 'valid'}"
        answer_path = os.path.join(src, cid + ".answer")
        if not errs and os.path.exists(answer_path):
            with open(answer_path, encoding="utf-8") as f:
                answer = {int(k): v for k, v in json.load(f).items()}
            with open(os.path.join(dst, cid + ".out"), encoding="utf-8") as f:
                rows = {r["hour"]: r["q"] for r in json.load(f)["hours"]}
            losses = [pinball(y, rows[h]) for h, y in answer.items()]
            cover = sum(rows[h][0] <= y <= rows[h][4] for h, y in answer.items()) / len(answer)
            line += f"  loss {sum(losses) / len(losses):.4f}  90% coverage {cover:.2f}"
        print(line)
    sys.exit(1 if bad else 0)


if __name__ == "__main__":
    main()
