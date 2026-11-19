"""Validate main.py on simulated launch weeks.

The final case's history does not exist yet (it accumulates until the
deadline), and the practice cases are flat (the launch-night page never
refreshed), so no real data can validate the model's trend logic. This
script builds synthetic launch weeks instead, from a cohort model:

- Tracked players N(t) grow as a power law, N0 * (1 + t/T)**p, fitted to
  the two real snapshots available at build time: N = 638 at
  2026-11-18T23:50Z and N = 3,868 at 2026-11-19T04:30Z (the house's
  recorder), and N ~ 15-25k at ~13:00Z on 2026-11-19 (the live Exophase
  page fetched while building, whose 0.01-point rate granularity implies
  N of that order). The central scenario (p = 1.6, T = 2h) passes near
  all three points.
- Each trophy i has an eventual share a_i of all players and a
  lognormal time-to-earn (median d_i hours). Players join at rate N'(t)
  and earn the trophy after their own delay, so E_i(t) is the cohort
  convolution. Classes: first mission (a=1.0, d=0.5h), story chapters
  (a 0.8..0.5, d 4..26h), final mission (a=0.38, d=42h), 100%
  (a=0.06, d=160h), platinum (a=0.02, d=220h), side missions, quick
  activities and grinds by keyword, with per-trophy jitter.
- Exophase refreshes the page every 1-6 hours (random, seeded); the
  house reads every 30 minutes, then every 10 minutes, and records the
  page as it stands. Rates are rounded to 2 decimals, as the page
  prints them.

Cases are cut like the real final (checkpoint 2026-11-21T11:30Z, target
2026-11-26T00:00Z) and scored by the contest's own rule: mean pinball
loss on the logit scale over every trophy and quantile, against the
answering read (the page as of the target). main.py, the house baseline
(starter) and the house curve baseline (logit rate vs ln(1 + hours
since 2026-11-19T00:00Z) over the last 24h, spread = random walk and
fit residual combined) are all run on the same cases.

Usage:
    python3 tools/simulate.py [--scenarios S1,S3] [--seed N]
"""

import argparse
import json
import math
import random
import subprocess
import sys
import tempfile
from datetime import datetime, timedelta, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

KIT = Path.home() / ".makefun" / "gta6-trophy-rarity-forecast-01-kit"
REPO = Path(__file__).resolve().parent.parent
LAUNCH = datetime(2026, 11, 19, 0, 0, tzinfo=timezone.utc)
T0 = datetime(2026, 11, 18, 23, 50, tzinfo=timezone.utc)
CHECKPOINT = datetime(2026, 11, 21, 11, 30, tzinfo=timezone.utc)
TARGET = datetime(2026, 11, 26, 0, 0, tzinfo=timezone.utc)
LEVELS = [0.05, 0.25, 0.5, 0.75, 0.95]
Z = [-1.645, -0.674, 0.0, 0.674, 1.645]


def norm_cdf(x):
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def logit(pct):
    p = min(99.99, max(0.01, pct)) / 100.0
    return math.log(p / (1.0 - p))


def round2(x):
    scaled = x * 100
    whole = math.floor(scaled)
    if scaled - whole >= 0.5:
        whole += 1
    return whole / 100.0


def round4(x):
    scaled = x * 10000
    whole = math.floor(scaled)
    if scaled - whole >= 0.5:
        whole += 1
    value = whole / 10000
    return int(value) if value.is_integer() else value


# --- trophy progression classes, by keyword on name + description --------
def trophy_params(trophy, rng):
    name = trophy["name"].lower()
    desc = trophy["description"].lower()
    text = name + " " + desc
    jitter = lambda a, b: a * math.exp(rng.uniform(-b, b))
    if "gone fishing" in text:
        return 1.0, 0.5
    if "chapter 1" in text:
        return 0.80, 4.0
    if "chapter 2" in text:
        return 0.70, 9.0
    if "chapter 3" in text:
        return 0.60, 16.0
    if "chapter 4" in text:
        return 0.50, 26.0
    if "final mission" in text and "for " not in desc:
        return 0.38, 42.0
    if "100% completion" in text:
        return 0.06, 160.0
    if trophy["tier"] == "platinum":
        return 0.02, 220.0
    if "250 stunts" in text or "50 challenges" in text or "100 locations" in text:
        return jitter(0.05, 0.4), jitter(150.0, 0.4)
    if "collectible card" in text or "meditate" in text or "unique animal" in text:
        return jitter(0.06, 0.4), jitter(120.0, 0.4)
    if "strangers" in text or "mission strands" in text:
        return jitter(0.10, 0.4), jitter(90.0, 0.4)
    if "final mission for" in text:
        return jitter(0.30, 0.3), jitter(22.0, 0.3)
    if "complete " in desc and ("heist" in text or "operation" in text
                                or "takedown" in text or "breakfast run" in text):
        return jitter(0.35, 0.3), jitter(18.0, 0.3)
    if "gold medal" in text or "drone race" in text or "20 different races" in text:
        return jitter(0.18, 0.3), jitter(30.0, 0.3)
    if "fishing challenge" in text:
        return jitter(0.45, 0.2), jitter(6.0, 0.3)
    if "arcade game" in text or "bowling" in text or "cocktail" in text:
        return jitter(0.50, 0.2), jitter(4.0, 0.3)
    if "activity" in text:
        return jitter(0.45, 0.2), jitter(8.0, 0.3)
    return jitter(0.25, 0.4), jitter(25.0, 0.4)


def simulate(p_power, t_hours, prog_mult, refresh_seed):
    """Return (reads, answer_rates): reads = [(at, players, rates)] at the
    house cadence, answer_rates = the page's rates at the target."""
    rng = random.Random(refresh_seed)
    trophies = json.loads((KIT / "input" / "p00.in").read_text())["trophies"]
    params = [trophy_params(t, rng) for t in trophies]
    for i, (a, d) in enumerate(params):
        params[i] = (a, d * prog_mult)
    # Gone Fishing is the opening mission, done within minutes of starting:
    # its delay does not scale with the general pace (the live page shows
    # it at 99.99% thirteen hours after launch).
    for i, t in enumerate(trophies):
        if "gone fishing" in (t["name"] + " " + t["description"]).lower():
            params[i] = (params[i][0], 0.5)

    hours = []
    t = T0
    end = datetime(2026, 11, 26, 1, 10, tzinfo=timezone.utc)
    while t <= end:
        hours.append(t)
        t += timedelta(hours=1)
    n_hours = len(hours)

    def n_at(dt):
        h = (dt - T0).total_seconds() / 3600.0
        return 638.0 * (1.0 + h / t_hours) ** p_power

    # Cohort convolution on the hourly grid.
    join_rate = [0.0] * n_hours  # players joining in each hour
    for j in range(1, n_hours):
        join_rate[j] = n_at(hours[j]) - n_at(hours[j - 1])
    join_rate[0] = 638.0
    earners = [[0.0] * n_hours for _ in trophies]
    for j in range(n_hours):
        tj = hours[j]
        for k in range(j + 1):
            dt_h = max((tj - hours[k]).total_seconds() / 3600.0, 0.25)
            w = join_rate[k]
            if w <= 0:
                continue
            for i, (a, d) in enumerate(params):
                earners[i][j] += w * a * norm_cdf(
                    (math.log(dt_h) - math.log(d)) / 0.8
                )

    # True rate on the hourly grid.
    true_rate = []
    for j in range(n_hours):
        n = n_at(hours[j])
        true_rate.append([100.0 * earners[i][j] / n for i in range(len(trophies))])

    # Exophase refreshes every 1-6h; the page holds the last refresh.
    refresh_at = []
    rt = T0
    while rt <= end:
        refresh_at.append(rt)
        rt += timedelta(hours=rng.uniform(1.0, 6.0))
    refresh_idx = 0

    def page_at(dt):
        nonlocal refresh_idx
        while (refresh_idx + 1 < len(refresh_at)
               and refresh_at[refresh_idx + 1] <= dt):
            refresh_idx += 1
        # nearest hourly grid point at or before the refresh
        rt = refresh_at[refresh_idx]
        j = min(int((rt - T0).total_seconds() // 3600), n_hours - 1)
        return [round2(r) for r in true_rate[j]], n_at(rt)

    # House reads: every 30 min until 2026-11-19T06:00Z, then every 10 min.
    reads = []
    slot = T0
    switch = datetime(2026, 11, 19, 6, 0, tzinfo=timezone.utc)
    while slot < CHECKPOINT:
        rates, n = page_at(slot)
        reads.append({
            "slot": slot.strftime("%Y-%m-%dT%H:%M:%SZ"),
            "at": slot.strftime("%Y-%m-%dT%H:%M:%S.000Z"),
            "players_tracked": int(round(n)),
            "trophies_earned": int(round(sum(r / 100.0 * n for r in rates))),
            "rates": rates,
        })
        step = timedelta(minutes=30) if slot < switch else timedelta(minutes=10)
        slot += step
    answer, _n = page_at(TARGET)
    return reads, answer, trophies


def make_case(reads, trophies):
    horizon = int((TARGET - CHECKPOINT).total_seconds() // 60)
    return {
        "schema": "makefun/gta6-trophies-case/v1",
        "id": "f00",
        "at": CHECKPOINT.strftime("%Y-%m-%dT%H:%M:%SZ"),
        "horizon_minutes": horizon,
        "target": TARGET.strftime("%Y-%m-%dT%H:%M:%SZ"),
        "levels": LEVELS,
        "game": {
            "name": "Grand Theft Auto VI",
            "platform": "PS5",
            "release_date": "2026-11-19",
            "source": "https://www.exophase.com/game/grand-theft-auto-vi-psn/trophies/",
        },
        "trophies": trophies,
        "history": reads,
        "notes": "simulated case for local validation only",
    }


def pinball_loss(forecast, answer):
    rows = {r["id"]: r["q"] for r in forecast["trophies"]}
    total = 0.0
    count = 0
    for i, y in enumerate(answer, start=1):
        for tau, q in zip(LEVELS, rows[i]):
            d = logit(y) - logit(q)
            total += 2.0 * max(tau * d, (tau - 1.0) * d)
            count += 1
    return total / count


def run_program(program_dir, input_dir, output_dir):
    subprocess.run(
        [sys.executable, str(Path(program_dir) / "main.py")],
        env={"MAKEFUN_INPUT_DIR": str(input_dir),
             "MAKEFUN_OUTPUT_DIR": str(output_dir),
             "PATH": "/usr/bin:/bin"},
        check=True, capture_output=True, timeout=120,
    )
    return json.loads((Path(output_dir) / "f00.out").read_text())


def baseline_forecast(case):
    """The house baseline (starter), reimplemented here for scoring."""
    last = case["history"][-1]["rates"]
    d = case["horizon_minutes"] / 1440.0
    spread = 0.5 * math.sqrt(d)
    rows = []
    for i, t in enumerate(case["trophies"]):
        rate = last[i] if last[i] is not None else 0.0
        lg = logit(rate)
        qs = [round4(100.0 / (1.0 + math.exp(-(lg + z * spread)))) for z in Z]
        rows.append({"id": t["id"], "q": qs})
    return {"trophies": rows}


def curve_baseline_forecast(case):
    """The house curve baseline: per trophy, OLS of logit(rate) on
    ln(1 + hours since 2026-11-19T00:00Z) over the last 24h of reads;
    spread = random walk and fit residual combined."""
    last = case["history"][-1]["rates"]
    d = case["horizon_minutes"] / 1440.0
    spread_rw = 0.5 * math.sqrt(d)
    t_target = (TARGET - LAUNCH).total_seconds() / 3600.0
    x_target = math.log1p(t_target)
    cutoff = (datetime.fromisoformat(case["at"].replace("Z", "+00:00"))
              - timedelta(hours=24))
    rows = []
    for i, t in enumerate(case["trophies"]):
        xs, ys = [], []
        for read in case["history"]:
            rt = datetime.fromisoformat(read["at"].replace("Z", "+00:00"))
            if rt < cutoff or read["rates"][i] is None:
                continue
            xs.append(math.log1p((rt - LAUNCH).total_seconds() / 3600.0))
            ys.append(logit(read["rates"][i]))
        if len(xs) < 3:
            lg = logit(last[i] if last[i] is not None else 0.0)
            sigma = spread_rw
        else:
            n = len(xs)
            mx, my = sum(xs) / n, sum(ys) / n
            sxx = sum((x - mx) ** 2 for x in xs)
            slope = sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / sxx if sxx else 0.0
            inter = my - slope * mx
            resid = math.sqrt(sum((y - (inter + slope * x)) ** 2
                                  for x, y in zip(xs, ys)) / (n - 2)) if n > 2 else 0.0
            lg = inter + slope * x_target
            sigma = math.sqrt(spread_rw ** 2 + resid ** 2)
        qs = []
        for z in Z:
            q = round4(100.0 / (1.0 + math.exp(-(lg + z * sigma))))
            qs.append(min(99.99, max(0.01, q)))
        rows.append({"id": t["id"], "q": qs})
    return {"trophies": rows}


SCENARIOS = {
    # prog_mult is the pace of progression, calibrated so the simulated
    # page at 13h after launch matches the real Exophase page fetched
    # 2026-11-19 ~12:53 UTC (Chapter 1 0.73%, Chapter 2 0.14%, most
    # trophies 0.03-0.10%, Gone Fishing 99.99%).
    "S1-central": dict(p_power=1.6, t_hours=2.0, prog_mult=7.0),
    "S2-fast-adoption": dict(p_power=1.9, t_hours=2.0, prog_mult=5.0),
    "S3-slow-adoption": dict(p_power=1.3, t_hours=2.0, prog_mult=10.0),
    "S4-fast-adopt-slow-prog": dict(p_power=1.85, t_hours=2.0, prog_mult=11.0),
    "S5-slow-adopt-fast-prog": dict(p_power=1.35, t_hours=2.0, prog_mult=4.5),
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scenarios", default=",".join(SCENARIOS))
    ap.add_argument("--seed", type=int, default=7)
    args = ap.parse_args()
    names = [s for s in args.scenarios.split(",") if s in SCENARIOS]
    results = {}
    for name in names:
        cfg = SCENARIOS[name]
        reads, answer, trophies = simulate(
            cfg["p_power"], cfg["t_hours"], cfg["prog_mult"], args.seed)
        case = make_case(reads, trophies)
        with tempfile.TemporaryDirectory() as td:
            inp, out = Path(td) / "in", Path(td) / "out"
            inp.mkdir()
            (inp / "f00.in").write_text(json.dumps(case))
            out.mkdir()
            mine = run_program(REPO, inp, out)
        base = baseline_forecast(case)
        curve = curve_baseline_forecast(case)
        results[name] = {
            "mine": pinball_loss(mine, answer),
            "baseline": pinball_loss(base, answer),
            "curve": pinball_loss(curve, answer),
            "reads": len(reads),
            "players_at_checkpoint": reads[-1]["players_tracked"],
        }
        r = results[name]
        print(f"{name}: reads={r['reads']} N@checkpoint={r['players_at_checkpoint']:,} "
              f"mine={r['mine']:.4f} baseline={r['baseline']:.4f} "
              f"curve={r['curve']:.4f}")
    print()
    for key in ("mine", "baseline", "curve"):
        mean = sum(r[key] for r in results.values()) / len(results)
        print(f"mean {key}: {mean:.4f}")
    best_house = min(sum(r[k] for r in results.values()) / len(results)
                     for k in ("baseline", "curve"))
    mine_mean = sum(r["mine"] for r in results.values()) / len(results)
    print(f"mine beats best house baseline by "
          f"{(best_house - mine_mean) / best_house * 100:.1f}% on average")


if __name__ == "__main__":
    main()
