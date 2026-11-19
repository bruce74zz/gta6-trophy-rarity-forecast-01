"""Experiment: which trend clock / decay best forecasts the final case?

Compares model variants on the same simulated launch weeks as
tools/simulate.py, across scenarios and seeds. Variants differ in the
clock the count trends are fitted on (linear time with decaying growth
vs log-time since launch) and in the regression window. The winner is
ported into main.py.

Usage:
    python3 tools/experiment.py [--seeds 7,11,23]
"""

import argparse
import json
import math
import sys
import tempfile
from datetime import datetime, timedelta, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import main as M  # noqa: E402
from simulate import (  # noqa: E402
    SCENARIOS, TARGET, CHECKPOINT, LAUNCH, Z, LEVELS,
    simulate, make_case, pinball_loss, baseline_forecast,
    curve_baseline_forecast,
)

LAUNCH_EPOCH = LAUNCH.timestamp()


def clock_value(t_epoch, clock):
    if clock == "linear":
        return t_epoch / 86400.0
    # log-time since launch (hours), the house curve baseline's clock
    hours = (t_epoch - LAUNCH_EPOCH) / 3600.0
    return math.log1p(max(hours, -0.99))


def variant_forecast(case, clock="logtime", tau_e=2.0, tau_n=3.0,
                     window_hours=36.0):
    trophies = case["trophies"]
    history = sorted(case.get("history") or [], key=lambda r: r.get("at", ""))
    horizon_days = float(case["horizon_minutes"]) / 1440.0
    reads = []
    seen = set()
    for read in history:
        try:
            t = M.parse_time(read["at"])
            players = read.get("players_tracked")
            rates = read.get("rates") or []
        except (KeyError, TypeError, ValueError):
            continue
        if players is None or players <= 0 or not rates or t in seen:
            continue
        seen.add(t)
        reads.append((t, float(players), rates))
    if not reads:
        return M.fallback_forecast(case)
    t_last = reads[-1][0]
    n_last = reads[-1][1]
    last_rates = reads[-1][2]

    def fit(points):
        """points: [(t_epoch, count)]. Returns (slope, se_slope) on the
        chosen clock, fitted over the last window_hours."""
        if len(points) < 2:
            return 0.0, 0.0
        cutoff = t_last - window_hours * 3600.0
        recent = [p for p in points if p[0] >= cutoff]
        use = recent if len(recent) >= 3 else points
        xs = [clock_value(t, clock) for t, _ in use]
        ys = [math.log(v + M.COUNT_SMOOTH) for _, v in use]
        slope, _i, _r, se, _n = M.ols(xs, ys)
        return slope, se

    def project(last_count, slope, tau, h_eff_clock):
        if clock == "linear":
            return (last_count + M.COUNT_SMOOTH) * math.exp(
                slope * h_eff_clock) - M.COUNT_SMOOTH
        return (last_count + M.COUNT_SMOOTH) * math.exp(
            slope * h_eff_clock) - M.COUNT_SMOOTH

    # --- players -----------------------------------------------------------
    g_n, se_n = fit([(t, n) for t, n, _ in reads])
    g_n = M.clamp(g_n, 0.0, M.MAX_GROWTH_PER_DAY)
    if clock == "linear":
        h_n = M.decaying_horizon(horizon_days, tau_n)
        n_target = n_last * math.exp(g_n * h_n)
        h_e = M.decaying_horizon(horizon_days, tau_e)
    else:
        x_last = clock_value(t_last, clock)
        x_tgt = clock_value(
            M.parse_time(case["target"]), clock)
        h_n = x_tgt - x_last
        n_target = n_last * math.exp(g_n * h_n)
        h_e = x_tgt - x_last

    medians, sigmas, earners = [], [], []
    for i, _t in enumerate(trophies):
        points, last_rate = [], None
        for t, n, rates in reads:
            if i < len(rates) and rates[i] is not None:
                rate = float(rates[i])
                points.append((t, rate / 100.0 * n))
                last_rate = rate
        if last_rate is None:
            last_rate = 0.0
        e_last = points[-1][1] if points else 0.0
        g_e, se_e = fit(points)
        g_e = M.clamp(g_e, 0.0, M.MAX_GROWTH_PER_DAY)
        e_target = project(e_last, g_e, tau_e, h_e)
        e_target = max(e_target, e_last)
        earners.append(e_target)
        median = M.clamp(100.0 * e_target / n_target,
                         M.RATE_FLOOR, M.RATE_CEIL)
        medians.append(median)
        move = abs(M.logit(median) - M.logit(last_rate))
        var = ((M.BASE_SIGMA * math.sqrt(horizon_days)) ** 2
               + (M.TREND_SPREAD * move) ** 2
               + (se_e * h_e) ** 2 + (se_n * h_n) ** 2)
        sigmas.append(M.clamp(math.sqrt(var), M.SIGMA_FLOOR, M.SIGMA_CAP))

    chain = M.story_chain_order(trophies)
    for a, b in zip(chain, chain[1:]):
        if earners[b] > earners[a]:
            earners[b] = earners[a]
            medians[b] = M.clamp(100.0 * earners[b] / n_target,
                                 M.RATE_FLOOR, M.RATE_CEIL)
    rows = []
    for i, t in enumerate(trophies):
        lg = M.logit(medians[i])
        sigma = sigmas[i]
        qs, prev = [], None
        for z in Z:
            q = M.round4(100.0 * M.logistic(lg + z * sigma))
            q = M.clamp(q, M.RATE_FLOOR, M.RATE_CEIL)
            if prev is not None and q < prev:
                q = prev
            qs.append(q)
            prev = q
        rows.append({"id": t["id"], "q": qs})
    return {"trophies": rows}


def blend_forecast(f1, f2, w=0.5, disagree=0.0):
    """Average two forecasts in logit space. f1 is the linear-clock
    forecast, f2 the log-time one; w weights f1. `disagree` scales the
    extra spread added for the two medians disagreeing (in logit units,
    half the distance between them)."""
    rows = []
    for r1, r2 in zip(f1["trophies"], f2["trophies"]):
        m1 = r1["q"][2]
        m2 = r2["q"][2]
        lg = w * M.logit(m1) + (1 - w) * M.logit(m2)
        s1 = (M.logit(r1["q"][4]) - M.logit(m1)) / 1.645
        s2 = (M.logit(r2["q"][4]) - M.logit(m2)) / 1.645
        sigma = math.sqrt(0.5 * (s1 ** 2 + s2 ** 2)
                          + (disagree * abs(M.logit(m1) - M.logit(m2))) ** 2)
        sigma = M.clamp(sigma, M.SIGMA_FLOOR, M.SIGMA_CAP)
        qs, prev = [], None
        for z in Z:
            q = M.round4(100.0 * M.logistic(lg + z * sigma))
            q = M.clamp(q, M.RATE_FLOOR, M.RATE_CEIL)
            if prev is not None and q < prev:
                q = prev
            qs.append(q)
            prev = q
        rows.append({"id": r1["id"], "q": qs})
    return {"trophies": rows}


VARIANTS = {
    "linear-tau2-w24": dict(clock="linear", tau_e=2.0, tau_n=3.0, window_hours=24.0),
    "linear-tau2-w18": dict(clock="linear", tau_e=2.0, tau_n=3.0, window_hours=18.0),
    "linear-tau2-w30": dict(clock="linear", tau_e=2.0, tau_n=3.0, window_hours=30.0),
    "linear-tau1.5-w24": dict(clock="linear", tau_e=1.5, tau_n=2.5, window_hours=24.0),
    "linear-tau2.5-w24": dict(clock="linear", tau_e=2.5, tau_n=3.5, window_hours=24.0),
    "logtime-w36": dict(clock="logtime", window_hours=36.0),
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", default="7,11,23,31,42")
    ap.add_argument("--scenarios", default=",".join(SCENARIOS))
    args = ap.parse_args()
    seeds = [int(s) for s in args.seeds.split(",")]
    names = [s for s in args.scenarios.split(",") if s in SCENARIOS]
    totals = {k: [] for k in list(VARIANTS) + [
        "blend-50-d0", "blend-50-d0.5", "blend-60-d0.5", "blend-70-d0.5",
        "baseline", "curve"]}
    for seed in seeds:
        for name in names:
            cfg = SCENARIOS[name]
            reads, answer, trophies = simulate(
                cfg["p_power"], cfg["t_hours"], cfg["prog_mult"], seed)
            case = make_case(reads, trophies)
            for vname, vcfg in VARIANTS.items():
                totals[vname].append(
                    pinball_loss(variant_forecast(case, **vcfg), answer))
            logtime = variant_forecast(case, **VARIANTS["logtime-w36"])
            linear = variant_forecast(case, **VARIANTS["linear-tau2-w24"])
            totals["blend-50-d0"].append(
                pinball_loss(blend_forecast(logtime, linear, 0.5, 0.0), answer))
            totals["blend-50-d0.5"].append(
                pinball_loss(blend_forecast(logtime, linear, 0.5, 0.5), answer))
            totals["blend-60-d0.5"].append(
                pinball_loss(blend_forecast(logtime, linear, 0.6, 0.5), answer))
            totals["blend-70-d0.5"].append(
                pinball_loss(blend_forecast(logtime, linear, 0.7, 0.5), answer))
            totals["baseline"].append(pinball_loss(baseline_forecast(case), answer))
            totals["curve"].append(
                pinball_loss(curve_baseline_forecast(case), answer))
    print(f"{'variant':24s} {'mean loss':>10s} {'worst':>8s} {'beats curve?':>11s}")
    curve_mean = sum(totals["curve"]) / len(totals["curve"])
    for k, v in totals.items():
        mean = sum(v) / len(v)
        print(f"{k:24s} {mean:10.4f} {max(v):8.4f} "
              f"{(curve_mean - mean) / curve_mean * 100:10.1f}%")


if __name__ == "__main__":
    main()
