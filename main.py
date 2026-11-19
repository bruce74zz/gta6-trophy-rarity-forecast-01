"""gta6-trophy-rarity-forecast-01: forecast GTA VI trophy earn rates on Exophase.

The program reads one case file per run (makefun/gta6-trophies-case/v1) and
writes five quantiles (0.05, 0.25, 0.5, 0.75, 0.95) of every trophy's earn
rate at the case's target, as /output/<id>.out. Standard library only, no
network, deterministic: the same case always gives the same bytes.

Method (counts, not rates)
--------------------------
A trophy's earn rate is a ratio, r = E / N, of two growing counts: E, the
players who earned it, and N, the players Exophase tracks. Around a launch
both move fast and in opposite directions for the rate: new players arrive
having earned nothing yet, so N grows faster than E and a trophy's rate can
fall even while its earners multiply. Forecasting the rate directly (a trend
on r) mixes those two motions up; forecasting the two counts separately and
dividing keeps them apart.

For each trophy, and for N itself:

1. Recover the counts from every read: E = rate / 100 * players_tracked.
   The page prints rates to 2 decimals, so a rare trophy's count is noisy
   by a whole earner or two; the regression below averages that out.
2. Fit log(E + 0.5) against time by least squares over the last 24 hours of
   reads (the whole history when there are fewer), giving a daily growth
   rate g. Growth rates are clamped to >= 0: earners never un-earn, and a
   negative fit is rounding noise. The fit's residual spread is kept as
   an uncertainty term. The window is 24 hours because the launch week's
   dynamics change fast: older reads, from the frozen launch-night page
   and the first transient, drag the fit backwards. (Chosen on simulated
   launch weeks in tools/simulate.py: 24h beat 18h, 30h and 36h, and a
   2-day earner decay beat 1.5 and 2.5 days, against both house
   baselines across 25 simulated finals.)
3. Extrapolate with decaying growth: the growth rate is assumed to decay
   with an e-folding time of 2 days for earners and 3 days for the tracked
   base, so the effective horizon is tau * (1 - exp(-H / tau)) instead of
   the full horizon H. A launch week's growth cannot compound forever.
4. Apply the story chain: Gone Fishing is the first mission, then Chapter
   1..4, the final mission, 100% completion and the platinum nest inside
   each other, so no later step's earners can exceed the earlier step's.
   Each projected count is capped at the previous step's.
5. Median rate = 100 * E(target) / N(target), clamped to 0.01..99.99.

Spread (logit scale)
--------------------
The quantile spread is a random-walk term (0.5 logit units per root day,
the house baseline's) plus a trend term that grows with how far the median
moved from the last read (0.35 per logit unit of move) plus the slope
uncertainty of both count fits (their standard errors carried forward over
the effective horizon). The page's 2-decimal rounding noise does not grow
with the horizon, so it enters only through the slope's standard error. When the history is flat
(a frozen page, as in the practice cases) every extra term is zero and the
forecast is exactly the house baseline's: the median stays at the last
read and only the random-walk spread remains.

Output rows are rounded to 4 decimals, half up, like the starter's, and
forced non-decreasing. The program never crashes on a malformed case: a
case it cannot read falls back to the last read's rates (or 0.5% with a
wide spread when there is no history at all), so it always writes a valid
forecast file.

Run locally:
    mkdir -p out && MAKEFUN_INPUT_DIR=input MAKEFUN_OUTPUT_DIR=out python3 main.py
"""

import json
import math
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

# The grader's -P leaves this file's directory OFF sys.path: put it back
# first, so a module shipped beside this file imports.
sys.path.insert(0, str(Path(__file__).resolve().parent))

INPUT_DIR = Path(os.environ.get("MAKEFUN_INPUT_DIR", "/input"))
OUTPUT_DIR = Path(os.environ.get("MAKEFUN_OUTPUT_DIR", "/output"))

# The quantile levels the contest scores, in the order each row lists them.
LEVELS = [0.05, 0.25, 0.5, 0.75, 0.95]
# The standard normal's quantile at each level.
Z = [-1.645, -0.674, 0.0, 0.674, 1.645]

# --- model constants (fixed in advance; see the module docstring) ---------
WINDOW_HOURS = 24.0        # regression window for the trend fits
TAU_EARNERS_DAYS = 2.0    # e-folding time for the decay of earner growth
TAU_PLAYERS_DAYS = 3.0    # e-folding time for the decay of player growth
BASE_SIGMA = 0.5          # random-walk spread, logit units per root day
TREND_SPREAD = 0.35       # extra spread per logit unit the median moved
SIGMA_FLOOR = 0.30        # never narrower than this (logit units)
SIGMA_CAP = 2.5           # never wider than this (logit units)
MAX_GROWTH_PER_DAY = 3.0  # sanity cap on a fitted daily growth rate
COUNT_SMOOTH = 0.5        # added to earner counts before taking logs
RATE_FLOOR = 0.01         # rates are clamped to 0.01..99.99 for logits
RATE_CEIL = 99.99


def parse_time(text):
    """ISO 8601 UTC timestamp -> epoch seconds (float)."""
    dt = datetime.fromisoformat(str(text).replace("Z", "+00:00"))
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.timestamp()


def clamp(x, low, high):
    return min(high, max(low, x))


def logit(pct):
    p = clamp(pct, RATE_FLOOR, RATE_CEIL) / 100.0
    return math.log(p / (1.0 - p))


def logistic(x):
    if x >= 0:
        return 1.0 / (1.0 + math.exp(-x))
    ex = math.exp(x)
    return ex / (1.0 + ex)


def round4(x):
    """x rounded to 4 decimals, half up, as JavaScript's
    Math.round(x * 10000) / 10000 rounds it. A whole number comes back as
    an int, so json.dumps writes 100 where JavaScript writes 100."""
    scaled = x * 10000
    whole = math.floor(scaled)
    if scaled - whole >= 0.5:
        whole += 1
    value = whole / 10000
    return int(value) if value.is_integer() else value


def ols(xs, ys):
    """Least-squares line y = a + b*x. Returns (slope, intercept,
    residual std, slope standard error, n)."""
    n = len(xs)
    if n == 0:
        return 0.0, 0.0, 0.0, 0.0, 0
    if n == 1:
        return 0.0, ys[0], 0.0, 0.0, 1
    mx = sum(xs) / n
    my = sum(ys) / n
    sxx = sum((x - mx) ** 2 for x in xs)
    if sxx <= 0:
        return 0.0, my, 0.0, 0.0, n
    sxy = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    slope = sxy / sxx
    intercept = my - slope * mx
    if n > 2:
        resid = math.sqrt(
            sum((y - (intercept + slope * x)) ** 2 for x, y in zip(xs, ys))
            / (n - 2)
        )
        se_slope = resid / math.sqrt(sxx)
    else:
        resid = 0.0
        se_slope = 0.0
    return slope, intercept, resid, se_slope, n


def window_fit(points, t_last, window_hours):
    """Fit log-counts against time (in days) over the last `window_hours`
    of `points` = [(t_epoch, value), ...], oldest first. Falls back to the
    whole series when the window holds fewer than 3 points. Returns
    (growth_per_day, residual_std, slope_standard_error, window_days)."""
    if len(points) < 2:
        return 0.0, 0.0, 0.0, 0.0
    cutoff = t_last - window_hours * 3600.0
    recent = [p for p in points if p[0] >= cutoff]
    use = recent if len(recent) >= 3 else points
    t0 = use[0][0]
    xs = [(t - t0) / 86400.0 for t, _ in use]
    ys = [math.log(v + COUNT_SMOOTH) for _, v in use]
    slope, _intercept, resid, se_slope, _n = ols(xs, ys)
    window_days = max((use[-1][0] - use[0][0]) / 86400.0, 0.25)
    return slope, resid, se_slope, window_days


def decaying_horizon(horizon_days, tau_days):
    """Effective horizon when growth decays with e-folding time tau."""
    if horizon_days <= 0:
        return 0.0
    return tau_days * (1.0 - math.exp(-horizon_days / tau_days))


def story_chain_order(trophies):
    """Indices of the trophies that nest inside each other, earliest first:
    Gone Fishing, Chapter 1..4, the final mission, 100% completion, the
    platinum. Matched on the trophy's name/description, so a case that
    renames a trophy simply skips that step."""
    import re

    chain = []

    def find(pattern):
        for i, t in enumerate(trophies):
            text = (t.get("name", "") + " " + t.get("description", "")).lower()
            if re.search(pattern, text):
                return i
        return None

    first = find(r"gone fishing")
    if first is not None:
        chain.append(first)
    chapters = []
    for i, t in enumerate(trophies):
        m = re.search(
            r"chapter\s+([1-9])",
            (t.get("name", "") + " " + t.get("description", "")).lower(),
        )
        if m:
            chapters.append((int(m.group(1)), i))
    for _num, i in sorted(chapters):
        if i not in chain:
            chain.append(i)
    final = find(r"final mission")
    if final is not None and final not in chain:
        chain.append(final)
    completion = find(r"100%\s*completion")
    if completion is not None and completion not in chain:
        chain.append(completion)
    for i, t in enumerate(trophies):
        if t.get("tier") == "platinum" and i not in chain:
            chain.append(i)
    return chain


def forecast_case(case):
    trophies = case["trophies"]
    history = sorted(case.get("history") or [], key=lambda r: r.get("at", ""))
    horizon_days = float(case["horizon_minutes"]) / 1440.0

    # --- clean the reads: (t, players, rates) with players > 0 -----------
    reads = []
    seen_times = set()
    for read in history:
        try:
            t = parse_time(read["at"])
            players = read.get("players_tracked")
            rates = read.get("rates") or []
        except (KeyError, TypeError, ValueError):
            continue
        if players is None or players <= 0 or not rates:
            continue
        if t in seen_times:
            continue
        seen_times.add(t)
        reads.append((t, float(players), rates))
    if not reads:
        raise ValueError("no usable reads in history")

    t_last = reads[-1][0]
    n_last = reads[-1][1]  # players tracked in the last read
    last_rates = reads[-1][2]

    # --- tracked players: growth rate and projection ----------------------
    n_points = [(t, n) for t, n, _ in reads]
    g_n, _resid_n, se_n, _win_n = window_fit(n_points, t_last, WINDOW_HOURS)
    g_n = clamp(g_n, 0.0, MAX_GROWTH_PER_DAY)
    h_eff_n = decaying_horizon(horizon_days, TAU_PLAYERS_DAYS)
    n_target = n_last * math.exp(g_n * h_eff_n)

    # --- per trophy: earner growth, projection, spread --------------------
    medians = []
    sigmas = []
    projected_earners = []
    for i, trophy in enumerate(trophies):
        points = []
        last_rate = None
        for t, n, rates in reads:
            if i < len(rates) and rates[i] is not None:
                rate = float(rates[i])
                points.append((t, rate / 100.0 * n))
                last_rate = rate
        if last_rate is None:
            last_rate = 0.0
        if points:
            e_last = points[-1][1]
            g_e, _resid, se_e, _win_days = window_fit(points, t_last, WINDOW_HOURS)
        else:
            e_last = 0.0
            g_e, se_e = 0.0, 0.0
        g_e = clamp(g_e, 0.0, MAX_GROWTH_PER_DAY)
        h_eff_e = decaying_horizon(horizon_days, TAU_EARNERS_DAYS)
        e_target = (e_last + COUNT_SMOOTH) * math.exp(g_e * h_eff_e) - COUNT_SMOOTH
        e_target = max(e_target, e_last)
        projected_earners.append(e_target)

        median = 100.0 * e_target / n_target
        median = clamp(median, RATE_FLOOR, RATE_CEIL)
        medians.append(median)

        # Spread: a random walk in the log-odds, plus a term for how far
        # the median moved from the last read, plus the slope uncertainty
        # of both count fits carried forward over the effective horizon.
        # Rounding noise in the page's 2-decimal rates does not grow with
        # the horizon, so it enters only through the slope's standard
        # error, not as a raw residual term.
        move = abs(logit(median) - logit(last_rate))
        var = (
            (BASE_SIGMA * math.sqrt(horizon_days)) ** 2
            + (TREND_SPREAD * move) ** 2
            + (se_e * h_eff_e) ** 2
            + (se_n * h_eff_n) ** 2
        )
        sigmas.append(clamp(math.sqrt(var), SIGMA_FLOOR, SIGMA_CAP))

    # --- story chain: later steps cannot exceed earlier ones --------------
    for a, b in zip(story_chain_order(trophies), story_chain_order(trophies)[1:]):
        if projected_earners[b] > projected_earners[a]:
            projected_earners[b] = projected_earners[a]
            medians[b] = clamp(
                100.0 * projected_earners[b] / n_target, RATE_FLOOR, RATE_CEIL
            )

    # --- no trophy's rate can exceed the tracked base ---------------------
    for i in range(len(trophies)):
        if medians[i] > RATE_CEIL:
            medians[i] = RATE_CEIL

    rows = []
    for i, trophy in enumerate(trophies):
        lg = logit(medians[i])
        sigma = sigmas[i]
        qs = []
        prev = None
        for z in Z:
            q = round4(100.0 * logistic(lg + z * sigma))
            q = clamp(q, RATE_FLOOR, RATE_CEIL)
            if prev is not None and q < prev:
                q = prev
            qs.append(q)
            prev = q
        rows.append({"id": trophy["id"], "q": qs})
    return {"trophies": rows}


def fallback_forecast(case):
    """A valid forecast when the case cannot be read: the last read's
    rates with a wide spread, or 0.5% with a wide spread with no history."""
    trophies = case.get("trophies") or []
    last = None
    for read in reversed(case.get("history") or []):
        rates = read.get("rates") or []
        if rates:
            last = rates
            break
    rows = []
    for i, trophy in enumerate(trophies):
        rate = 0.5
        if last is not None and i < len(last) and last[i] is not None:
            rate = float(last[i])
        lg = logit(rate)
        qs = []
        prev = None
        for z in Z:
            q = round4(100.0 * logistic(lg + z * 1.5))
            q = clamp(q, RATE_FLOOR, RATE_CEIL)
            if prev is not None and q < prev:
                q = prev
            qs.append(q)
            prev = q
        rows.append({"id": trophy["id"], "q": qs})
    return {"trophies": rows}


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    for path in sorted(INPUT_DIR.glob("*.in")):
        try:
            case = json.loads(path.read_text(encoding="utf-8"))
            forecast = forecast_case(case)
        except Exception:
            try:
                case = json.loads(path.read_text(encoding="utf-8"))
                forecast = fallback_forecast(case)
            except Exception:
                continue
        (OUTPUT_DIR / (path.stem + ".out")).write_text(
            json.dumps(forecast, separators=(",", ":")) + "\n", encoding="utf-8"
        )


if __name__ == "__main__":
    main()
