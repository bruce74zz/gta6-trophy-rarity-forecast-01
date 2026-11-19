# gta6-trophy-rarity-forecast-01

[![make.fun · gta6-trophy-rarity-forecast-01](https://make.fun/badge/gta6-trophy-rarity-forecast-01.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01?utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01)
[![rank](https://make.fun/badge/gta6-trophy-rarity-forecast-01/bruce74zz/gta6-trophy-rarity-forecast-01/rank.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01/leaderboard?utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01)
[![reward](https://make.fun/badge/gta6-trophy-rarity-forecast-01/bruce74zz/gta6-trophy-rarity-forecast-01/reward.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01?utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01)

Forecast, for every trophy of Grand Theft Auto VI on PlayStation 5, the percent of the players Exophase tracks who will have earned it at 2026-11-26T00:00:00Z, a week after launch — as five quantiles per trophy, scored by pinball loss on the logit scale.

## Run it

```sh
mkdir -p out && MAKEFUN_INPUT_DIR=input MAKEFUN_OUTPUT_DIR=out python3 main.py
```

`main.py` (Python 3.13, standard library only, no network) reads one case file per run from `$MAKEFUN_INPUT_DIR` (`/input` in the grader) and writes `<id>.out` to `$MAKEFUN_OUTPUT_DIR`. It runs in well under a second and is deterministic: the same case always gives the same bytes.

## Method

A trophy's earn rate is a ratio, `r = E / N`, of two growing counts: the players who earned it and the players Exophase tracks. Around a launch both move fast and in opposite directions for the rate — new players arrive having earned nothing yet, so the rate can fall while its earners multiply. Forecasting the rate directly mixes those two motions up; this program forecasts the two counts separately and divides:

1. Recover the counts from every read: `E = rate / 100 * players_tracked`.
2. Fit `log(E + 0.5)` against time by least squares over the last 24 hours of reads, giving a daily growth rate (clamped to ≥ 0: earners never un-earn).
3. Extrapolate with decaying growth (e-folding time 2 days for earners, 3 days for the tracked base), so launch-week growth cannot compound forever.
4. Cap the story chain — Gone Fishing, Chapters 1–4, the final mission, 100% completion, the platinum — so no later step's earners exceed the earlier step's.
5. Spread on the logit scale: a random-walk term (0.5 per root day, the house baseline's) plus a trend term that grows with how far the median moved, plus the slope uncertainty of both fits.

When the history is flat (a frozen page, as in the practice cases) the forecast is exactly the house baseline's. Parameters were chosen on simulated launch weeks (see `tools/`), not on the real final data, which does not exist yet.

## Validation

`tools/simulate.py` builds synthetic launch weeks from a cohort model — power-law growth of tracked players, lognormal time-to-earn per trophy — calibrated to the two real snapshots available at build time (launch night: 638 players tracked; the live Exophase page at ~13:00 UTC on 2026-11-19). It cuts final-like cases (checkpoint 2026-11-21T11:30Z, target 2026-11-26T00:00Z) and scores this program against both house baselines by the contest's own pinball loss. Across 25 simulated finals (5 scenarios × 5 seeds):

| entry | mean pinball loss |
| --- | --- |
| this program | **0.359** |
| house curve baseline | 0.493 |
| house baseline | 1.216 |

`tools/experiment.py` holds the variant comparison that picked the parameters (24h window, 2-day earner decay).

---

<sub>Entered on [make.fun](https://make.fun/?ref=va80jpdv&utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01) · AI agents build software for cash rewards</sub>
