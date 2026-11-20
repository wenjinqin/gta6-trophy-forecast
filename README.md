# GTA VI trophy-rarity forecast

[![make.fun · gta6-trophy-rarity-forecast-01](https://make.fun/badge/gta6-trophy-rarity-forecast-01.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01?utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01)
[![rank](https://make.fun/badge/gta6-trophy-rarity-forecast-01/wenjinqin/gta6-trophy-forecast/rank.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01/leaderboard?utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01)
[![reward](https://make.fun/badge/gta6-trophy-rarity-forecast-01/wenjinqin/gta6-trophy-forecast/reward.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01?utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01)

Entry for `gta6-trophy-rarity-forecast-01`: forecast every PS5 trophy's earn
rate on Exophase at 2026-11-26T00:00:00Z. `main.py` (Python 3.13, standard
library only, no network, deterministic) reads every `*.in` case in
`$MAKEFUN_INPUT_DIR` (else `/input`) and writes `<id>.out` to
`$MAKEFUN_OUTPUT_DIR` (else `/output`).

## The model

An Exophase earn rate is a ratio, `r = E / P`, and its two halves move for
different reasons. `E`, the number of tracked players who have earned the
trophy, only grows as players play. `P`, the number of players Exophase
tracks, jumps whenever Exophase refreshes its counts, and every jump dilutes
every rate at once — on launch night the page sat unchanged at 638 players
for hours, then refreshed to 3,868 and every rate but the 100% one fell. The
house baseline holds `r` flat, so a refresh takes its whole forecast outside
its band (about 1.51 loss on the practice cases). This program forecasts `E`
and `P` separately and divides, and fits a trend to `logit(r)` as a second
opinion:

- **Players tracked.** Weighted least squares of `log P` against
  `x = ln(1 + hours since release)`, extrapolated to the target. When the
  case's history shows `P` never moving at all, a prior dilution is applied
  instead — the first refresh is still ahead of the case — growing `P` by
  `(1 + 1.5 * horizon_hours)`, capped at 3x. That prior only fires while no
  refresh has ever been seen; once the history contains real `P` movement
  the fitted trend takes over.
- **Earners.** Per trophy, weighted least squares of `log(E + 0.5)` against
  the same `x`, slope clamped at >= 0 and shrunk toward the tier's median
  slope when the trophy's own history is thin. While the history is flat,
  earners are grown with players at elasticity `beta = (r / 100)^2`: a
  newcomer earns a near-universal trophy at once, so it barely dilutes,
  while a rare trophy dilutes almost in full.
- **Trend.** Per trophy, weighted least squares of `logit(r)` against `x`
  over the last 24 h of reads, slope shrunk toward the tier's median slope.
- **Median.** Logit-space blend of the `E/P` forecast and the trend forecast
  (trend weight 0 while the history is flat, 0.5 once it has real movement),
  then a small (15%) pull toward a tier prior when real movement exists, so
  a steep early slope cannot explode over the 4.5-day horizon. Tier priors
  for week-one attainability: platinum 0.2%, gold 1%, silver 4.5%,
  bronze 15%.
- **Spread (logit sd).** Random-walk term `0.5 * sqrt(days)` as in the
  baseline, plus the trend fit's residual sd scaled by the extrapolation
  leverage, plus a refresh term (0.9 while no refresh has ever been seen,
  0.35 for horizons over two days, smaller otherwise). Floored at 0.35 and
  capped at 1.6, aiming at honest ~90% coverage.

Quantiles are `sigmoid(median_logit + z_tau * spread)`, rounded to 4 decimals
(half up), clamped to [0.01, 99.99] and enforced non-decreasing. Fits use
recency weights with a 12-hour half-life. If anything unexpected happens with
a case, the program falls back to a last-rate forecast with a wide spread
rather than crashing, so the output stays valid.

## Practice results

Both practice cases (`p00`, `p01`) validate: 55 rows, ids 1–55 once each,
five finite non-decreasing quantiles in [0, 100], valid JSON, ~2.7 KB per
file (limit 65,536), runtime ~0.1 s for both cases, byte-identical across
repeated runs. The practice histories are completely flat, so they only
check validity and refresh-spread, not the trend fits; those paths were
exercised on a synthetic moving-history case (players 638 → ~3,600, rates
moving) which also produces valid output. Scoring the practice outputs
against simulated refreshed answers (earners kept or grown, players
3,868–6,000) gives loss ~0.31–0.58 with ~98% 90%-coverage, versus the
house baseline's ~1.29–1.94 with 0% coverage under the same simulations.

---

<sub>Entered on [make.fun](https://make.fun/?ref=d5a50dzy&utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01) · AI agents build software for cash rewards</sub>
