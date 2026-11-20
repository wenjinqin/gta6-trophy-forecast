"""GTA VI trophy-rarity forecast (gta6-trophy-rarity-forecast-01).

Model: earners / players-tracked decomposition plus a per-trophy logit trend.

Why this shape: an Exophase earn rate is a ratio, r = E / P, of two things that
move for different reasons.  E (how many tracked players have earned the
trophy) only grows, as players play; P (how many players Exophase tracks)
jumps whenever Exophase refreshes its counts, and every jump dilutes every
rate at once.  The house baseline holds r flat, so a refresh takes its whole
forecast outside its band.  This program instead forecasts E and P separately
and divides, and fits a trend to logit(r) as a second opinion:

  * P: weighted least squares of log P against x = ln(1 + hours since the
    release date), extrapolated to the target.  When the case's history shows
    P never moving at all (launch night: the page sat at 638 players for
    hours), a prior dilution is applied instead, because the first refresh is
    still ahead of the case: P is assumed to grow by (1 + 1.5 * horizon_hours),
    capped at 3x.  That prior only fires while no refresh has ever been seen;
    once the history contains real P movement the fitted trend takes over.
  * E: per trophy, weighted least squares of log(E + 0.5) against the same x,
    slope clamped at >= 0 (earners never un-earn) and shrunk toward the
    tier's median slope when the trophy's own history is thin.  While the
    history is flat, earners are instead grown with players at elasticity
    beta = (r / 100)^2: newcomers earn a near-universal trophy at once, so
    it barely dilutes, while a rare trophy dilutes almost in full.
  * Trend: per trophy, weighted least squares of logit(r) against x over the
    last 24 h of reads, slope shrunk toward the tier's median slope.
  * Median: logit-space blend of the E/P forecast and the trend forecast
    (trend weight 0 while the history is flat, 0.5 once it has real movement),
    then a small (15%) pull toward a tier prior when the history has real
    movement, so a steep early slope cannot explode over a 4.5-day horizon.
    Tier priors (week-one attainability): platinum 0.2%, gold 1%, silver
    4.5%, bronze 15%.
  * Spread (logit sd): random-walk term 0.5*sqrt(days) as in the baseline,
    plus the trend fit's residual sd scaled by the extrapolation leverage,
    plus a refresh term (0.9 while no refresh has ever been seen, 0.35 for
    horizons over two days, smaller otherwise).  The spread is floored at
    0.35 and capped at 1.6 logit units, aiming at honest ~90% coverage.

Python 3.13, standard library only, no network, fully deterministic (no
randomness anywhere).  Reads every *.in in $MAKEFUN_INPUT_DIR (else /input)
and writes <id>.out to $MAKEFUN_OUTPUT_DIR (else /output).
"""

import json
import math
import os
import statistics
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

INPUT_DIR = Path(os.environ.get("MAKEFUN_INPUT_DIR", "/input"))
OUTPUT_DIR = Path(os.environ.get("MAKEFUN_OUTPUT_DIR", "/output"))

LEVELS = [0.05, 0.25, 0.5, 0.75, 0.95]

# Week-one attainability prior per tier, as logits of the percent rate.
TIER_PRIOR_LOGIT = {
    "platinum": math.log(0.2 / 99.8),
    "gold": math.log(1.0 / 99.0),
    "silver": math.log(4.5 / 95.5),
    "bronze": math.log(15.0 / 85.0),
}
DEFAULT_PRIOR_LOGIT = TIER_PRIOR_LOGIT["bronze"]

_ND = statistics.NormalDist()


def parse_time(text):
    """ISO-8601 timestamp -> aware datetime (accepts a trailing Z)."""
    return datetime.fromisoformat(str(text).replace("Z", "+00:00"))


def clamp_pct(x):
    return min(99.99, max(0.01, x))


def logit_pct(pct):
    p = clamp_pct(pct)
    return math.log(p / (100.0 - p))


def sigmoid_logit(lg):
    if lg >= 0:
        return 100.0 / (1.0 + math.exp(-lg))
    e = math.exp(lg)
    return 100.0 * e / (1.0 + e)


def round4(x):
    """Round to 4 decimals, half up; whole numbers come back as ints."""
    scaled = x * 10000.0
    whole = math.floor(scaled)
    if scaled - whole >= 0.5:
        whole += 1
    value = whole / 10000.0
    return int(value) if value == int(value) else value


def wls(xs, ys, ws):
    """Weighted least squares line y ~ a + b*x. Returns (a, b, resid_sd, xbar, sxx).

    Returns None when the fit is degenerate (fewer than 2 points or no
    spread in x).
    """
    n = len(xs)
    if n < 2:
        return None
    sw = sum(ws)
    if sw <= 0:
        return None
    xbar = sum(w * x for w, x in zip(ws, xs)) / sw
    ybar = sum(w * y for w, y in zip(ws, ys)) / sw
    sxx = sum(w * (x - xbar) ** 2 for w, x in zip(ws, xs))
    if sxx <= 1e-12:
        return None
    sxy = sum(w * (x - xbar) * (y - ybar) for w, x, y in zip(ws, xs, ys))
    b = sxy / sxx
    a = ybar - b * xbar
    resid = sum(w * (y - (a + b * x)) ** 2 for w, x, y in zip(ws, xs, ys))
    resid_sd = math.sqrt(resid / sw) if sw > 0 else 0.0
    return (a, b, resid_sd, xbar, sxx)


def median(values):
    vals = sorted(values)
    m = len(vals)
    if m == 0:
        return 0.0
    if m % 2:
        return vals[m // 2]
    return 0.5 * (vals[m // 2 - 1] + vals[m // 2])


def forecast_case(case):
    trophies = case["trophies"]
    history = case["history"]
    horizon_days = case["horizon_minutes"] / 1440.0
    horizon_hours = case["horizon_minutes"] / 60.0

    # Time axis: hours since the release date at 00:00 UTC (x = ln(1 + hours)).
    try:
        release = parse_time(case["game"]["release_date"])
    except Exception:
        release = parse_time(history[0]["at"]) if history else datetime(
            2026, 11, 19, tzinfo=timezone.utc)
    if release.tzinfo is None:
        release = release.replace(tzinfo=timezone.utc)

    if not history:
        # No reads at all: tier priors, very wide spread.
        return {"trophies": [
            {"id": t["id"], "q": quantiles(TIER_PRIOR_LOGIT.get(t.get("tier"), DEFAULT_PRIOR_LOGIT), 1.6)}
            for t in trophies]}

    times = [parse_time(h["at"]) for h in history]
    t_last = times[-1]
    target = parse_time(case["target"])
    hours = [max(0.0, (t - release).total_seconds() / 3600.0) for t in times]
    xs = [math.log1p(h) for h in hours]
    x_star = math.log1p(max(0.0, (target - release).total_seconds() / 3600.0))
    # Recency weights: half-life 12 h, so the recent reads dominate the fit.
    weights = [0.5 ** (max(0.0, (t_last - t).total_seconds() / 3600.0) / 12.0)
               for t in times]
    # Last-24h window for the trend fit (indices into history).
    recent_idx = [j for j, t in enumerate(times)
                  if (t_last - t).total_seconds() <= 24 * 3600.0]
    if len(recent_idx) < 3:
        recent_idx = list(range(len(times)))

    players = [h.get("players_tracked") for h in history]
    players = [p if isinstance(p, (int, float)) and p > 0 else None for p in players]
    p_last = next((p for p in reversed(players) if p is not None), 638.0)
    p_seen_move = len({p for p in players if p is not None}) > 1

    # --- Forecast players tracked at the target. ---
    p_points = [(x, math.log(p), w) for x, p, w in zip(xs, players, weights)
                if p is not None]
    p_pred = p_last
    if len(p_points) >= 2:
        fit_p = wls([p[0] for p in p_points], [p[1] for p in p_points],
                    [p[2] for p in p_points])
        if fit_p is not None:
            a_p, b_p, _, _, _ = fit_p
            b_p = min(3.0, max(0.0, b_p))
            p_pred = math.exp(a_p + b_p * x_star)
    if not p_seen_move:
        # No refresh has ever been seen: the first one is still ahead.
        # Prior dilution, growing with the horizon, capped at 3x.
        p_pred = p_last * min(3.0, 1.0 + horizon_hours * 1.5)
    p_pred = min(max(p_pred, p_last), p_last * 20.0)

    levels = case.get("levels") or LEVELS
    zvals = [_ND.inv_cdf(min(0.999, max(0.001, lv))) for lv in levels]

    # --- First pass: raw per-trophy slopes, for tier-median shrinkage. ---
    trend_slopes = {}
    earn_slopes = {}
    for i, trophy in enumerate(trophies):
        series = []
        for j, h in enumerate(history):
            r = h["rates"][i] if i < len(h["rates"]) else None
            if r is None or players[j] is None:
                continue
            series.append((j, r, players[j]))
        if not series:
            continue
        tier = trophy.get("tier") or "bronze"
        lg = [logit_pct(r) for _, r, _ in series]
        xw = [xs[j] for j, _, _ in series]
        ww = [weights[j] for j, _, _ in series]
        fit = wls(xw, lg, ww)
        if fit is not None:
            trend_slopes.setdefault(tier, []).append(fit[1])
        ev = [math.log(max(r, 0.0) * p / 100.0 + 0.5) for _, r, p in series]
        fit_e = wls(xw, ev, ww)
        if fit_e is not None:
            earn_slopes.setdefault(tier, []).append(max(0.0, fit_e[1]))
    tier_trend_med = {t: median(v) for t, v in trend_slopes.items()}
    tier_earn_med = {t: median(v) for t, v in earn_slopes.items()}

    # --- Second pass: forecast each trophy. ---
    rows = []
    for i, trophy in enumerate(trophies):
        tier = trophy.get("tier") or "bronze"
        prior_lg = TIER_PRIOR_LOGIT.get(tier, DEFAULT_PRIOR_LOGIT)
        series = []
        for j, h in enumerate(history):
            r = h["rates"][i] if i < len(h["rates"]) else None
            if r is None or players[j] is None:
                continue
            series.append((j, r, players[j]))
        if not series:
            rows.append({"id": trophy["id"], "q": quantiles(prior_lg, 1.6)})
            continue
        last_rate = series[-1][1]
        last_lg = logit_pct(last_rate)
        distinct = len({r for _, r, _ in series})
        n = len(series)

        # Trend fit on the recent window.
        ridx = [k for k, (j, _, _) in enumerate(series) if j in set(recent_idx)]
        if len(ridx) < 2:
            ridx = list(range(n))
        tx = [xs[series[k][0]] for k in ridx]
        ty = [logit_pct(series[k][1]) for k in ridx]
        tw = [weights[series[k][0]] for k in ridx]
        fit_t = wls(tx, ty, tw)
        resid_sd = 0.30
        leverage = 1.0
        trend_lg = None
        if fit_t is not None:
            a_t, b_t, resid_sd, xbar, sxx = fit_t
            shrink = distinct / (distinct + 3.0)
            b_med = tier_trend_med.get(tier, 0.0)
            b_t = shrink * b_t + (1.0 - shrink) * b_med
            b_t = min(3.0, max(-3.0, b_t))
            # Re-anchor the line at the shrunk slope through the weighted
            # mean, so shrinkage moves the line, not just its tail.
            trend_lg = (sum(w * y for w, y in zip(tw, ty)) / sum(tw)
                        + b_t * (x_star - xbar))
            leverage = min(2.5, 1.0 + (x_star - xbar) ** 2 / max(sxx, 1e-9))
        if trend_lg is None:
            trend_lg = last_lg
        trend_lg = min(4.6, max(-9.0, trend_lg))

        # Earners forecast: log(E + 0.5) against x, slope >= 0, shrunk.
        ex = [xs[j] for j, _, _ in series]
        ey = [math.log(max(r, 0.0) * p / 100.0 + 0.5) for _, r, p in series]
        ew = [weights[j] for j, _, _ in series]
        fit_e = wls(ex, ey, ew)
        e_pred = max(last_rate, 0.0) * p_last / 100.0
        if fit_e is not None:
            a_e, b_e, _, xbar_e, _ = fit_e
            shrink_e = distinct / (distinct + 3.0)
            b_e = shrink_e * max(0.0, b_e) + (1.0 - shrink_e) * tier_earn_med.get(tier, 0.0)
            b_e = min(3.0, max(0.0, b_e))
            e_mean = sum(w * y for w, y in zip(ew, ey)) / sum(ew)
            e_pred = math.exp(e_mean + b_e * (x_star - xbar_e)) - 0.5
        e_pred = max(e_pred, 0.0)
        if distinct <= 1:
            # Flat history: the fit says earners never grew, but that is
            # only because no refresh has been seen yet.  When the refresh
            # lands, the new players it brings do earn trophies: a newcomer
            # earns an almost-universal trophy (rate near 100%) at once,
            # while almost no newcomer earns a rare one quickly.  So let
            # earners grow with players, with elasticity beta = (r / 100)^2:
            # beta -> 1 for a 100% trophy (earners track players, no
            # dilution) and -> 0 for a rare one (full dilution).  The square
            # keeps the correction where it belongs: only near-universal
            # trophies resist dilution, matching the launch-night refresh,
            # after which every rate but the 100% one fell.
            beta = (last_rate / 100.0) ** 2
            e_pred *= (p_pred / p_last) ** beta
        r_ep = 100.0 * e_pred / p_pred if p_pred > 0 else last_rate
        ep_lg = logit_pct(r_ep)

        # Blend: no trend weight while the history is flat.
        if distinct <= 1:
            w_trend = 0.0
        elif distinct == 2:
            w_trend = 0.25
        else:
            w_trend = 0.5
        med_lg = (1.0 - w_trend) * ep_lg + w_trend * trend_lg
        # Small pull toward the tier prior once real movement exists.
        if distinct >= 3:
            med_lg = 0.85 * med_lg + 0.15 * prior_lg
        med_lg = min(4.6, max(-9.0, med_lg))

        # Spread: random walk + fit residual (leveraged) + refresh risk.
        if p_seen_move:
            s_refresh = 0.35 if horizon_days > 2 else (0.25 if horizon_days > 0.5 else 0.15)
        else:
            s_refresh = 0.9
        s_fit = min(1.0, resid_sd * math.sqrt(leverage))
        spread = math.sqrt(0.25 * horizon_days + s_fit * s_fit + s_refresh * s_refresh)
        spread = min(1.6, max(0.35, spread))

        rows.append({"id": trophy["id"],
                     "q": quantiles(med_lg, spread, zvals)})
    return {"trophies": rows}


def quantiles(med_lg, spread, zvals=None):
    if zvals is None:
        zvals = [_ND.inv_cdf(lv) for lv in LEVELS]
    out = []
    prev = None
    for z in zvals:
        v = round4(clamp_pct(sigmoid_logit(med_lg + z * spread)))
        if prev is not None and v < prev:
            v = prev
        out.append(v)
        prev = v
    return out


def fallback_forecast(case):
    """Last-rate persistence with a wide spread; never crashes, never invalid."""
    try:
        trophies = case["trophies"]
        history = case["history"]
        last = history[-1]["rates"] if history else []
        rows = []
        for i, t in enumerate(trophies):
            r = last[i] if i < len(last) and last[i] is not None else 1.0
            rows.append({"id": t["id"],
                         "q": quantiles(logit_pct(r), 1.2)})
        return {"trophies": rows}
    except Exception:
        return {"trophies": [
            {"id": t["id"], "q": quantiles(TIER_PRIOR_LOGIT.get(
                t.get("tier"), DEFAULT_PRIOR_LOGIT), 1.6)}
            for t in case.get("trophies", [])]}


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
        (OUTPUT_DIR / f"{path.stem}.out").write_text(
            json.dumps(forecast, separators=(",", ":")) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
