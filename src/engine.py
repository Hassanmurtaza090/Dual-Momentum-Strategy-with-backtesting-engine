import numpy as np
import pandas as pd

RISKY = ["SPY", "QQQ"]
CASH = "CASH"
WARMUP = 201


def compute_indicators(prices: pd.DataFrame) -> dict:
    """
    Strategy indicators. `prices` = adjusted closes for SPY, QQQ and the synthetic CASH index.
    Row T uses only rows <= T (verified by the truncation audit in main.py).
    """
    log_returns = np.log(prices / prices.shift(1))

    # Momentum: 60% 3M (63D) + 40% 6M (126D)
    composite_mom = 0.60 * (prices / prices.shift(63) - 1.0) + 0.40 * (prices / prices.shift(126) - 1.0)
    vol63 = log_returns.rolling(window=63).std() * np.sqrt(252)
    vol20 = log_returns.rolling(window=20).std() * np.sqrt(252)

    return pd.DataFrame({
        "spy_close": prices["SPY"],
        "spy_sma200": prices["SPY"].rolling(window=200).mean(),
        "mom_spy": composite_mom["SPY"],
        "mom_cash": composite_mom[CASH],
        "ram_spy": (composite_mom / (vol63 + 1e-9))["SPY"],
        "ram_qqq": (composite_mom / (vol63 + 1e-9))["QQQ"],
        "vol20_spy": vol20["SPY"],
        "vol20_qqq": vol20["QQQ"],
    })


def decide_schedule(ind: pd.DataFrame, target_vol: float = 0.12, rebalance_freq: int = 21,
                    start_idx: int = WARMUP) -> dict:
    """
    Unchanged strategy rules. Signal on close of day s = t-1, order executes at open of day t.
    Returns {exec_idx: (signal_idx, asset, risky_weight)}; exec_idx == len(ind) is the pending
    order for the next session (not simulated).
    """
    x = {c: ind[c].tolist() for c in ind.columns}
    n = len(ind)
    schedule = {}
    current_asset = CASH
    days_held = 0

    for t in range(start_idx, n + 1):
        s = t - 1
        days_held += 1

        # Macro trend check on signal day close
        regime_bullish = (x["spy_close"][s] > x["spy_sma200"][s]) or (x["mom_spy"][s] > x["mom_cash"][s])
        should_rebalance = (days_held >= rebalance_freq) or (not regime_bullish and current_asset != CASH)
        if not should_rebalance:
            continue

        days_held = 0
        if not regime_bullish:
            target_asset, target_w = CASH, 0.0
        else:
            target_asset = "QQQ" if x["ram_qqq"][s] > x["ram_spy"][s] else "SPY"
            inst_vol = x["vol20_qqq" if target_asset == "QQQ" else "vol20_spy"][s]
            target_w = min(1.0, target_vol / inst_vol) if inst_vol > 0 else 1.0
        schedule[t] = (s, target_asset, target_w)
        current_asset = target_asset

    return schedule


def simulate(panels: dict, cash_ret: pd.Series, schedule: dict, start_idx: int,
             initial_capital: float = 100000.0, slippage_bps: float = 2.0,
             commission_per_share: float = 0.005, min_commission: float = 1.0) -> tuple[pd.DataFrame, list]:
    """
    Dollar-based simulation, one risky asset + synthetic cash, weights drift between trades.
    Day t: close(t-1)->open(t) at OLD holdings, trade at open(t), open(t)->close(t) at NEW holdings.
    Costs: per-share commission (min per order) + slippage bps on traded notional; cash leg is free.
    """
    col = {a: i for i, a in enumerate(RISKY)}
    ac = panels["adj_close"][RISKY].to_numpy().tolist()
    ao = panels["adj_open"][RISKY].to_numpy().tolist()
    op = panels["open"][RISKY].to_numpy().tolist()
    rf = cash_ret.tolist()
    n = len(ac)
    slip = slippage_bps / 1e4

    asset, v_risky, v_cash = None, 0.0, initial_capital
    equity, holding, exposure, cost_paid, traded = [initial_capital], [CASH], [0.0], [0.0], [0.0]
    trades = []

    for t in range(start_idx, n):
        # (a) overnight gap at old holdings
        if asset is not None:
            v_risky *= ao[t][col[asset]] / ac[t - 1][col[asset]]
        v_cash *= 1.0 + rf[t]

        cost = notional_total = 0.0
        order = schedule.get(t)
        if order is not None:
            s, new_asset, w = order
            new_asset = None if new_asset == CASH else new_asset
            total = v_risky + v_cash
            target = w * total if new_asset is not None else 0.0

            orders = []
            if asset is not None and asset == new_asset:
                if target != v_risky:
                    orders.append((asset, abs(target - v_risky)))
            else:
                if asset is not None:
                    orders.append((asset, v_risky))
                if new_asset is not None and target > 0:
                    orders.append((new_asset, target))

            for a, notional in orders:
                shares = notional / op[t][col[a]]
                cost += max(min_commission, commission_per_share * shares) + slip * notional
                notional_total += notional

            if orders:
                px_asset = new_asset if new_asset is not None else asset
                trades.append({
                    "signal_idx": s, "exec_idx": t,
                    "asset_from": asset or CASH, "asset_to": new_asset or CASH, "weight_to": w,
                    "traded_asset": px_asset, "notional": notional_total, "cost": cost,
                    "turnover": notional_total / total,
                })
            v_cash = total - target - cost
            v_risky = target
            asset = new_asset

        # (b) intraday at new holdings
        if asset is not None:
            v_risky *= ac[t][col[asset]] / ao[t][col[asset]]

        eq = v_risky + v_cash
        if not np.isfinite(eq) or eq <= 0:
            raise ValueError(f"Impossible equity {eq} at index {t}")
        equity.append(eq)
        holding.append(asset or CASH)
        exposure.append(v_risky / eq)
        cost_paid.append(cost)
        traded.append(notional_total)

    idx = panels["adj_close"].index[start_idx - 1:]
    out = pd.DataFrame({"equity": equity, "holding": holding, "exposure": exposure,
                        "cost": cost_paid, "traded": traded}, index=idx)
    return out, trades


def buy_and_hold_schedule(asset: str, start_idx: int) -> dict:
    return {start_idx: (start_idx - 1, asset, 1.0)}


def monthly_60_40_schedule(index: pd.DatetimeIndex, start_idx: int) -> dict:
    """Rebalance to 60% SPY / 40% cash at the first open of each month (signal: prior month-end close)."""
    months = index.to_period("M")
    firsts = [t for t in range(start_idx, len(index)) if t == start_idx or months[t] != months[t - 1]]
    return {t: (t - 1, "SPY", 0.6) for t in firsts}


def random_schedule(rng: np.random.Generator, n: int, start_idx: int, n_switches: int,
                    avg_exposure: float) -> dict:
    """
    Random control: n_switches switch days drawn uniformly, asset drawn at random from the two assets
    not currently held. Risky weight k is set so average exposure matches the real strategy.
    """
    days = np.sort(rng.choice(np.arange(start_idx, n), size=n_switches, replace=False))
    options = {CASH: ["SPY", "QQQ"], "SPY": ["QQQ", CASH], "QQQ": ["SPY", CASH]}
    current, assets = CASH, []
    for _ in days:
        current = options[current][rng.integers(2)]
        assets.append(current)

    # fraction of simulated days spent in a risky asset
    bounds = list(days) + [n]
    risky_days = sum(bounds[i + 1] - bounds[i] for i, a in enumerate(assets) if a != CASH)
    frac = risky_days / (n - start_idx)
    k = min(1.0, avg_exposure / frac) if frac > 0 else 0.0
    return {int(d): (int(d) - 1, a, k if a != CASH else 0.0) for d, a in zip(days, assets)}
