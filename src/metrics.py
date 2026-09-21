import numpy as np
import pandas as pd


def _cagr(equity: pd.Series) -> float:
    years = (len(equity) - 1) / 252.0
    return (equity.iloc[-1] / equity.iloc[0]) ** (1.0 / years) - 1.0


def drawdown_episodes(equity: pd.Series) -> pd.DataFrame:
    """One row per drawdown: peak date, trough date, recovery date (NaT if not recovered), depth."""
    peak = equity.cummax()
    dd = equity / peak - 1.0
    under = dd < 0
    rows = []
    episode = (under != under.shift()).cumsum()[under]
    for _, days in dd[under].groupby(episode):
        start_pos = equity.index.get_loc(days.index[0])
        end_pos = equity.index.get_loc(days.index[-1])
        rows.append({
            "peak": equity.index[start_pos - 1],
            "trough": days.idxmin(),
            "recovery": equity.index[end_pos + 1] if end_pos + 1 < len(equity) else pd.NaT,
            "depth": days.min(),
        })
    return pd.DataFrame(rows, columns=["peak", "trough", "recovery", "depth"])


def compute_metrics(equity: pd.Series, rf_daily: pd.Series) -> dict:
    """Full metric set. rf_daily is the same per-day cash return series used by the cash leg."""
    ret = equity.pct_change().dropna()
    excess = ret - rf_daily.reindex(ret.index)
    years = len(ret) / 252.0
    cagr = _cagr(equity)
    dd = equity / equity.cummax() - 1.0
    eps = drawdown_episodes(equity)
    end = equity.index[-1]
    underwater_days = ((eps["recovery"].fillna(end) - eps["peak"]).dt.days) if len(eps) else pd.Series([0])
    recovery_days = ((eps["recovery"].fillna(end) - eps["trough"]).dt.days) if len(eps) else pd.Series([0])

    yearly = equity.resample("YE").last()
    yearly = pd.concat([equity.iloc[:1], yearly]).pct_change().dropna()
    monthly = equity.resample("ME").last()
    monthly = pd.concat([equity.iloc[:1], monthly]).pct_change().dropna()
    downside = np.sqrt((np.minimum(excess, 0.0) ** 2).mean())
    max_dd = dd.min()

    return {
        "CAGR": cagr,
        "Total Return": equity.iloc[-1] / equity.iloc[0] - 1.0,
        "Ann. Volatility": ret.std() * np.sqrt(252),
        "Sharpe": excess.mean() / excess.std() * np.sqrt(252),
        "Sortino": excess.mean() / downside * np.sqrt(252),
        "Calmar": cagr / abs(max_dd) if max_dd < 0 else np.nan,
        "Max Drawdown": max_dd,
        "Avg Drawdown (episode troughs)": eps["depth"].mean() if len(eps) else 0.0,
        "Drawdowns > 5%": int((eps["depth"] < -0.05).sum()),
        "Longest Drawdown (cal. days)": int(underwater_days.max()),
        "Longest Trough->Recovery (cal. days)": int(recovery_days.max()),
        "Unrecovered at End": bool(len(eps) and pd.isna(eps["recovery"].iloc[-1])),
        "% Days in Drawdown": (dd < 0).mean(),
        "Worst Calendar Year": yearly.min(),
        "Worst Year": int(yearly.idxmin().year),
        "Worst Rolling 12M": (equity / equity.shift(252) - 1.0).min(),
        "Worst Month": monthly.min(),
        "Worst Month Date": monthly.idxmin().strftime("%Y-%m"),
        "Skew (daily)": ret.skew(),
        "Excess Kurtosis (daily)": ret.kurt(),
        "Years": years,
    }


def activity_metrics(sim: pd.DataFrame, trades: list, zero_cost_cagr: float, cagr: float) -> dict:
    years = (len(sim) - 1) / 252.0
    h = sim["holding"].iloc[1:]
    spells = h.groupby((h != h.shift()).cumsum()).agg(["first", "size"])
    out = {
        "Position Switches": sum(tr["asset_from"] != tr["asset_to"] for tr in trades),
        "Trade Events (incl. resizes)": len(trades),
        "Annual Turnover": sum(tr["turnover"] for tr in trades) / years,
        "Avg Risky Exposure": sim["exposure"].iloc[1:].mean(),
        "Total Cost ($)": sim["cost"].sum(),
        "Cost Drag (%/yr)": zero_cost_cagr - cagr,
    }
    for a in ["SPY", "QQQ", "CASH"]:
        out[f"Time in {a}"] = (h == a).mean()
        s = spells[spells["first"] == a]["size"]
        out[f"Avg Holding {a} (trading days)"] = s.mean() if len(s) else 0.0
    return out


def levered_equity(equity: pd.Series, rf_annual_pct: pd.Series, leverage: float,
                   spread: float = 1.5) -> pd.Series:
    """Daily-rebalanced leverage on the strategy; borrowed (L-1) pays T-bill + spread (annual %)."""
    ret = equity.pct_change().dropna()
    borrow = (1 + (rf_annual_pct.shift(1).reindex(ret.index).bfill() + spread) / 100) ** (1 / 252) - 1
    lev_ret = leverage * ret - (leverage - 1.0) * borrow
    return pd.concat([equity.iloc[:1], equity.iloc[0] * (1 + lev_ret).cumprod()])


def leverage_ceiling(equity: pd.Series, rf_annual_pct: pd.Series, rf_daily: pd.Series,
                     dd_limit: float = 0.22, target_cagr: float = 0.30) -> dict:
    ret_idx = equity.index[1:]
    avg_rf = (1 + rf_daily.reindex(ret_idx)).prod() ** (252 / len(ret_idx)) - 1
    cagr = _cagr(equity)
    max_dd = (equity / equity.cummax() - 1.0).min()

    l_max = dd_limit / abs(max_dd)
    eq_l = levered_equity(equity, rf_annual_pct, l_max)

    def cagr_at(lev):
        e = levered_equity(equity, rf_annual_pct, lev)
        return _cagr(e) if (e > 0).all() else -1.0

    # CAGR vs leverage is concave (volatility drag); scan, then bisect if 30% is reachable.
    grid = np.round(np.arange(1.0, 20.01, 0.05), 2)
    cagrs = [cagr_at(g) for g in grid]
    best = int(np.argmax(cagrs))
    l_target = None
    if cagrs[best] >= target_cagr:
        lo, hi = 1.0, float(grid[best])
        if cagr_at(lo) >= target_cagr:
            l_target = lo
        else:
            for _ in range(60):
                mid = (lo + hi) / 2
                lo, hi = (mid, hi) if cagr_at(mid) < target_cagr else (lo, mid)
            l_target = hi

    out = {
        "Avg Risk-Free Rate (geometric, ann.)": avg_rf,
        "Strategy CAGR": cagr,
        "Excess Return over RF (CAGR - avg RF)": cagr - avg_rf,
        "Max Drawdown (1x)": max_dd,
        "Max Leverage for 22% DD (22/|MDD|)": l_max,
        "CAGR at Max Leverage (after T-bill+1.5% financing)": _cagr(eq_l),
        "Realised MDD at Max Leverage (daily-rebalanced sim)": (eq_l / eq_l.cummax() - 1).min(),
        "Peak Achievable CAGR (leverage 1-20x scan)": cagrs[best],
        "Leverage at Peak CAGR": float(grid[best]),
    }
    if l_target is None:
        out["Leverage for 30% CAGR"] = "NOT ATTAINABLE"
        out["Implied DD at that leverage"] = "n/a"
    else:
        eq_t = levered_equity(equity, rf_annual_pct, l_target)
        out["Leverage for 30% CAGR"] = l_target
        out["Implied DD, linear (L x |MDD|)"] = -l_target * abs(max_dd)
        out["Implied DD, simulated"] = (eq_t / eq_t.cummax() - 1).min()
    return out
