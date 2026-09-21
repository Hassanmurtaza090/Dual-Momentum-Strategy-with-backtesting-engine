import json
import os
import sys
import time
from datetime import datetime

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.data_loader import build_price_panels, fetch_fred_rf, fetch_sharadar_etf_data
from src.engine import (CASH, WARMUP, buy_and_hold_schedule, compute_indicators, decide_schedule,
                        monthly_60_40_schedule, random_schedule, simulate)
from src.metrics import activity_metrics, compute_metrics, leverage_ceiling

PARAMS = {
    "initial_capital": 100000.0, "target_vol": 0.12, "rebalance_freq": 21, "warmup_bars": WARMUP,
    "sma_regime": 200, "mom_weights": "0.6*63D + 0.4*126D", "risk_adj_vol": 63, "sizing_vol": 20,
    "assets": "SPY, QQQ, synthetic cash (FRED DTB3)",
    "commission_per_share": 0.005, "min_commission": 1.0, "slippage_bps": 2.0,
    "margin_spread": 1.5, "random_paths": 1000, "random_seed": 42,
}
SAMPLES = {"A": "2007-06-01", "B": None}  # A reproduces the original data start; B = all data (QQQ inception)
SENSITIVITY = [1, 2, 3, 5]
MAX_BIL_TRACKING_DIFF = 0.004
PCT_KEYS = {"CAGR", "Total Return", "Ann. Volatility", "Max Drawdown", "Avg Drawdown (episode troughs)",
            "% Days in Drawdown", "Worst Calendar Year", "Worst Rolling 12M", "Worst Month",
            "Annual Turnover", "Avg Risky Exposure", "Cost Drag (%/yr)",
            "Time in SPY", "Time in QQQ", "Time in CASH"}


def banner(title):
    print("\n" + "=" * 78 + f"\n{title}\n" + "=" * 78)


def fmt(key, v):
    if isinstance(v, (float, np.floating)):
        pct_words = ["CAGR", "Rate", "Return", "Drawdown", "MDD", "Implied DD"]
        if key in PCT_KEYS or (any(w in key for w in pct_words) and not key.startswith(("Max Leverage", "Leverage"))):
            return f"{v * 100:.2f}%"
        return f"{v:,.2f}"
    return str(v)


def print_table(rows: dict):
    """rows: {column_name: metrics_dict} -> aligned text table."""
    cols = list(rows)
    keys = list(dict.fromkeys(k for r in rows.values() for k in r))
    df = pd.DataFrame({c: [fmt(k, rows[c].get(k, "")) for k in keys] for c in cols}, index=keys)
    print(df.to_string())


def sim_cagr(eq):
    return (eq.iloc[-1] / eq.iloc[0]) ** (252 / (len(eq) - 1)) - 1


def fast_sharpe(eq, rf):
    ex = eq.pct_change().dropna() - rf.reindex(eq.index[1:])
    return ex.mean() / ex.std() * np.sqrt(252)


def main():
    t0 = time.time()
    run_id = datetime.now().strftime("%Y-%m-%d_%H%M%S")
    out_dir = os.path.join("results", run_id)
    os.makedirs(out_dir, exist_ok=False)
    cap = PARAMS["initial_capital"]

    # ------------------------------------------------------------------ FIX 1
    banner("FIX 1: next-open execution with split gap/intraday accounting")
    raw = fetch_sharadar_etf_data(["SPY", "QQQ", "BIL"], start_date="1999-01-01")
    panels = build_price_panels(raw, ["SPY", "QQQ"])
    idx = panels["adj_close"].index
    print(f"Data: {idx[0].date()} -> {idx[-1].date()}, {len(idx)} common SPY/QQQ sessions")
    print("adj_open = open * closeadj / close (same row); signal on close T, fill at open T+1;")
    print("day T+1 return = gap close(T)->open(T+1) at OLD holdings + open(T+1)->close(T+1) at NEW holdings.")
    print("Weights drift between trades (no free daily rebalancing).")
    print(f"[FIX 1 APPLIED] open fallbacks so far: {len(panels['open_fallbacks'])} (final count in FIX 7)")

    # ------------------------------------------------------------------ FIX 2
    banner("FIX 2: FRED DTB3 risk-free rate")
    rf_annual, rf_daily_quote, rf_gap_log = fetch_fred_rf(idx)
    # The rate quoted at close T-1 is what cash earns over day T.
    cash_ret = rf_daily_quote.shift(1).fillna(rf_daily_quote.iloc[0])
    print(f"DTB3 on trading days: min {rf_annual.min():.2f}%  max {rf_annual.max():.2f}%  mean {rf_annual.mean():.2f}%")
    print(f"Forward-fill runs > 5 business days: {len(rf_gap_log)} {rf_gap_log if rf_gap_log else ''}")
    print("daily_rf = (1 + annual/100)**(1/252) - 1; same series used for cash leg, Sharpe and Sortino.")
    print("[FIX 2 APPLIED]")

    # ------------------------------------------------------------------ FIX 3
    banner("FIX 3: synthetic cash leg + sanity check vs BIL")
    bil = build_price_panels(raw, ["BIL"])
    _, bil_rf_quote, _ = fetch_fred_rf(bil["adj_close"].index)
    bil_cash_ret = bil_rf_quote.shift(1).iloc[1:]
    bil_ret = bil["adj_close"]["BIL"].pct_change().iloc[1:]
    yrs = len(bil_ret) / 252
    syn_cagr = (1 + bil_cash_ret).prod() ** (1 / yrs) - 1
    bil_cagr = (1 + bil_ret).prod() ** (1 / yrs) - 1
    diff = syn_cagr - bil_cagr
    te = (bil_cash_ret - bil_ret).std() * np.sqrt(252)
    print(f"BIL window {bil_ret.index[0].date()} -> {bil_ret.index[-1].date()} ({yrs:.1f} yrs)")
    print(f"Synthetic cash CAGR {syn_cagr:.4%} | BIL total-return CAGR {bil_cagr:.4%}")
    print(f"Annualised tracking difference (synthetic - BIL): {diff:.4%}  | tracking error {te:.4%}")
    if abs(diff) > MAX_BIL_TRACKING_DIFF:
        print(f"STOP: tracking difference exceeds {MAX_BIL_TRACKING_DIFF:.2%} per year.")
        sys.exit(1)
    print("Regime check's 'momentum vs BIL' now compares against the synthetic cash index.")
    print("[FIX 3 APPLIED] tracking difference within 0.40%/yr; running Sample A and Sample B")

    # ------------------------------------------------------------------ run samples
    rng = np.random.default_rng(PARAMS["random_seed"])
    res = {}
    for name, start in SAMPLES.items():
        sl = slice(start, None)
        p = {k: panels[k].loc[sl] for k in ["adj_close", "adj_open", "open", "close"]}
        cr = cash_ret.loc[sl]
        cash_index = (1 + cr).cumprod()
        prices = p["adj_close"].assign(**{CASH: cash_index})
        ind = compute_indicators(prices)
        sched_full = decide_schedule(ind, PARAMS["target_vol"], PARAMS["rebalance_freq"], WARMUP)
        n = len(prices)
        pending = sched_full.get(n)
        sched = {t: v for t, v in sched_full.items() if t < n}

        sim, trades = simulate(p, cr, sched, WARMUP, cap)
        zero, _ = simulate(p, cr, sched, WARMUP, cap, slippage_bps=0, commission_per_share=0, min_commission=0)
        sens = {m: sim_cagr(simulate(p, cr, sched, WARMUP, cap, slippage_bps=2.0 * m)[0]["equity"])
                for m in SENSITIVITY}

        eq = sim["equity"]
        rf_s = cr.reindex(eq.index)
        bench = {
            "SPY B&H": simulate(p, cr, buy_and_hold_schedule("SPY", WARMUP), WARMUP, cap)[0]["equity"],
            "QQQ B&H": simulate(p, cr, buy_and_hold_schedule("QQQ", WARMUP), WARMUP, cap)[0]["equity"],
            "60/40 SPY/Cash": simulate(p, cr, monthly_60_40_schedule(prices.index, WARMUP), WARMUP, cap)[0]["equity"],
        }

        metrics = compute_metrics(eq, rf_s)
        act = activity_metrics(sim, trades, sim_cagr(zero["equity"]), metrics["CAGR"])

        # random control
        rand_eq, rand_cagr, rand_sharpe, rand_expo = [], [], [], []
        for _ in range(PARAMS["random_paths"]):
            rs = random_schedule(rng, n, WARMUP, act["Position Switches"], act["Avg Risky Exposure"])
            rsim, _ = simulate(p, cr, rs, WARMUP, cap)
            rand_eq.append(rsim["equity"].to_numpy())
            rand_cagr.append(sim_cagr(rsim["equity"]))
            rand_sharpe.append(fast_sharpe(rsim["equity"], rf_s))
            rand_expo.append(rsim["exposure"].iloc[1:].mean())
        rand_eq = np.array(rand_eq)
        med_path = int(np.argsort(rand_cagr)[len(rand_cagr) // 2])
        bench["Random (median path)"] = pd.Series(rand_eq[med_path], index=eq.index)

        res[name] = dict(p=p, cr=cr, prices=prices, ind=ind, sched_full=sched_full, pending=pending, sim=sim,
                         trades=trades, sens=sens, eq=eq, rf=rf_s, bench=bench, metrics=metrics, act=act,
                         rand_eq=rand_eq, rand_cagr=np.array(rand_cagr), rand_sharpe=np.array(rand_sharpe),
                         rand_expo=np.array(rand_expo))
        print(f"Sample {name}: results {eq.index[1].date()} -> {eq.index[-1].date()} "
              f"({len(eq) - 1} sessions, {metrics['Years']:.1f} yrs)")

    # ------------------------------------------------------------------ FIX 4
    banner("FIX 4: cost model")
    print("PREVIOUS RUN assumed: NOT zero cost. 3 bps x 'turnover' on each rebalance, where a switch counted")
    print("  old_weight + new_weight (overstated when coming from BIL) and a resize counted |dw|. Weights were")
    print("  held constant every day, i.e. an untaxed implicit daily rebalance between ETF and BIL.")
    print("  Rebalance: daily check; scheduled every 21 trading days, plus immediate exit when regime turns off.")
    print("NEW MODEL: $0.005/share (min $1/order, share count from raw open) + 2 bps slippage on traded")
    print("  notional, one way; only traded amount is charged; synthetic cash leg trades free; no cost if no trade.")
    print("Rebalance frequency used (unchanged): daily regime check, 21-trading-day scheduled rebalance,")
    print("  immediate move to cash when regime turns bearish.")
    sens_df = pd.DataFrame({f"Sample {k}": {f"{m}x slippage ({2 * m} bps)": f"{v * 100:.2f}%"
                                             for m, v in r["sens"].items()} for k, r in res.items()})
    print("\nCAGR cost sensitivity (commission unchanged):")
    print(sens_df.to_string())
    for k, r in res.items():
        print(f"Sample {k}: position switches {r['act']['Position Switches']}, "
              f"trade events incl. resizes {r['act']['Trade Events (incl. resizes)']}")
    print("[FIX 4 APPLIED]")

    # ------------------------------------------------------------------ FIX 5
    banner("FIX 5: full metric set (strategy)")
    print_table({f"Sample {k}": {**r["metrics"], **r["act"]} for k, r in res.items()})
    cal_tables = {}
    for k, r in res.items():
        yr = lambda e: pd.concat([e.iloc[:1], e.resample("YE").last()]).pct_change().dropna()
        cal = pd.DataFrame({"Strategy": yr(r["eq"]), "SPY": yr(r["bench"]["SPY B&H"]),
                            "QQQ": yr(r["bench"]["QQQ B&H"])})
        cal.index = cal.index.year
        cal_tables[k] = cal
        print(f"\nCalendar-year returns, Sample {k} (first year partial, starts {r['eq'].index[0].date()}):")
        print((cal * 100).round(2).astype(str).add("%").to_string())

    for k, r in res.items():
        series = {"Strategy": r["eq"], **{b: e for b, e in r["bench"].items()}}
        colors = {"Strategy": "#004488", "SPY B&H": "#888888", "QQQ B&H": "#BB5566",
                  "60/40 SPY/Cash": "#DDAA33", "Random (median path)": "#228833"}

        fig, ax = plt.subplots(figsize=(12, 6))
        lo, hi = np.percentile(r["rand_eq"], [5, 95], axis=0)
        ax.fill_between(r["eq"].index, lo, hi, color="#228833", alpha=0.12, label="Random 5-95% band")
        for s, e in series.items():
            ax.plot(e.index, e, label=s, color=colors[s], lw=1.8 if s == "Strategy" else 1.0)
        ax.set_yscale("log"); ax.set_title(f"Equity (log), Sample {k}"); ax.set_ylabel("$")
        ax.grid(alpha=0.3); ax.legend(); fig.tight_layout()
        fig.savefig(os.path.join(out_dir, f"equity_log_{k}.png"), dpi=150); plt.close(fig)

        fig, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
        for ax, win, lab in [(axes[0], 252, "1-year"), (axes[1], 756, "3-year")]:
            for s, e in series.items():
                ax.plot(e.index, (e / e.shift(win)) ** (252 / win) - 1, label=s, color=colors[s],
                        lw=1.6 if s == "Strategy" else 0.9)
            ax.axhline(0, color="black", lw=0.6); ax.set_title(f"Rolling {lab} CAGR, Sample {k}")
            ax.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0)); ax.grid(alpha=0.3)
        axes[0].legend(fontsize=8); fig.tight_layout()
        fig.savefig(os.path.join(out_dir, f"rolling_cagr_{k}.png"), dpi=150); plt.close(fig)

        fig, ax = plt.subplots(figsize=(12, 5))
        for s, e in series.items():
            ax.plot(e.index, e / e.cummax() - 1, label=s, color=colors[s], lw=1.6 if s == "Strategy" else 0.9)
        ax.set_title(f"Underwater (drawdown), Sample {k}")
        ax.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0))
        ax.grid(alpha=0.3); ax.legend(fontsize=8); fig.tight_layout()
        fig.savefig(os.path.join(out_dir, f"underwater_{k}.png"), dpi=150); plt.close(fig)
    print(f"\nCharts saved to {out_dir}/ (equity_log_*, rolling_cagr_*, underwater_*)")
    print("[FIX 5 APPLIED]")

    # ------------------------------------------------------------------ FIX 6
    banner("FIX 6: leverage ceiling (22% max-drawdown budget, margin = T-bill + 1.5%)")
    lev = {f"Sample {k}": leverage_ceiling(r["eq"], rf_annual.reindex(r["eq"].index), r["rf"])
           for k, r in res.items()}
    print_table(lev)
    print("[FIX 6 APPLIED]")

    # ------------------------------------------------------------------ FIX 7
    banner("FIX 7: leakage audit")
    for k, r in res.items():
        dates = r["prices"].index
        tdf = pd.DataFrame(r["trades"])
        tdf["signal_date"] = dates[tdf["signal_idx"]].date
        tdf["execution_date"] = dates[tdf["exec_idx"]].date
        tdf["execution_price"] = [r["p"]["open"][a].iloc[e] for a, e in zip(tdf["traded_asset"], tdf["exec_idx"])]
        tdf["signal_date_close"] = [r["p"]["close"][a].iloc[s] for a, s in zip(tdf["traded_asset"], tdf["signal_idx"])]
        assert (tdf["exec_idx"] > tdf["signal_idx"]).all(), "execution not strictly after signal"
        assert (pd.to_datetime(tdf["execution_date"]) > pd.to_datetime(tdf["signal_date"])).all()
        r["trades_df"] = tdf
        cols = ["signal_date", "execution_date", "asset_from", "asset_to", "traded_asset",
                "execution_price", "signal_date_close", "weight_to"]
        print(f"\nSample {k}: {len(tdf)} trade events. First 20 (raw unadjusted prices of traded asset):")
        print(tdf[cols].head(20).round(4).to_string(index=False))
        print(f"Sample {k}: last 20:")
        print(tdf[cols].tail(20).round(4).to_string(index=False))
        print(f"[ASSERT PASSED] execution_date > signal_date for all {len(tdf)} trades")

        # Rebuild indicators on data truncated at every row T and require bit-identical row T.
        prices, ind = r["prices"], r["ind"]
        for T in range(len(prices)):
            row_t = compute_indicators(prices.iloc[:T + 1]).iloc[T]
            full_t = ind.iloc[T]
            same = (row_t == full_t) | (row_t.isna() & full_t.isna())
            assert same.all(), f"indicator leakage at {prices.index[T].date()}: {list(same[~same].index)}"
        print(f"[ASSERT PASSED] indicators rebuilt on truncated data identical at all {len(prices)} rows")

        # End-to-end: decisions from truncated data (incl. the next-open order) match the full run.
        check_ts = np.linspace(WARMUP + 5, len(prices) - 1, 40).astype(int)
        for T in check_ts:
            trunc = decide_schedule(compute_indicators(prices.iloc[:T + 1]), PARAMS["target_vol"],
                                    PARAMS["rebalance_freq"], WARMUP)
            full = {t: v for t, v in r["sched_full"].items() if t <= T + 1}
            assert trunc == full, f"decision leakage at {prices.index[T].date()}"
        print(f"[ASSERT PASSED] trade decisions from truncated data identical at {len(check_ts)} cut dates")

    fb = panels["open_fallbacks"]
    print(f"\nForward-filled price cells (SPY/QQQ, common calendar): {panels['n_ffilled']}")
    print(f"Interpolated price cells: 0 (none performed)")
    print(f"Open-price fallbacks to adjusted close: {len(fb)} {fb[:20] if fb else ''}")
    print(f"Risk-free forward-fill runs > 5 business days: {len(rf_gap_log)}")
    print("[FIX 7 APPLIED]")

    # ------------------------------------------------------------------ FIX 8
    banner("FIX 8: benchmarks and random control")
    for k, r in res.items():
        rows = {"Strategy": r["metrics"]}
        rows.update({b: compute_metrics(e, r["rf"]) for b, e in r["bench"].items()})
        r["bench_metrics"] = rows
        print(f"\nSample {k}:")
        print_table(rows)
        pc = (r["rand_cagr"] < r["metrics"]["CAGR"]).mean() * 100
        ps = (r["rand_sharpe"] < r["metrics"]["Sharpe"]).mean() * 100
        r["pct_cagr"], r["pct_sharpe"] = pc, ps
        print(f"Random control ({PARAMS['random_paths']} paths, seed {PARAMS['random_seed']}, "
              f"{r['act']['Position Switches']} random switches, target avg exposure "
              f"{r['act']['Avg Risky Exposure']:.1%}, realised mean {r['rand_expo'].mean():.1%}):")
        print(f"  random CAGR   p5/p50/p95: {np.percentile(r['rand_cagr'], [5, 50, 95]).round(4)}"
              f"  -> strategy CAGR {r['metrics']['CAGR']:.2%} at percentile {pc:.1f}")
        print(f"  random Sharpe p5/p50/p95: {np.percentile(r['rand_sharpe'], [5, 50, 95]).round(3)}"
              f"  -> strategy Sharpe {r['metrics']['Sharpe']:.2f} at percentile {ps:.1f}")
    print(f"\nEquity charts include all benchmarks + random 5-95% band ({out_dir}/equity_log_*.png)")
    print("[FIX 8 APPLIED]")

    # ------------------------------------------------------------------ FIX 9
    banner("FIX 9: write the record")
    for k, r in res.items():
        daily = pd.DataFrame({"strategy": r["eq"], **r["bench"],
                              "random_p5": np.percentile(r["rand_eq"], 5, axis=0),
                              "random_p95": np.percentile(r["rand_eq"], 95, axis=0),
                              "strategy_holding": r["sim"]["holding"],
                              "strategy_exposure": r["sim"]["exposure"],
                              "rf_daily": r["rf"]})
        daily.to_csv(os.path.join(out_dir, f"daily_equity_{k}.csv"), index_label="date")
        r["trades_df"].to_csv(os.path.join(out_dir, f"trades_{k}.csv"), index=False)
        cal_tables[k].to_csv(os.path.join(out_dir, f"calendar_years_{k}.csv"), index_label="year")
        pd.DataFrame(r["bench_metrics"]).to_csv(os.path.join(out_dir, f"metrics_{k}.csv"))
        pd.DataFrame({"cagr": r["rand_cagr"], "sharpe": r["rand_sharpe"], "exposure": r["rand_expo"]}) \
            .to_csv(os.path.join(out_dir, f"random_control_{k}.csv"), index_label="path")
        if r["pending"]:
            print(f"Sample {k}: pending order for next open: {r['pending'][1]} @ weight {r['pending'][2]:.3f}")
    pd.DataFrame(lev).to_csv(os.path.join(out_dir, "leverage_ceiling.csv"))

    row = {
        "experiment_id": f"EXP-{run_id}",
        "date": datetime.now().strftime("%Y-%m-%d"),
        "strategy": "Dual-momentum SPY/QQQ/cash rotation, 12% vol target",
        "market": "US equity ETFs (Sharadar SFP) + FRED DTB3 cash",
        "hypothesis": "Regime + risk-adjusted momentum rotation beats buy-and-hold on risk-adjusted basis "
                      "and beats random timing with equal activity, net of realistic costs",
        "parameters": json.dumps(PARAMS),
        "cost_assumption": "$0.005/sh min $1/order + 2bps slippage one way on traded notional",
        "output_dir": out_dir,
    }
    for k, r in res.items():
        row[f"{k}_period"] = f"{r['eq'].index[1].date()} to {r['eq'].index[-1].date()}"
        for m, v in {**r["metrics"], **r["act"]}.items():
            row[f"{k}_{m}"] = v
        for m, v in r["sens"].items():
            row[f"{k}_CAGR_{m}x_slippage"] = v
        row[f"{k}_random_pct_CAGR"] = r["pct_cagr"]
        row[f"{k}_random_pct_Sharpe"] = r["pct_sharpe"]
    row["status"] = "COMPLETED - parameter provenance UNKNOWN, treat as in-sample"
    log_path = "experiment_log.csv"
    log = pd.DataFrame([row])
    if os.path.exists(log_path):
        log = pd.concat([pd.read_csv(log_path), log], ignore_index=True)
    log.to_csv(log_path, index=False)
    print(f"Appended {row['experiment_id']} to {log_path} (now {len(log)} rows)")
    print(f"Outputs in {out_dir}: {sorted(os.listdir(out_dir))}")
    print(f"[FIX 9 APPLIED]  total runtime {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
