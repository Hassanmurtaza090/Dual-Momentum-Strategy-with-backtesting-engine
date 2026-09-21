"""Final measurement/audit run: ablations, pre-2008 slice, audit checks, write-up. No parameter changes."""
import json
import os
from datetime import datetime

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from main import PARAMS, banner, fast_sharpe, print_table, sim_cagr
from src.data_loader import build_price_panels, fetch_fred_rf, fetch_sharadar_etf_data
from src.engine import (CASH, RISKY, WARMUP, buy_and_hold_schedule, compute_indicators, decide_schedule,
                        simulate)
from src.metrics import activity_metrics, compute_metrics, drawdown_episodes

TV, FREQ, CAP = PARAMS["target_vol"], PARAMS["rebalance_freq"], PARAMS["initial_capital"]
SAMPLES = {"A": "2007-06-01", "B": None}
HOLDOUT_END = "2008-02-29"
CORE = ["CAGR", "Ann. Volatility", "Sharpe", "Sortino", "Max Drawdown"]


def vt_weight(ind, asset, s):
    vol = ind["vol20_qqq" if asset == "QQQ" else "vol20_spy"].iloc[s]
    return min(1.0, TV / vol) if vol > 0 else 1.0


def spy_vol_target_schedule(ind, n):
    """ABLATION_B: long SPY from the first open, vol-targeted, resized every 21 sessions."""
    return {t: (t - 1, "SPY", vt_weight(ind, "SPY", t - 1)) for t in range(WARMUP, n, FREQ)}


def switch_events(sched):
    """Real-strategy events flagged as switch (asset changed) or resize."""
    out, cur = [], CASH
    for t in sorted(sched):
        s, a, _ = sched[t]
        out.append((t, s, a != cur))
        cur = a
    return out


def random_signal_schedule(rng, events, probs, ind):
    """ABLATION_A: same event dates; at real switch dates draw an asset at random, else resize the held one."""
    names = list(probs)
    p = np.array([probs[k] for k in names])
    cur, out = CASH, {}
    for t, s, is_switch in events:
        if is_switch:
            cur = names[rng.choice(len(names), p=p)]
        out[t] = (s, cur, 0.0 if cur == CASH else vt_weight(ind, cur, s))
    return out


def attribution_sim(panels, cash_kind, cash_ret, sched, exec_open, drift, cost_kind):
    """
    Flexible replica used only for fix-by-fix attribution.
    exec_open=False: trade at close(t-1), new weights earn all of day t (old look-ahead).
    drift=False: free reset to target weights every day (old constant-weight accounting).
    cost_kind 'old': 3 bps x old turnover formula; 'new': $0.005/sh min $1 + 2 bps (BIL traded if it is the cash leg).
    """
    ac = panels["adj_close"]; ao = panels["adj_open"]; op = panels["open"]; cl = panels["close"]
    ac = {a: ac[a].tolist() for a in ac}; ao = {a: ao[a].tolist() for a in ao}
    op = {a: op[a].tolist() for a in op}; cl = {a: cl[a].tolist() for a in cl}
    cr = cash_ret.tolist()
    n = len(cr)
    hold = {"SPY": 0.0, "QQQ": 0.0, CASH: CAP}
    asset, w = CASH, 0.0
    eq = [CAP]

    def grow(t, kind):
        for a in RISKY:
            if hold[a]:
                hold[a] *= {"full": ac[a][t] / ac[a][t - 1], "gap": ao[a][t] / ac[a][t - 1],
                            "intra": ac[a][t] / ao[a][t]}[kind]
        if kind != "intra":
            hold[CASH] *= 1.0 + cr[t]

    for t in range(WARMUP, n):
        if not drift:
            V = sum(hold.values())
            hold.update({"SPY": 0.0, "QQQ": 0.0, CASH: V})
            if asset != CASH:
                hold[asset], hold[CASH] = w * V, (1 - w) * V
        order = sched.get(t)
        cost = 0.0
        if order is None:
            grow(t, "full")
        else:
            if exec_open:
                grow(t, "gap")
            _, na, nw = order
            V = sum(hold.values())
            target = {"SPY": 0.0, "QQQ": 0.0}
            if na != CASH:
                target[na] = nw * V
            if cost_kind == "old":
                we_old = 1.0 if asset == CASH else w
                we_new = 1.0 if na == CASH else nw
                cost = (we_old + we_new if na != asset else abs(we_new - we_old)) * 0.0003 * V
            else:
                px = (lambda a: op[a][t]) if exec_open else (lambda a: cl[a][t - 1])
                legs = [(a, abs(target[a] - hold[a])) for a in RISKY]
                if cash_kind == "BIL":
                    legs.append(("BIL", abs((V - sum(target.values())) - hold[CASH])))
                for a, d in legs:
                    if d > 0:
                        cost += max(1.0, 0.005 * d / px(a)) + 0.0002 * d
            hold.update(target)
            hold[CASH] = V - sum(target.values())
            asset, w = na, nw
            grow(t, "intra" if exec_open else "full")
        hold[CASH] -= cost
        eq.append(sum(hold.values()))
    return pd.Series(eq, index=panels["adj_close"].index[WARMUP - 1:])


def main():
    run_id = datetime.now().strftime("%Y-%m-%d_%H%M%S")
    out_dir = os.path.join("results", f"{run_id}_final")
    os.makedirs(out_dir, exist_ok=False)
    rng = np.random.default_rng(PARAMS["random_seed"])

    raw = fetch_sharadar_etf_data(["SPY", "QQQ", "BIL"], start_date="1999-01-01")
    panels = build_price_panels(raw, ["SPY", "QQQ"])
    rf_annual, rf_quote, _ = fetch_fred_rf(panels["adj_close"].index)
    cash_ret = rf_quote.shift(1).fillna(rf_quote.iloc[0])

    res = {}
    for name, start in SAMPLES.items():
        sl = slice(start, None)
        p = {k: panels[k].loc[sl] for k in ["adj_close", "adj_open", "open", "close"]}
        cr = cash_ret.loc[sl]
        prices = p["adj_close"].assign(**{CASH: (1 + cr).cumprod()})
        ind = compute_indicators(prices)
        n = len(prices)
        sched = {t: v for t, v in decide_schedule(ind, TV, FREQ, WARMUP).items() if t < n}
        sim, trades = simulate(p, cr, sched, WARMUP, CAP)
        zero, _ = simulate(p, cr, sched, WARMUP, CAP, slippage_bps=0, commission_per_share=0, min_commission=0)
        rf_s = cr.reindex(sim.index)
        res[name] = dict(p=p, cr=cr, prices=prices, ind=ind, sched=sched, sim=sim, trades=trades,
                         zero=zero, rf=rf_s, n=n)

    # ------------------------------------------------------------------ A. ablation
    banner("A. 2x2 ablation (same 21-day cadence, cost model, next-open execution)")
    ablation_rows = {}
    for name, r in res.items():
        p, cr, ind, n, sched, rf_s = r["p"], r["cr"], r["ind"], r["n"], r["sched"], r["rf"]
        eq_full = r["sim"]["equity"]
        eq_b = simulate(p, cr, spy_vol_target_schedule(ind, n), WARMUP, CAP)[0]["equity"]
        sched_c = {t: (s, a, 0.0 if a == CASH else 1.0) for t, (s, a, _) in sched.items()}
        eq_c = simulate(p, cr, sched_c, WARMUP, CAP)[0]["equity"]

        events = switch_events(sched)
        dests = pd.Series([sched[t][1] for t, _, sw in events if sw])
        probs = dests.value_counts(normalize=True).reindex(["SPY", "QQQ", CASH], fill_value=0).to_dict()
        paths = []
        for _ in range(PARAMS["random_paths"]):
            e = simulate(p, cr, random_signal_schedule(rng, events, probs, ind), WARMUP, CAP)[0]["equity"]
            paths.append(e)
        a_cagr = np.array([sim_cagr(e) for e in paths])
        a_sharpe = np.array([fast_sharpe(e, rf_s) for e in paths])
        med = paths[int(np.argsort(a_cagr)[len(a_cagr) // 2])]

        m = {k: compute_metrics(e, rf_s) for k, e in
             [("Strategy (signal + VT)", eq_full), ("ABL_A random + VT (median)", med),
              ("ABL_B SPY + VT (no signal)", eq_b), ("ABL_C signal, 100% (no VT)", eq_c)]}
        print(f"\nSample {name} ({eq_full.index[1].date()} -> {eq_full.index[-1].date()}):")
        print_table({k: {c: v[c] for c in CORE} for k, v in m.items()})
        pc = (a_cagr < m["Strategy (signal + VT)"]["CAGR"]).mean() * 100
        ps = (a_sharpe < m["Strategy (signal + VT)"]["Sharpe"]).mean() * 100
        print(f"ABL_A: {len(events)} events, {int(dests.size)} switch dates; draw probabilities "
              f"{ {k: round(v, 3) for k, v in probs.items()} }")
        print(f"  random CAGR   p5/p50/p95 {np.percentile(a_cagr, [5, 50, 95]).round(4)} -> strategy percentile {pc:.1f}")
        print(f"  random Sharpe p5/p50/p95 {np.percentile(a_sharpe, [5, 50, 95]).round(3)} -> strategy percentile {ps:.1f}")
        ablation_rows[name] = dict(m=m, pc=pc, ps=ps, probs=probs,
                                   a_cagr=a_cagr, a_sharpe=a_sharpe)
        pd.DataFrame({k: {c: v[c] for c in CORE} for k, v in m.items()}).to_csv(
            os.path.join(out_dir, f"ablation_{name}.csv"))
        pd.DataFrame({"cagr": a_cagr, "sharpe": a_sharpe}).to_csv(
            os.path.join(out_dir, f"ablation_A_paths_{name}.csv"), index_label="path")
        pd.DataFrame({"strategy": eq_full, "abl_a_median": med, "abl_b": eq_b, "abl_c": eq_c}).to_csv(
            os.path.join(out_dir, f"ablation_equity_{name}.csv"), index_label="date")

    # ------------------------------------------------------------------ B. pre-2008 slice
    banner(f"B. Pre-2008 slice: 1999-12 -> 2008-02 (taken from the Sample B path)")
    r = res["B"]
    sl = slice(None, HOLDOUT_END)
    spy = simulate(r["p"], r["cr"], buy_and_hold_schedule("SPY", WARMUP), WARMUP, CAP)[0]["equity"].loc[sl]
    qqq = simulate(r["p"], r["cr"], buy_and_hold_schedule("QQQ", WARMUP), WARMUP, CAP)[0]["equity"].loc[sl]
    sim_h = r["sim"].loc[sl]
    eq_h = sim_h["equity"]
    end_idx = len(eq_h) - 1 + WARMUP - 1
    trades_h = [tr for tr in r["trades"] if tr["exec_idx"] <= end_idx]
    rf_h = r["rf"].loc[sl]
    mh = compute_metrics(eq_h, rf_h)
    mh.update(activity_metrics(sim_h, trades_h, sim_cagr(r["zero"]["equity"].loc[sl]), mh["CAGR"]))
    print(f"Slice {eq_h.index[1].date()} -> {eq_h.index[-1].date()}, {len(eq_h) - 1} sessions")
    print_table({"Strategy": mh, "SPY B&H": compute_metrics(spy, rf_h), "QQQ B&H": compute_metrics(qqq, rf_h)})
    yr = lambda e: pd.concat([e.iloc[:1], e.resample("YE").last()]).pct_change().dropna()
    cal = pd.DataFrame({"Strategy": yr(eq_h), "SPY": yr(spy), "QQQ": yr(qqq)})
    cal.index = cal.index.year
    print("\nCalendar years (1999 = 22-31 Dec only, 2008 = Jan-Feb only):")
    print((cal * 100).round(2).astype(str).add("%").to_string())
    cal.to_csv(os.path.join(out_dir, "holdout_calendar_years.csv"), index_label="year")
    pd.DataFrame({"Strategy": mh}).to_csv(os.path.join(out_dir, "holdout_metrics.csv"))
    pd.DataFrame({"strategy": eq_h, "spy": spy, "qqq": qqq}).to_csv(
        os.path.join(out_dir, "holdout_equity.csv"), index_label="date")
    fig, ax = plt.subplots(figsize=(12, 6))
    for lab, e, c, lw in [("Strategy", eq_h, "#004488", 1.8), ("SPY B&H", spy, "#888888", 1.0),
                          ("QQQ B&H", qqq, "#BB5566", 1.0)]:
        ax.plot(e.index, e, label=lab, color=c, lw=lw)
    ax.set_yscale("log"); ax.set_title("Equity (log), 1999-12 to 2008-02 slice"); ax.set_ylabel("$")
    ax.grid(alpha=0.3); ax.legend(); fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "holdout_equity_log.png"), dpi=150); plt.close(fig)

    # ------------------------------------------------------------------ C. audits
    banner("C1. Maximum drawdown event per sample")
    for name, rr in res.items():
        eps = drawdown_episodes(rr["sim"]["equity"])
        worst = eps.loc[eps["depth"].idxmin()]
        print(f"Sample {name}: peak {worst['peak'].date()}  trough {worst['trough'].date()}  "
              f"recovery {worst['recovery'].date() if pd.notna(worst['recovery']) else 'not recovered'}  "
              f"depth {worst['depth']:.4%}")
    ea = res["A"]["sim"]["equity"].loc["2010-01-01":"2011-12-31"]
    eb = res["B"]["sim"]["equity"].loc["2010-01-01":"2011-12-31"]
    ra, rb = ea.pct_change().dropna(), eb.pct_change().dropna()
    print(f"Daily returns A vs B in 2010-2011: max abs difference {(ra - rb).abs().max():.2e} "
          f"(same holdings on {(res['A']['sim']['holding'].loc[ra.index] == res['B']['sim']['holding'].loc[ra.index]).mean():.1%} of days)")

    banner("C2. Decisions near the regime threshold, and BIL vs synthetic cash")
    bil_panels = build_price_panels(raw, ["SPY", "QQQ", "BIL"])
    bp = {k: bil_panels[k].loc[SAMPLES["A"]:] for k in ["adj_close", "adj_open", "open", "close"]}
    assert bp["adj_close"].index.equals(res["A"]["prices"].index), "BIL calendar differs from Sample A"
    prices_bil = bp["adj_close"][RISKY].assign(**{CASH: bp["adj_close"]["BIL"]})
    ind_bil = compute_indicators(prices_bil)
    n = len(prices_bil)
    sched_bil = {t: v for t, v in decide_schedule(ind_bil, TV, FREQ, WARMUP).items() if t < n}
    print(f"Old-engine (BIL) decisions: {len(sched_bil)}")
    rows = []
    for t, (s, a, _) in sched_bil.items():
        d1 = ind_bil["spy_close"].iloc[s] / ind_bil["spy_sma200"].iloc[s] - 1
        d2 = ind_bil["mom_spy"].iloc[s] - ind_bil["mom_cash"].iloc[s]
        n1, n2 = abs(d1) < 0.005, abs(d2) < 0.005
        c1, c2 = d1 > 0, d2 > 0
        if c1 and c2:
            fragile = n1 and n2
        elif c1:
            fragile = n1
        elif c2:
            fragile = n2
        else:
            fragile = n1 or n2
        rows.append((n1, n2, fragile))
    near = pd.DataFrame(rows, columns=["sma", "mom", "fragile"])
    print(f"SPY within 0.5% of its 200-day SMA:                 {int(near['sma'].sum())}")
    print(f"SPY momentum within 0.5 pp of cash momentum:        {int(near['mom'].sum())}")
    print(f"Regime outcome flips with a <=0.5% move (OR logic): {int(near['fragile'].sum())}")

    sched_syn = res["A"]["sched"]
    dates = prices_bil.index
    all_t = sorted(set(sched_bil) | set(sched_syn))
    same = only_bil = only_syn = diff_asset = diff_w = 0
    for t in all_t:
        b, s = sched_bil.get(t), sched_syn.get(t)
        if b is None:
            only_syn += 1
        elif s is None:
            only_bil += 1
        elif b[1] != s[1]:
            diff_asset += 1
        elif b[1] != CASH and b[2] != s[2]:
            diff_w += 1
        else:
            same += 1
    print(f"Synthetic-cash decisions: {len(sched_syn)}. Identical {same}; only in BIL run {only_bil}; "
          f"only in synthetic run {only_syn}; same date different asset {diff_asset}; different weight {diff_w}")
    hb = pd.Series(CASH, index=dates)
    hs = pd.Series(CASH, index=dates)
    for sched_x, h in [(sched_bil, hb), (sched_syn, hs)]:
        for t in sorted(sched_x):
            h.iloc[t:] = sched_x[t][1]
    diff_days = (hb.iloc[WARMUP:] != hs.iloc[WARMUP:])
    print(f"Days with a different target holding: {int(diff_days.sum())} of {len(diff_days)} "
          f"({diff_days.mean():.2%}); changed-decision dates: "
          f"{[dates[t].date().isoformat() for t in all_t if sched_bil.get(t, (0, 0))[1:2] != sched_syn.get(t, (0, 0))[1:2]][:15]}")

    banner("C3. Attribution: old 9.27% -> new 9.33% (Sample A)")
    bil_ret = bp["adj_close"]["BIL"].pct_change().fillna(0.0)
    syn_ret = res["A"]["cr"]
    cfg = {  # name: (cash_kind, exec_open, drift, cost_kind)
        "0 baseline (old engine)": ("BIL", False, False, "old"),
        "FIX 1a next-open execution only": ("BIL", True, False, "old"),
        "FIX 1b weight drift only": ("BIL", False, True, "old"),
        "FIX 4 new cost model only": ("BIL", False, False, "new"),
        "FIX 2+3 synthetic cash only": ("SYN", False, False, "old"),
        "cum: + 1a": ("BIL", True, False, "old"),
        "cum: + 1a + 1b": ("BIL", True, True, "old"),
        "cum: + 1a + 1b + 4": ("BIL", True, True, "new"),
        "cum: + 1a + 1b + 4 + 2/3 (= new run)": ("SYN", True, True, "new"),
    }
    attr = {}
    for label, (ck, eo, dr, co) in cfg.items():
        sch = sched_bil if ck == "BIL" else sched_syn
        e = attribution_sim(bp, ck, bil_ret if ck == "BIL" else syn_ret, sch, eo, dr, co)
        attr[label] = sim_cagr(e)
    base = attr["0 baseline (old engine)"]
    for label, v in attr.items():
        print(f"{label:<42} CAGR {v:.4%}   change vs baseline {(v - base) * 1e4:+7.1f} bp")
    main_cagr = sim_cagr(res["A"]["sim"]["equity"])
    print(f"Check: replica of full new run {attr['cum: + 1a + 1b + 4 + 2/3 (= new run)']:.6%} "
          f"vs main engine {main_cagr:.6%}")
    print("FIX 2 alone changes no CAGR (rate only feeds the cash leg via FIX 3, and Sharpe/Sortino); "
          "FIX 5-9 are reporting only.")
    pd.Series(attr, name="CAGR").to_csv(os.path.join(out_dir, "attribution_sample_A.csv"))

    # ------------------------------------------------------------------ D. record
    banner("D. Record")
    log_path = "experiment_log.csv"
    log = pd.read_csv(log_path)
    new_rows = []
    specs = {
        "ABLATION_A": ("Random signal + 12% vol target (1000 paths, real switch dates)",
                       "ABL_A random + VT (median)", "Sharpe advantage comes from the momentum signal"),
        "ABLATION_B": ("Always long SPY + 12% vol target, 21-day resize",
                       "ABL_B SPY + VT (no signal)", "Sharpe advantage comes from vol targeting alone"),
        "ABLATION_C": ("Momentum signal, fixed 100% weight (no vol target)",
                       "ABL_C signal, 100% (no VT)", "Sharpe advantage survives without vol targeting"),
    }
    for abl, (desc, key, hyp) in specs.items():
        row = {"experiment_id": f"EXP-{run_id}-{abl}", "date": datetime.now().strftime("%Y-%m-%d"),
               "strategy": f"{abl}: {desc}", "market": "US equity ETFs (Sharadar SFP) + FRED DTB3 cash",
               "hypothesis": f"Ablation test: {hyp}", "parameters": json.dumps(PARAMS),
               "cost_assumption": "$0.005/sh min $1/order + 2bps slippage one way on traded notional",
               "output_dir": out_dir}
        for name, ar in ablation_rows.items():
            eq = res[name]["sim"]["equity"]
            row[f"{name}_period"] = f"{eq.index[1].date()} to {eq.index[-1].date()}"
            for c in CORE:
                row[f"{name}_{c}"] = ar["m"][key][c]
            if abl == "ABLATION_A":
                row[f"{name}_strategy_pct_CAGR_vs_paths"] = ar["pc"]
                row[f"{name}_strategy_pct_Sharpe_vs_paths"] = ar["ps"]
                row[f"{name}_draw_probabilities"] = json.dumps(ar["probs"])
        row["status"] = "ABLATION - measurement only; parent strategy REJECTED"
        new_rows.append(row)
    log = pd.concat([log, pd.DataFrame(new_rows)], ignore_index=True)
    log.to_csv(log_path, index=False)
    print(f"Appended {len(new_rows)} ablation rows to {log_path} (now {len(log)} rows)")
    print(f"Outputs in {out_dir}: {sorted(os.listdir(out_dir))}")


if __name__ == "__main__":
    main()
