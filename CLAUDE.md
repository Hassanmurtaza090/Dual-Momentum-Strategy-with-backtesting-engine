# Project: Quant Dual-Momentum Engine (SPY / QQQ / BIL)

## Environment & Tooling
- Python virtual environment: `.venv`
- Always run commands using `.venv/bin/python` (macOS/Linux) or `.venv\Scripts\python.exe` (Windows).
- Core dependencies: `requests` (direct Sharadar REST API — never `nasdaqdatalink`), `pandas`, `numpy`, `matplotlib`, `python-dotenv`.

## Architecture & Data Guidelines
- **Sharadar Table Rules**: 
  - ETFs (`SPY`, `QQQ`, `BIL`) MUST be fetched from `https://api.sharadar.com/v1.0/data/funds` (SFP, Fund Prices). Never query `stocks` (SEP) for ETFs.
  - Always use `closeadj` to account for split and monthly dividend total return (mandatory for `BIL`).
  - Cache parquet files inside `data/` to avoid unnecessary API consumption.
- **Backtest Integrity**:
  - Zero lookahead bias: signals derived on bar $t-1$ close execute on bar $t$ open.
  - All rolling calculations must have explicit warmup handling (minimum 201 bars).
  - Always deduct 3 bps ($0.0003$) per unit of turnover friction.

## Commands
- Run backtest: `python main.py`
- Run unit/regression tests: `pytest tests/`

## Reporting standard: fixed size and compounded, always

Every backtest must report BOTH return measures, side by side, in every
results table. Never one alone.

1. Compounded (geometric, CAGR): the rate that turns the start equity into
   the end equity. Position sizes grow with the account.
2. Fixed size (arithmetic mean of yearly returns): the average of each
   calendar year's return, computed on a constant base. Position sizes never
   grow. Profits are treated as withdrawn.

Both must be simulated properly. Do not derive the fixed-size figure by
averaging the compounded run's yearly percentages. Run a second pass where
position notional is reset to the starting equity at the beginning of every
year, and report that pass's average.

Also print the gap between them, labelled "volatility drag", and the check:

    gap should be approximately (variance of yearly returns) / 2

If the observed gap differs from that approximation by more than half a
point, say so. It usually means fat tails or a small number of years.

Drawdown is reported for both passes too. They will differ: the fixed-size
pass usually shows a smaller percentage drawdown because losses do not
compound either.

Partial years at the start or end of a sample are labelled partial and
excluded from the fixed-size average. Never annualise a partial year.