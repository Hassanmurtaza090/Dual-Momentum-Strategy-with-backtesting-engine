## Hypothesis
The rotation loses return in two places, both caused by the rule being a
binary switch with no memory:
1. Defensive years: while mostly in cash it still earned less than
   T-bills (2001, 2002, 2008), and long bonds rallied in those years.
2. Choppy years: it switched in and out and earned about 0.6% a year while
   SPY earned about 8.8% (2004, 2005, 2010, 2011, 2012, 2015, 2016).

## Change A: bonds as the defensive holding
- Cash (T-bill index) stays as the TEST: the strategy is invested in
  equities only if the winner's momentum beats cash momentum. Unchanged.
- When the test fails, hold an intermediate US Treasury position instead
  of cash.
- Source: Antonacci, Dual Momentum Investing (2014), Global Equities
  Momentum, which holds bonds when equities fail the absolute momentum test.
- Known deviation: GEM uses an aggregate bond index. We use intermediate
  Treasuries because a long, clean history is available from yields. This
  choice is made for data reasons, before seeing any result.
- Known risk: in 2022 bonds fell with stocks. This year is expected to get
  worse and must be reported, not hidden.

## Change B: re-entry harder than exit
- Exit rule unchanged: leave equities immediately when the regime turns off.
- Re-entry only on a scheduled 21-day rebalance date, and only if BOTH:
  a) winner's blended momentum exceeds cash momentum by at least 1.0
     percentage point, and
  b) the same condition also held on the previous scheduled rebalance date.
- Why 1.0 point: the Round 3 audit found 12 decisions within 0.5 points of
  the threshold. 1.0 point is twice that noise band. It is derived from the
  audit, not from any performance result, and will not be changed.

## Variants and trial count
- V0: original strategy (control)
- VA: Change A only
- VB: Change B only
- VAB: both changes
Three new variants are tested, so the trial count is 3. This number goes
into the log for any multiple-testing adjustment.

## Success criteria (judge VAB against V0, full 1999-12 to 2026-09)
VAB is accepted only if ALL of these hold:
1. Average return across the seven choppy years (2004, 2005, 2010, 2011,
   2012, 2015, 2016) improves by at least 2.0 percentage points.
2. In the defensive years (2000, 2001, 2002, 2008), return minus cash
   improves on average.
3. Full-period max drawdown is not worse than 16%.
4. Full-period compounded CAGR is not lower than V0.
5. The pre-2008 slice (1999-12 to 2008-02) does not get worse.

## Failure
If any criterion fails, VAB is rejected. It will not be rescued by
changing the 1.0 point buffer, the confirmation rule, or the bond choice.
If VA or VB passes alone but VAB fails, report it but do not adopt it
without a separate pre-registration.

## Expected result, written before running
CAGR about 1 to 2.5 points above V0. Max drawdown similar or slightly
lower. 2022 worse than V0. If CAGR comes back more than 4 points above
V0, treat it as a probable bug and audit before reporting.
