# 0004 Pessimistic costs, real funding and intraday liquidation

**Context.** Small edges disappear under realistic costs. Hourly strategies in round 1 lost about 18 % of starting
equity to fees and slippage.

**Decision.**
- Spot: 0.1 % fee per fill. Futures: 0.05 %. 0.05 % slippage on every market or stop fill. Stops that gap fill at the
  open.
- Futures pay or receive the historical funding rate on each 8-hour event.
- Leveraged portfolios are checked for liquidation with the day's low (longs) and high (shorts) at the same time,
  using a 2 % maintenance margin.
- Exchange lot steps and minimum order values are applied in small-account runs.

**Consequences.**
- Results are conservative. Funding alone cut the futures version of the best strategy from +215 % to +158 %.
- Small accounts often cannot place orders at all, and the reports say so.
