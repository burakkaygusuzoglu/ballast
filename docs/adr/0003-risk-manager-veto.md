# 0003 The risk manager can veto any order

**Context.** Strategy code proposes trades. Without an independent layer, a bug or an unusual market can size a position
far beyond the intended risk.

**Decision.**
- Every proposal passes through `bot/risk.py` before it reaches a broker.
- Sizing: the loss at the stop, including fees and slippage, equals a fixed fraction of equity (0.5 % for the reference
  bot).
- Limits: maximum position size and number of positions, trades per day, cooldown after an exit, daily loss limit
  and a cost filter.
- Circuit breaker: a drawdown of 20 % from the peak halts the bot until a human resets it.

**Consequences.**
- Strategies stay simple; safety lives in one tested place.
- The breaker sometimes stops the bot before a recovery. The research reports results with and without it so that
  this cost is visible.
