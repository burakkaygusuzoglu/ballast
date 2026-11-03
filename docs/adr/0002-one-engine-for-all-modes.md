# 0002 One engine and one fill model for backtest, paper and live

**Context.** Many trading bots backtest with one code path and trade with another. The two drift apart, and live
results stop matching the research.

**Decision.**
- The engine (`bot/engine.py`) processes closed bars only. It talks to a broker interface (`bot/brokers/base.py`).
- Backtest and paper trading use the same simulated broker and fill model (`bot/brokers/simulated.py`).
- Live trading uses a broker with the same interface (`bot/brokers/live.py`). Futures use a simulator subclass.

**Consequences.**
- Paper trades can be replayed through the backtester and compared trade by trade. A test enforces this equivalence.
- New execution venues only need a new broker class.
