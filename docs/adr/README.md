# Architecture decision records

Short records of the decisions that shape Ballast: the context, the decision, and its consequences.
Format after Michael Nygard's ADRs.

| # | Decision | Status |
|---|---|---|
| [0001](0001-pre-registered-research.md) | Pre-register every research round and pin the protocol with SHA-256 | Accepted |
| [0002](0002-one-engine-for-all-modes.md) | One engine and one fill model for backtest, paper and live | Accepted |
| [0003](0003-risk-manager-veto.md) | The risk manager can veto any order; a drawdown breaker halts the bot | Accepted |
| [0004](0004-pessimistic-cost-model.md) | Pessimistic fills, real funding and intraday liquidation in simulation | Accepted |
| [0005](0005-spot-first-no-leverage.md) | Spot, long-only and unlevered until a leveraged setup passes out of sample | Accepted |
| [0006](0006-secrets-and-key-permissions.md) | Keys only in `.env`; refuse keys that can withdraw | Accepted |
