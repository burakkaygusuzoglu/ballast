# 0006 Secrets live only in `.env`; refuse keys that can withdraw

**Context.** An exchange API key is the most valuable secret in the project. A leaked key with withdrawal rights means
lost funds.

**Decision.**
- Keys are read from a local `.env` file. Git ignores it; only `.env.example` is committed.
- At start-up, live mode reads the key's permissions and refuses to run if withdrawals are enabled.
- Recommended key settings: read and spot trading only, withdrawals disabled, IP whitelist on.
- Real-money mode requires a typed confirmation.

**Consequences.**
- A leaked key can, at worst, place trades from a whitelisted IP. It cannot move funds out.
