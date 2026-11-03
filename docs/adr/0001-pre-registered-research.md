# 0001 Pre-register every research round

**Context.** Backtests invite silent tuning. When parameters and pass/fail rules are chosen after looking at results,
an apparently profitable strategy is often just the luckiest of many trials. Published anomalies lose about half their
returns after publication (McLean & Pontiff, 2016).

**Decision.**
- Before any backtest of a strategy family, write a protocol: candidates, parameters, periods, gates and the
  selection rule.
- Pin the protocol's SHA-256 in the research script. `protocol_check()` refuses to run if the file changed.
- Run each stage once. A bug fix is allowed, but it is documented next to the result. Parameters never change after
  results are seen.

**Consequences.**
- Most rounds end in "not adopted". That is the point: the published failures make the published results credible.
- A near miss counts as a fail. Round 8 missed its profit gate by 1.1 % and was not adopted.
- See the [protocol registry](../research/PROTOCOL_REGISTRY.md).
