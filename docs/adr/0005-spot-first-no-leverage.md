# 0005 Spot, long-only and unlevered by default

**Context.** Leverage scales both the edge and the losses. The growth-optimal (Kelly) fraction depends on an edge that
is only estimated, and over-betting is punished far more than under-betting.

**Decision.**
- The reference bot trades spot, long-only, without leverage.
- A leveraged or short setup can only be adopted after it passes a pre-registered out-of-sample test.
- Real money with leverage needs the owner's explicit approval, isolated or cross margin limits and exchange-side
  stops.

**Consequences.**
- Rounds 6 and 9 tested leverage and shorting. Both failed out of sample, so the default stands.
