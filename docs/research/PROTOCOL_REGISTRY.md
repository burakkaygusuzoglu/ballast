# Protocol registry

Every research round in this project followed the same rule: **the protocol (candidates, parameters, pass/fail
gates and the selection rule) was written and fingerprinted before any backtest of that strategy family was run.**
Each script refuses to run if its protocol file has changed (`protocol_check()`), so a gate cannot be softened after
the results are known.

## Rounds 6–9 (protocol files published in this repository)

The working copies of these protocols contained the owner's personal account size and country of residence.
For this public repository those details were redacted (the small-account scenario is a 50 USDT placeholder,
and country names were replaced with "the user's country"). Redaction changes the file bytes, so each protocol has two
fingerprints:

| Round | Topic | Protocol written (UTC) | Run once (UTC) | Original SHA-256 (private copy) | Public SHA-256 (file in `docs/`) | Outcome |
|---|---|---|---|---|---|---|
| 6 | Futures, leverage and shorting on the 4h breakout bots | 2026-10-15 ~11:05 | 2026-10-15 11:17 | `f22b0c2bc52a1842d701944063d09c47682116e2b2b5988bb1e6faced1020dc9` | `8deb024e2bf70914db46d8d093dd07d0bdf6e9583d9055d104143f3c365c1708` | No candidate passed |
| 7 | Drawdown-scaled risk (Grossman–Zhou) | 2026-10-16 ~08:40 | 2026-10-16 08:49 | `4deb40d32ca895460fe01963eecd5848fd346e9484f0e7f693d0a74670274647` | `cac5c2b1311201194acfe3e9b9b669257b060c2096a4540b4a45bd7e30de40bc` | No candidate passed |
| 8 | Multi-horizon trend ensemble with volatility targeting | 2026-10-17 ~21:55 | 2026-10-29 16:15 | `9469b6ca707f16236f17181859c8a6e26c3842a968d75988312ee0dd5eef34f2` | `aeade125c5f5d847d5febad9c9dc0c3b1367e7b56c4dfdad8bb8f0563b0de7d1` | No candidate passed (best risk/return so far) |
| 9 | Growth-optimal (Kelly) leverage | 2026-10-18 ~00:10 | 2026-10-29 16:15 | `392a3ac619c854ce9c04f0b725cf61a469d9d29a62828708218ab49c4d6fd15a` | `46bcf77905d7d4f18b0f119eef48d8c7e73e880a7684d189b93e65efc62628cc` | Leverage not recommended |

Rounds 8 and 9 were written eleven days before the data they were evaluated on was downloaded.

## Rounds 1–5

Rounds 1–5 were pre-registered in the private research journal (in Turkish) using the same discipline. Their
outcomes are summarised in the main README; their protocol texts are not published because they are interleaved
with personal operating notes.

## What "run once" means here

- The decision gates are evaluated exactly once on the pre-registered periods.
- A bug found later is fixed and the fix and its effect are written next to the result; parameters are never
  changed after seeing results.
- A strategy that *almost* passes is still a fail. Round 8's best candidate missed the "profit in every period"
  gate by 1.1 % and was not adopted.
