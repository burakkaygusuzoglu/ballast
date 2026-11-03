# Security policy

## Reporting a vulnerability

Please do **not** open a public issue for security problems. Use GitHub's private
[security advisory form](https://github.com/burakkaygusuzoglu/ballast/security/advisories/new) instead.
You will get a reply as soon as possible.

## How the project handles secrets

- API keys are read only from a local `.env` file. Git ignores `.env`, and only `.env.example` is committed.
- Live mode refuses to start if the API key has withdrawal permission.
- Recommended key settings: read and spot trading only, withdrawals disabled, IP whitelist enabled.
- Never paste keys into issues, pull requests or chat tools.
