# Security Policy

`jax-evogym` is a research simulator and does not currently process private user
data or operate a hosted service. Security concerns are still welcome, especially
around package installation, dependency behavior, generated artifacts, or unsafe
file handling.

## Supported Versions

The current public target is the latest `main` branch and the latest published
`0.x` release, if one exists.

## Reporting

Please do not open a public issue for a vulnerability. Use GitHub's private
vulnerability reporting for this repository: open the **Security** tab, select
**Report a vulnerability**, and submit the report privately.

Include:

- affected version or commit
- reproduction steps
- expected impact
- whether the issue affects package installation, local execution, docs/site, or
  experiment tooling

## Scope

In scope:

- package installation or dependency-chain risk
- unsafe file reads/writes in the core package
- docs/designer behavior that could expose local files or execute untrusted code

Out of scope:

- expected behavior of user-authored Python code
- stochastic training failures
- performance regressions without a security impact
