# Security Policy

## Reporting a vulnerability

If you've found a security issue in `pisama-detectors`, please do
**not** open a public GitHub issue. Instead:

- Email **security@pisama.ai** with a description, reproducer, and
  the affected version.
- We'll acknowledge within 2 business days and aim to ship a fix or
  mitigation within 7 business days for high-severity issues.

## What counts as a security issue

- Code execution via crafted detector input (e.g., `detect_injection`
  processing attacker-controlled text in a way that escapes the
  sandbox).
- Dependency vulnerabilities that affect the detector surface.
- Any path where a detector's return value could be manipulated by the
  input being analyzed to hide a failure from downstream consumers.

## Supported versions

Only the latest 0.x release line is supported. When 1.0 ships we'll
document an LTS policy.

## Credit

We'll credit reporters in release notes unless you prefer to stay
anonymous.
