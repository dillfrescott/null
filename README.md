# Null CAPTCHA

A lightweight, adaptive behavioral CAPTCHA service written in Rust. It combines short-lived signed proof-of-work challenges, mouse/touch/keyboard telemetry, a compact local transformer classifier, an accessible slider fallback, single-use validation tokens, and SQLite-backed replay protection.

The classifier continuously fine-tunes on recent, independently high-confidence interactions. Candidate models are checked for finite weights, at least 90% clean holdout accuracy, and no meaningful regression against the active model before they are atomically promoted and persisted. Low-confidence/self-confirming labels are excluded to limit feedback loops and training-data poisoning.

## Run locally

```bash
export NULL_CAPTCHA_SECRET='replace-with-at-least-32-random-characters'
cargo run --release
```

Open <http://localhost:3000>. The service stores state in `captcha.db` unless `DATABASE_URL` is set.

### Configuration

| Variable | Default | Description |
|---|---:|---|
| `PORT` | `3000` | HTTP listen port |
| `DATABASE_URL` | `captcha.db` | SQLite database path |
| `NULL_CAPTCHA_SECRET` | random per startup | HMAC key; set this in production |
| `NULL_CAPTCHA_MIN_SCORE` | `0.5` | Required score, from `0.0` through `1.0` |
| `NULL_CAPTCHA_DIFFICULTY` | `4.0` | PoW difficulty, from `1.0` through `6.0` |
| `URL` | unrestricted | Public URL; when set, its host is enforced |

## Integration

Load `/js/null.js`, render a widget, and send the returned token to your backend. Your backend must POST the token exactly once to `/api/validate`. See [`/llms.txt`](src/static/llms.txt) or the demo's **How to Use** dialog for examples.

## Container

```bash
docker compose up --build
```

Always configure `NULL_CAPTCHA_SECRET` through your runtime's secret manager in production. No secret is stored in `fly.toml`.

## Checks

```bash
cargo fmt --all -- --check
cargo clippy --all-targets --all-features -- -D warnings
cargo test --all-targets
```
