# syntax=docker/dockerfile:1
FROM rust:1.85-slim-bookworm AS builder

RUN apt-get update && apt-get install -y --no-install-recommends build-essential \
    && rm -rf /var/lib/apt/lists/*
WORKDIR /usr/src/null-captcha

# Cache dependency compilation separately from application code.
COPY Cargo.toml Cargo.lock ./
RUN mkdir src && printf 'fn main() {}\n' > src/main.rs \
    && cargo build --release --locked \
    && rm -rf src

COPY src ./src
RUN touch src/main.rs && cargo build --release --locked

FROM debian:bookworm-slim
RUN apt-get update && apt-get install -y --no-install-recommends ca-certificates \
    && rm -rf /var/lib/apt/lists/* \
    && mkdir -p /app/data

WORKDIR /app
COPY --from=builder /usr/src/null-captcha/target/release/null-captcha /app/null-captcha

ENV PORT=3000 \
    DATABASE_URL=/app/data/captcha.db \
    NULL_CAPTCHA_MIN_SCORE=0.5 \
    NULL_CAPTCHA_DIFFICULTY=4.0
EXPOSE 3000
STOPSIGNAL SIGTERM
CMD ["/app/null-captcha"]
