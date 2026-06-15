# Multi-stage build
# Stage 1: Build
FROM rust:1.85-slim-bookworm AS builder

# Install build dependencies (needed for compiling bundled SQLite)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /usr/src/null-captcha

# Create blank cargo project to cache dependencies
COPY Cargo.toml ./
RUN mkdir src && echo "fn main() {}" > src/main.rs
RUN cargo build --release
RUN rm -rf src

# Copy real source code
COPY src ./src

# Build for release (touch main.rs to force rebuild)
RUN touch src/main.rs && cargo build --release

# Stage 2: Final minimal image
FROM debian:bookworm-slim

# Install CA certificates
RUN apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy binary from builder
COPY --from=builder /usr/src/null-captcha/target/release/null-captcha /app/null-captcha

# Environment configurations
ENV PORT=3000
ENV DATABASE_URL=/app/data/captcha.db
ENV NULL_CAPTCHA_MIN_SCORE=0.5

# Expose port
EXPOSE 3000

# Create directory for stateful SQLite database persistence
RUN mkdir -p /app/data

# Run application
CMD ["/app/null-captcha"]
