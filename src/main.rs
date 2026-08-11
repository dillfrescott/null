use axum::{
    Json, Router,
    extract::{ConnectInfo, DefaultBodyLimit},
    http::{HeaderMap, HeaderValue, Method, StatusCode, header},
    response::{Html, IntoResponse, Response},
    routing::{get, post},
};
use base64::Engine;
use hmac::Mac;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::net::{IpAddr, SocketAddr};
use std::sync::Arc;
use tower_http::cors::{Any, CorsLayer};
use tracing::{Level, info, warn};
use tracing_subscriber::FmtSubscriber;

mod cache;
mod crypto;
mod db;
mod features;
mod model;

// Embed frontend assets directly into the binary
const INDEX_HTML: &str = include_str!("static/index.html");
const NULL_JS: &str = include_str!("static/null.js");
const LLMS_TXT: &str = include_str!("static/llms.txt");
const CHALLENGE_TTL_SECS: u64 = 300;
const MAX_TELEMETRY_POINTS: usize = 200;
const MAX_REQUEST_BYTES: usize = 64 * 1024;
const TELEMETRY_RETENTION_SECS: u64 = 7 * 24 * 60 * 60;

struct LogMessage {
    point_count: usize,
    score: f32,
    is_human: bool,
    webdriver: bool,
    user_agent: Option<String>,
    ip_address: Option<String>,
    features_json: String,
    points_hash: String,
    is_high_confidence: bool,
}

/// Simple in-memory rate limiter: per-IP sliding window tracker
struct RateLimiter {
    buckets: HashMap<String, Vec<u64>>,
    max_requests: usize,
    window_secs: u64,
}

impl RateLimiter {
    fn new(max_requests: usize, window_secs: u64) -> Self {
        RateLimiter {
            buckets: HashMap::new(),
            max_requests,
            window_secs,
        }
    }

    fn check_and_record(&mut self, ip: &str, now_secs: u64) -> bool {
        let cutoff = now_secs.saturating_sub(self.window_secs);
        if self.buckets.len() > 1_000 {
            self.remove_expired(now_secs);
        }
        let timestamps = self.buckets.entry(ip.to_string()).or_default();
        // Remove expired timestamps
        timestamps.retain(|&t| t > cutoff);
        if timestamps.len() >= self.max_requests {
            false
        } else {
            timestamps.push(now_secs);
            true
        }
    }

    fn remove_expired(&mut self, now_secs: u64) {
        let cutoff = now_secs.saturating_sub(self.window_secs);
        self.buckets.retain(|_, timestamps| {
            timestamps.retain(|&t| t > cutoff);
            !timestamps.is_empty()
        });
    }
}

/// Application state containing neural network, database connection, and security configurations
struct AppState {
    model: Arc<std::sync::RwLock<model::MiniTransformer>>,
    db_conn: std::sync::Mutex<rusqlite::Connection>,
    secret_key: Vec<u8>,
    min_score: f32,
    difficulty: f64,
    cache: cache::MemoryCache,
    log_tx: tokio::sync::mpsc::Sender<LogMessage>,
    expected_host: Option<String>,
    rate_limiter: std::sync::Mutex<RateLimiter>,
}

#[derive(Deserialize)]
struct VerifyPayload {
    payload: String,
    salt: String,
    signature: String,
    timestamp: u64,
    difficulty: f64,
    #[serde(rename = "encryptionKey")]
    encryption_key: String,
    nonce: u64,
    #[serde(default, rename = "sliderX")]
    slider_x: Option<i32>,
}

#[derive(Serialize)]
#[serde(rename_all = "camelCase")]
struct ChallengeResponse {
    salt: String,
    difficulty: f64,
    encryption_key: String,
    timestamp: u64,
    signature: String,
}

#[derive(Deserialize, Serialize)]
struct ClientScreen {
    w: i32,
    h: i32,
    ow: i32,
    oh: i32,
}

#[derive(Deserialize, Serialize)]
#[serde(rename_all = "camelCase")]
struct DecryptedPayload {
    salt: String,
    points: Vec<features::TelemetryPoint>,
    webdriver: bool,
    plugins: usize,
    languages: usize,
    screen: ClientScreen,
    time_taken: u64,
    #[serde(default)]
    accessibility_mode: bool,
}

#[derive(Serialize, Default)]
#[serde(rename_all = "camelCase")]
struct VerifyResponse {
    success: bool,
    score: f32,
    token: Option<String>,
    error: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    fallback_required: Option<bool>,
    #[serde(skip_serializing_if = "Option::is_none")]
    slider_target: Option<i32>,
}

#[derive(Deserialize)]
struct ValidatePayload {
    token: String,
}

#[derive(Serialize)]
struct ValidateResponse {
    success: bool,
    score: f32,
    error: Option<String>,
}

#[derive(Serialize)]
struct ErrorResponse {
    error: String,
}

#[tokio::main]
async fn main() {
    // Initialize tracing logger
    let subscriber = FmtSubscriber::builder()
        .with_max_level(Level::INFO)
        .finish();
    tracing::subscriber::set_global_default(subscriber).expect("setting default subscriber failed");

    info!("Initializing Null CAPTCHA Server...");

    // 1. Initialize SQLite Database (statefulness)
    let db_path = std::env::var("DATABASE_URL").unwrap_or_else(|_| "captcha.db".to_string());
    info!("Connecting to SQLite database at {}...", db_path);
    let db_conn = db::init_db(&db_path).expect("Failed to initialize SQLite database");

    // 2. Load or train the Neural Network model
    info!("Checking database for existing trained neural network weights...");
    let raw_model = match db::load_model(&db_conn) {
        Ok(Some(loaded)) if loaded.is_sane() => {
            info!("Loaded trained neural network weights from database.");
            loaded
        }
        Ok(Some(_)) => {
            warn!("Stored model contains invalid weights; rebuilding it.");
            let new_model = model::MiniTransformer::new_default();
            db::save_model(&db_conn, &new_model)
                .unwrap_or_else(|e| warn!("Failed to replace invalid model: {e}"));
            new_model
        }
        Err(error) => {
            warn!("Could not load the stored model ({error}); rebuilding it.");
            let new_model = model::MiniTransformer::new_default();
            db::save_model(&db_conn, &new_model)
                .unwrap_or_else(|e| warn!("Failed to save rebuilt model: {e}"));
            new_model
        }
        Ok(None) => {
            info!("No saved weights found. Training neural network from synthetic data...");
            let new_model = model::MiniTransformer::new_default();
            if let Err(e) = db::save_model(&db_conn, &new_model) {
                warn!("Failed to save trained neural network to database: {}", e);
            } else {
                info!("Saved trained neural network weights to database for persistence.");
            }
            new_model
        }
    };
    info!(
        "Mini Transformer model ready. (Input: 13 features, Embedding: 16, FF: 32, Heads: 2, Output: 1)"
    );

    let model = Arc::new(std::sync::RwLock::new(raw_model));

    // 3. Read Configuration
    let port = std::env::var("PORT")
        .unwrap_or_else(|_| "3000".to_string())
        .parse::<u16>()
        .expect("PORT must be a valid u16 port number");

    let secret_key = std::env::var("NULL_CAPTCHA_SECRET")
        .map(|s| {
            assert!(
                s.len() >= 32,
                "NULL_CAPTCHA_SECRET must contain at least 32 bytes"
            );
            s.into_bytes()
        })
        .unwrap_or_else(|_| {
            warn!("NULL_CAPTCHA_SECRET is not set; generating an ephemeral key.");
            warn!("Challenges and tokens will be invalidated when the server restarts.");
            let mut key = vec![0u8; 32];
            rand::Rng::fill(&mut rand::thread_rng(), &mut key[..]);
            key
        });

    let min_score = parse_bounded_env_f32("NULL_CAPTCHA_MIN_SCORE", 0.5, 0.0, 1.0);
    let difficulty = parse_bounded_env_f64("NULL_CAPTCHA_DIFFICULTY", 4.0, 1.0, 6.0);

    // Initialize Memory Cache for token and telemetry replay checks
    let cache = cache::MemoryCache::new();

    // Setup asynchronous background telemetry logger to prevent SQLite write lock blocks
    let (log_tx, mut log_rx) = tokio::sync::mpsc::channel::<LogMessage>(5000);
    let db_path_clone = db_path.clone();
    let model_clone = Arc::clone(&model);
    tokio::task::spawn_blocking(move || {
        match rusqlite::Connection::open(&db_path_clone) {
            Ok(conn) => {
                // Configure WAL and busy timeout for the background connection as well
                let _ = conn.execute_batch(
                    "PRAGMA journal_mode = WAL;
                 PRAGMA synchronous = NORMAL;
                 PRAGMA busy_timeout = 5000;",
                );

                if let Err(error) = db::prune_old_tokens(&conn, CHALLENGE_TTL_SECS) {
                    warn!("Failed to prune old tokens: {error}");
                }
                if let Err(error) = db::prune_old_telemetry(&conn, TELEMETRY_RETENTION_SECS) {
                    warn!("Failed to prune old telemetry: {error}");
                }

                let mut last_prune = std::time::Instant::now();
                let mut logged_interactions_count = 0;

                while let Some(msg) = log_rx.blocking_recv() {
                    let logged = db::log_telemetry(
                        &conn,
                        &db::TelemetryRecord {
                            point_count: msg.point_count,
                            score: msg.score,
                            is_human: msg.is_human,
                            webdriver: msg.webdriver,
                            user_agent: msg.user_agent.as_deref(),
                            ip_address: msg.ip_address.as_deref(),
                            features_json: &msg.features_json,
                            points_hash: &msg.points_hash,
                            is_high_confidence: msg.is_high_confidence,
                        },
                    );
                    if let Err(error) = &logged {
                        warn!("Failed to persist telemetry: {error}");
                    }

                    // Count only durable, independently high-confidence labels.
                    if logged.is_ok() && !msg.features_json.is_empty() && msg.is_high_confidence {
                        logged_interactions_count += 1;
                    }

                    // Periodically retrain and adapt the model (every 50 new logged interactions)
                    if logged_interactions_count >= 50 {
                        info!(
                            "Adapting neural network: retraining model on recent telemetry logs + synthetic dataset..."
                        );

                        // Fetch recent logs
                        match db::get_recent_telemetry(&conn, 500) {
                            Ok(mut recent_data) => {
                                info!(
                                    "Fetched {} recent telemetry samples from database.",
                                    recent_data.len()
                                );

                                // Generate synthetic dataset
                                let mut synthetic_data =
                                    model::MiniTransformer::generate_synthetic_dataset();

                                // Shuffle synthetic dataset and split it
                                use rand::seq::SliceRandom;
                                let mut rng = rand::thread_rng();
                                synthetic_data.shuffle(&mut rng);

                                // Reserve 15% of the clean synthetic dataset strictly for validation.
                                // This validation set is never mixed with recent real-world telemetry,
                                // ensuring the model is always gated against clean baseline behavior.
                                let val_size = synthetic_data.len() * 15 / 100;
                                let val_dataset = synthetic_data[0..val_size].to_vec();

                                // The training dataset consists of the rest of the synthetic data + recent real telemetry
                                let mut train_dataset = synthetic_data[val_size..].to_vec();
                                train_dataset.append(&mut recent_data);
                                train_dataset.shuffle(&mut rng);

                                // Fine-tune the live model rather than starting over, preserving
                                // its learned boundary while incorporating recent behavior.
                                let current_model = model_clone
                                    .read()
                                    .unwrap_or_else(|e| e.into_inner())
                                    .clone();
                                let baseline_accuracy = current_model.validate(&val_dataset);
                                let mut new_model = current_model;
                                let loss = new_model.train(&train_dataset, 40, 0.005);

                                // Promote only numerically sane candidates that retain the clean
                                // baseline. This prevents drift and limits poisoning damage.
                                let accuracy = new_model.validate(&val_dataset);
                                let is_sane = new_model.is_sane() && loss.is_finite();
                                let retained_baseline = accuracy + 0.005 >= baseline_accuracy;

                                if is_sane && accuracy >= 0.90 && retained_baseline {
                                    info!(
                                        "Model retraining succeeded. Loss: {:.6}, Validation Accuracy: {:.2}%",
                                        loss,
                                        accuracy * 100.0
                                    );

                                    // Swap the model under a brief write lock (minimal block time)
                                    {
                                        let mut model_guard =
                                            model_clone.write().unwrap_or_else(|e| e.into_inner());
                                        *model_guard = new_model;
                                    }

                                    // Save the updated model weights back to the database
                                    {
                                        let model_guard =
                                            model_clone.read().unwrap_or_else(|e| e.into_inner());
                                        if let Err(e) = db::save_model(&conn, &model_guard) {
                                            warn!(
                                                "Failed to save adapted model weights to database: {}",
                                                e
                                            );
                                        } else {
                                            info!(
                                                "Adapted model weights successfully saved/persisted to database."
                                            );
                                        }
                                    }
                                } else {
                                    warn!(
                                        "Rejected adaptive model: sane={}, candidate accuracy={:.2}%, baseline={:.2}%",
                                        is_sane,
                                        accuracy * 100.0,
                                        baseline_accuracy * 100.0
                                    );
                                }
                            }
                            Err(e) => {
                                warn!("Failed to load telemetry for adaptation: {}", e);
                            }
                        }
                        logged_interactions_count = 0;
                    }

                    // Periodically prune tokens every 5 minutes
                    if last_prune.elapsed().as_secs() > CHALLENGE_TTL_SECS {
                        if let Err(error) = db::prune_old_tokens(&conn, CHALLENGE_TTL_SECS) {
                            warn!("Failed to prune old tokens: {error}");
                        }
                        if let Err(error) = db::prune_old_telemetry(&conn, TELEMETRY_RETENTION_SECS)
                        {
                            warn!("Failed to prune old telemetry: {error}");
                        }
                        last_prune = std::time::Instant::now();
                    }
                }
            }
            Err(error) => warn!("Background database connection failed: {error}"),
        }
    });

    // Initialize expected host from URL environment variable if set
    let expected_host = std::env::var("URL").ok().map(|url| {
        url.parse::<axum::http::Uri>()
            .ok()
            .and_then(|uri| {
                uri.authority()
                    .map(|authority| authority.host().to_lowercase())
            })
            .filter(|host| !host.is_empty())
            .unwrap_or_else(|| panic!("URL must be an absolute URL with a valid host"))
    });

    if let Some(ref host) = expected_host {
        info!(
            "URL restriction active: only allowing access via domain '{}'",
            host
        );
    }

    let state = Arc::new(AppState {
        model,
        db_conn: std::sync::Mutex::new(db_conn),
        secret_key,
        min_score,
        difficulty,
        cache,
        log_tx,
        expected_host,
        rate_limiter: std::sync::Mutex::new(RateLimiter::new(30, 60)),
    });

    // 4. Setup router with CORS and host restriction middleware
    let app = Router::new()
        .route("/", get(serve_index))
        .route("/js/null.js", get(serve_js))
        .route("/llms.txt", get(serve_llms))
        .route("/healthz", get(health_handler))
        .route("/api/challenge", get(challenge_handler))
        .route("/api/verify", post(verify_handler))
        .route("/api/validate", post(validate_handler))
        .layer(axum::middleware::from_fn_with_state(
            Arc::clone(&state),
            host_guard_middleware,
        ))
        .layer(DefaultBodyLimit::max(MAX_REQUEST_BYTES))
        .layer(
            CorsLayer::new()
                .allow_origin(Any)
                .allow_methods([Method::GET, Method::POST])
                .allow_headers([header::CONTENT_TYPE]),
        )
        .with_state(state);

    // 5. Bind and run server
    let addr = SocketAddr::from(([0, 0, 0, 0], port));
    let listener = tokio::net::TcpListener::bind(addr)
        .await
        .unwrap_or_else(|e| panic!("Failed to bind to port {}: {}", port, e));

    info!("Null CAPTCHA server running at http://localhost:{}", port);
    axum::serve(
        listener,
        app.into_make_service_with_connect_info::<SocketAddr>(),
    )
    .with_graceful_shutdown(shutdown_signal())
    .await
    .expect("HTTP server failed");
}

fn parse_bounded_env_f32(name: &str, default: f32, min: f32, max: f32) -> f32 {
    let value = std::env::var(name)
        .map(|raw| {
            raw.parse::<f32>()
                .unwrap_or_else(|_| panic!("{name} must be a number"))
        })
        .unwrap_or(default);
    assert!(
        value.is_finite() && (min..=max).contains(&value),
        "{name} must be between {min} and {max}"
    );
    value
}

fn parse_bounded_env_f64(name: &str, default: f64, min: f64, max: f64) -> f64 {
    let value = std::env::var(name)
        .map(|raw| {
            raw.parse::<f64>()
                .unwrap_or_else(|_| panic!("{name} must be a number"))
        })
        .unwrap_or(default);
    assert!(
        value.is_finite() && (min..=max).contains(&value),
        "{name} must be between {min} and {max}"
    );
    value
}

async fn shutdown_signal() {
    let ctrl_c = async {
        tokio::signal::ctrl_c()
            .await
            .expect("failed to install Ctrl+C handler");
    };

    #[cfg(unix)]
    let terminate = async {
        tokio::signal::unix::signal(tokio::signal::unix::SignalKind::terminate())
            .expect("failed to install SIGTERM handler")
            .recv()
            .await;
    };
    #[cfg(not(unix))]
    let terminate = std::future::pending::<()>();

    tokio::select! {
        () = ctrl_c => {},
        () = terminate => {},
    }
    info!("Shutdown signal received");
}

async fn health_handler() -> StatusCode {
    StatusCode::NO_CONTENT
}

/// Serve the demo landing page.
async fn serve_index() -> impl IntoResponse {
    Html(INDEX_HTML)
}

/// Serve the telemetry tracker JavaScript library with correct content type
async fn serve_js() -> impl IntoResponse {
    Response::builder()
        .header(
            header::CONTENT_TYPE,
            HeaderValue::from_static("application/javascript"),
        )
        .body(NULL_JS.to_string())
        .unwrap()
}

/// Serve the llms.txt guide for AI agents
async fn serve_llms() -> impl IntoResponse {
    Response::builder()
        .header(
            header::CONTENT_TYPE,
            HeaderValue::from_static("text/plain; charset=utf-8"),
        )
        .body(LLMS_TXT.to_string())
        .unwrap()
}

fn verify_pow(salt: &str, nonce: u64, difficulty: f64) -> bool {
    if !difficulty.is_finite() || !(1.0..=6.0).contains(&difficulty) {
        return false;
    }

    use sha2::{Digest, Sha256};
    let mut hasher = Sha256::new();
    hasher.update(format!("{}{}", salt, nonce).as_bytes());
    let hash_bytes = hasher.finalize();

    let required_zero_bits = (difficulty * 4.0).round() as usize;
    let mut zero_bits = 0;
    for &byte in hash_bytes.iter() {
        let leading = byte.leading_zeros() as usize;
        zero_bits += leading;
        if leading < 8 {
            break;
        }
    }
    zero_bits >= required_zero_bits
}

fn derive_slider_target(salt: &str, secret: &[u8]) -> i32 {
    use sha2::{Digest, Sha256};
    let mut hasher = Sha256::new();
    hasher.update(salt.as_bytes());
    hasher.update(secret);
    let result = hasher.finalize();
    let val = u32::from_be_bytes([result[0], result[1], result[2], result[3]]);
    (50 + (val % 200)) as i32
}

async fn challenge_handler(
    axum::extract::State(state): axum::extract::State<Arc<AppState>>,
    headers: HeaderMap,
    ConnectInfo(addr): ConnectInfo<SocketAddr>,
) -> Response {
    use rand::Rng;
    use rand::distributions::Alphanumeric;

    let now = unix_time_secs();

    let salt: String = rand::thread_rng()
        .sample_iter(&Alphanumeric)
        .take(16)
        .map(char::from)
        .collect();

    let encryption_key: String = rand::thread_rng()
        .sample_iter(&Alphanumeric)
        .take(32)
        .map(char::from)
        .collect();

    let ip = client_ip(&headers, addr);
    {
        let mut limiter = state.rate_limiter.lock().unwrap_or_else(|e| e.into_inner());
        if !limiter.check_and_record(&format!("challenge:{ip}"), now) {
            return (
                StatusCode::TOO_MANY_REQUESTS,
                Json(ErrorResponse {
                    error: "Too many challenge requests. Please slow down.".to_string(),
                }),
            )
                .into_response();
        }
    }

    let difficulty = state.difficulty;
    let sign_payload = format!("{}.{}.{}.{}", now, salt, difficulty, encryption_key);

    let mut mac =
        hmac::Hmac::<sha2::Sha256>::new_from_slice(&state.secret_key).expect("HMAC sign failed");
    mac.update(sign_payload.as_bytes());
    let signature =
        base64::engine::general_purpose::URL_SAFE_NO_PAD.encode(mac.finalize().into_bytes());

    let mut response = Json(ChallengeResponse {
        salt,
        difficulty,
        encryption_key,
        timestamp: now,
        signature,
    })
    .into_response();
    response.headers_mut().insert(
        header::CACHE_CONTROL,
        HeaderValue::from_static("no-store, max-age=0"),
    );
    response
}

/// API endpoint to verify client-side telemetry
async fn verify_handler(
    axum::extract::State(state): axum::extract::State<Arc<AppState>>,
    headers: HeaderMap,
    ConnectInfo(addr): ConnectInfo<SocketAddr>,
    Json(payload): Json<VerifyPayload>,
) -> impl IntoResponse {
    if payload.salt.len() != 16
        || payload.encryption_key.len() != 32
        || payload.signature.len() > 64
        || payload.payload.is_empty()
        || !payload.difficulty.is_finite()
        || !(1.0..=6.0).contains(&payload.difficulty)
        || payload.slider_x.is_some_and(|x| !(20..=280).contains(&x))
    {
        return Json(VerifyResponse {
            error: Some("Invalid challenge data.".to_string()),
            ..Default::default()
        });
    }

    // Verify the signed challenge before performing expensive work.
    let signed_data = format!(
        "{}.{}.{}.{}",
        payload.timestamp, payload.salt, payload.difficulty, payload.encryption_key
    );
    let signature = match base64::engine::general_purpose::URL_SAFE_NO_PAD
        .decode(payload.signature.as_bytes())
    {
        Ok(signature) => signature,
        Err(_) => {
            return Json(VerifyResponse {
                error: Some("Invalid challenge signature. Re-verification required.".to_string()),
                ..Default::default()
            });
        }
    };
    let mut mac = hmac::Hmac::<sha2::Sha256>::new_from_slice(&state.secret_key)
        .expect("HMAC verification setup failed");
    mac.update(signed_data.as_bytes());
    if mac.verify_slice(&signature).is_err() {
        warn!("Verification challenge signature mismatch");
        return Json(VerifyResponse {
            error: Some("Invalid challenge signature. Re-verification required.".to_string()),
            ..Default::default()
        });
    }

    let now = unix_time_secs();
    if now < payload.timestamp || now - payload.timestamp > CHALLENGE_TTL_SECS {
        return Json(VerifyResponse {
            error: Some("Verification challenge expired. Please retry.".to_string()),
            ..Default::default()
        });
    }

    let ip_str = client_ip(&headers, addr);
    {
        let mut limiter = state.rate_limiter.lock().unwrap_or_else(|e| e.into_inner());
        if !limiter.check_and_record(&format!("verify:{ip_str}"), now) {
            warn!("Rate limit exceeded for IP: {ip_str}");
            return Json(VerifyResponse {
                error: Some("Too many requests. Please slow down.".to_string()),
                ..Default::default()
            });
        }
    }

    // Atomically allow one passive attempt and, if requested, one fallback attempt.
    let consumed_key = format!("salt:consumed:{}", payload.salt);
    let fallback_key = format!("salt:fallback:{}", payload.salt);
    if state.cache.contains(&consumed_key) {
        return Json(VerifyResponse {
            error: Some("Challenge already completed. Re-verification required.".to_string()),
            ..Default::default()
        });
    }

    if payload.slider_x.is_some() {
        let slider_attempt_key = format!("salt:slider-attempt:{}", payload.salt);
        if !state.cache.contains(&fallback_key)
            || !state
                .cache
                .check_and_insert(slider_attempt_key, CHALLENGE_TTL_SECS)
        {
            return Json(VerifyResponse {
                error: Some("Invalid challenge sequence. Re-verification required.".to_string()),
                ..Default::default()
            });
        }
        state.cache.remove(&fallback_key);
        state.cache.insert(consumed_key.clone(), CHALLENGE_TTL_SECS);
    } else {
        let passive_attempt_key = format!("salt:passive-attempt:{}", payload.salt);
        if !state
            .cache
            .check_and_insert(passive_attempt_key, CHALLENGE_TTL_SECS)
        {
            return Json(VerifyResponse {
                error: Some("Challenge already attempted. Re-verification required.".to_string()),
                fallback_required: Some(state.cache.contains(&fallback_key)),
                ..Default::default()
            });
        }
    }

    if !verify_pow(&payload.salt, payload.nonce, payload.difficulty) {
        warn!("Verification PoW mismatch for salt {}", payload.salt);
        return Json(VerifyResponse {
            error: Some("Invalid Proof of Work solution.".to_string()),
            ..Default::default()
        });
    }

    // Decrypt and parse the telemetry payload.
    let decrypted_str =
        match crypto::decrypt_payload(&payload.payload, &payload.salt, &payload.encryption_key) {
            Ok(s) => s,
            Err(e) => {
                warn!("Payload decryption failed: {}", e);
                return Json(VerifyResponse {
                    error: Some(
                        "Payload validation failed. Ensure JS client is up to date.".to_string(),
                    ),
                    ..Default::default()
                });
            }
        };

    let client_data: DecryptedPayload = match serde_json::from_str(&decrypted_str) {
        Ok(d) => d,
        Err(e) => {
            warn!("Failed to parse decrypted JSON: {}", e);
            return Json(VerifyResponse {
                error: Some("Telemetry structure mismatch.".to_string()),
                ..Default::default()
            });
        }
    };

    // Verify cryptographic salt binding to prevent cross-challenge replay attacks
    if client_data.salt != payload.salt {
        warn!("Verification failed: Telemetry salt mismatch");
        return Json(VerifyResponse {
            error: Some("Telemetry payload is not bound to this challenge.".to_string()),
            ..Default::default()
        });
    }

    let invalid_points = client_data.points.len() > MAX_TELEMETRY_POINTS
        || client_data.points.iter().any(|point| {
            !point.x.is_finite()
                || !point.y.is_finite()
                || !point.t.is_finite()
                || point.x.abs() > 1_000_000.0
                || point.y.abs() > 1_000_000.0
                || !(0.0..=600_000.0).contains(&point.t)
        })
        || client_data
            .points
            .windows(2)
            .any(|pair| pair[1].t < pair[0].t);
    let invalid_screen = client_data.screen.w <= 0
        || client_data.screen.h <= 0
        || client_data.screen.ow < 0
        || client_data.screen.oh < 0
        || [
            client_data.screen.w,
            client_data.screen.h,
            client_data.screen.ow,
            client_data.screen.oh,
        ]
        .into_iter()
        .any(|dimension| dimension > 100_000);
    if invalid_points
        || invalid_screen
        || client_data.time_taken > 600_000
        || client_data.plugins > 10_000
        || client_data.languages > 1_000
        || (client_data.accessibility_mode && payload.slider_x.is_none())
    {
        warn!("Rejected malformed telemetry payload");
        return Json(VerifyResponse {
            error: Some("Telemetry values are out of range.".to_string()),
            ..Default::default()
        });
    }

    // Verify telemetry points are unique to prevent replay of the same movements
    let points_hash = compute_points_hash(&client_data.points);
    if !client_data.points.is_empty() {
        // Check in-memory cache first for speed
        if !state
            .cache
            .check_and_insert(format!("telemetry:{}", points_hash), 300)
        {
            warn!("Verification failed: Replayed telemetry detected in cache!");
            return Json(VerifyResponse {
                error: Some("Automated behavior detected (replayed telemetry).".to_string()),
                ..Default::default()
            });
        }

        // Check persistent database for replay checks
        let state_clone = Arc::clone(&state);
        let hash_clone = points_hash.clone();
        let is_replay = tokio::task::spawn_blocking(move || {
            let conn = state_clone
                .db_conn
                .lock()
                .unwrap_or_else(|e| e.into_inner());
            db::is_telemetry_replayed(&conn, &hash_clone).unwrap_or_else(|error| {
                warn!("Telemetry replay lookup failed: {error}");
                true
            })
        })
        .await
        .unwrap_or(true);
        if is_replay {
            warn!("Verification failed: Replayed telemetry detected in database!");
            return Json(VerifyResponse {
                error: Some("Automated behavior detected (replayed telemetry).".to_string()),
                ..Default::default()
            });
        }
    }

    // Determine verification based on telemetry score or slider fallback
    let mut is_human = false;
    let mut score = 0.0f32;
    let mut bot_flag = false;

    // Apply anti-bot fingerprint rules (Bot Proof checks)
    if client_data.webdriver {
        warn!("Automation detected: navigator.webdriver is true");
        bot_flag = true;
    }
    if client_data.time_taken < 200 {
        warn!(
            "Automation detected: click happened too fast ({}ms)",
            client_data.time_taken
        );
        bot_flag = true;
    }
    // Some legitimate mobile/PWA browsers report zero outer dimensions, so
    // inner dimensions are the authoritative sanity check above.

    // Check slider fallback puzzle solve status
    let mut solved_slider = false;
    let mut is_accessibility = false;
    let target = derive_slider_target(&payload.salt, &state.secret_key);

    let slider_aligned = payload
        .slider_x
        .is_some_and(|x| (i64::from(x) - i64::from(target)).abs() <= 6);
    if slider_aligned {
        solved_slider = true;
        is_accessibility = client_data.accessibility_mode;
    }

    // Filter out keyboard/accessibility events (y == -1.0) for mouse telemetry extraction
    let mouse_points: Vec<features::TelemetryPoint> = client_data
        .points
        .iter()
        .filter(|p| (p.y - -1.0).abs() > 0.001)
        .cloned()
        .collect();

    // 1. Passive checking (mouse pointer telemetry)
    let features_opt = features::extract_features(&mouse_points);

    let mut features_json = String::new();
    if let Some(ref features) = features_opt {
        features_json = serde_json::to_string(&features).unwrap_or_default();
        let feature_arr = features.to_array();
        let mut model_score = state
            .model
            .read()
            .unwrap_or_else(|e| e.into_inner())
            .predict(&feature_arr);

        if client_data.plugins == 0 {
            model_score = (model_score - 0.25).max(0.0);
        }

        score = model_score;

        // Only allow a passive check pass if the active slider fallback challenge is not being submitted.
        // If they are submitting the slider, they must pass the solved_slider check in step 2.
        if !bot_flag && score >= state.min_score && payload.slider_x.is_none() {
            is_human = true;
        }
    }

    // 2. Active fallback check: If passive telemetry failed or was insufficient, check slider puzzle solution
    if !is_human && !bot_flag && solved_slider {
        if is_accessibility {
            // Keyboard / accessibility user successfully aligned the slider.
            // Verify that keyboard telemetry looks human to prevent simple script bypass.
            let kb_points: Vec<&features::TelemetryPoint> = client_data
                .points
                .iter()
                .filter(|p| (p.y - -1.0).abs() < 0.001)
                .collect();

            if kb_points.is_empty() {
                warn!(
                    "Automation detected: accessibility mode claimed but no keyboard events recorded."
                );
                bot_flag = true;
            } else {
                // Check for automated rapid keystrokes (less than 15ms interval)
                let mut too_fast = false;
                for i in 0..(kb_points.len() - 1) {
                    let dt = kb_points[i + 1].t - kb_points[i].t;
                    if dt < 15.0 {
                        too_fast = true;
                        break;
                    }
                }
                if too_fast {
                    warn!(
                        "Automation detected: keyboard events are too fast (less than 15ms interval)"
                    );
                    bot_flag = true;
                } else {
                    // Check if the number of keypresses matches target distance roughly
                    // Initial slider position is 25. Each ArrowRight/ArrowLeft moves by 5.
                    let target_x = payload.slider_x.unwrap_or(25);
                    let expected_presses =
                        ((i64::from(target_x) - 25).unsigned_abs() as f32 / 5.0).ceil() as usize;

                    // Allow some flexibility but ensure they did at least half of the expected keystrokes
                    if kb_points.len() < expected_presses / 2 {
                        warn!(
                            "Automation detected: accessibility keypress count {} too low for target distance (expected ~{})",
                            kb_points.len(),
                            expected_presses
                        );
                        bot_flag = true;
                    } else {
                        score = 1.0;
                        is_human = true;
                        info!(
                            "Accessibility validation succeeded via slider puzzle ({} keypresses).",
                            kb_points.len()
                        );
                    }
                }
            }
        } else {
            // Mouse user aligned the slider. We check if the drag gesture itself is human.
            let mut valid_drag_dist = true;
            let xs: Vec<f32> = mouse_points.iter().map(|p| p.x).collect();
            if !xs.is_empty() {
                let min_x = xs.iter().copied().fold(f32::INFINITY, f32::min);
                let max_x = xs.iter().copied().fold(f32::NEG_INFINITY, f32::max);
                let span_x = max_x - min_x;
                let claimed_drag_dist =
                    (i64::from(payload.slider_x.unwrap_or(25)) - 25).unsigned_abs() as f32;

                if span_x < claimed_drag_dist - 15.0 {
                    warn!(
                        "Automation detected: horizontal mouse movement span ({:.2}px) is less than claimed drag distance ({:.2}px)",
                        span_x, claimed_drag_dist
                    );
                    valid_drag_dist = false;
                }
            } else {
                valid_drag_dist = false;
            }

            if valid_drag_dist {
                if let Some(features) = features::extract_features(&mouse_points) {
                    let feature_arr = features.to_array();
                    let drag_score = state
                        .model
                        .read()
                        .unwrap_or_else(|e| e.into_inner())
                        .predict(&feature_arr);
                    if drag_score >= 0.25 {
                        score = drag_score;
                        is_human = true;
                        info!(
                            "Slider puzzle verification succeeded with drag score: {:.4}",
                            drag_score
                        );
                    } else {
                        warn!(
                            "Slider puzzle verification failed: drag score {:.4} below human threshold",
                            drag_score
                        );
                    }
                }
            } else {
                bot_flag = true;
            }
        }
    }

    // Get client details for logging
    let user_agent = headers
        .get(header::USER_AGENT)
        .and_then(|value| value.to_str().ok())
        .map(|value| value.chars().take(256).collect::<String>());
    // Persist only a keyed pseudonymous identifier, never the raw client IP.
    let ip_address = hash_identifier(&state.secret_key, &ip_str);

    let is_high_confidence = if bot_flag {
        // Only mark bot as high confidence for retraining if the telemetry features themselves
        // look automated (e.g., straight line, low speed variance/jitter), to prevent poisoning
        // by attackers spoofing browser flags with human movements.
        if let Some(ref features) = features_opt {
            features.speed_var < 0.35 && features.angular_jitter < 0.05
        } else {
            false
        }
    } else {
        if is_human {
            // Only mark as high confidence human if mouse features exhibit typical human variance
            if let Some(ref features) = features_opt {
                if payload.slider_x.is_none() {
                    // Passive check: human movements must be curved, with speed variance, duration and entropy
                    features.speed_var >= 0.08 &&
                    features.total_duration >= 0.075 && // 0.075 is 150ms normalized (duration / 2000.0)
                    features.entropy >= 0.10 &&
                    features.straightness < 0.95 &&
                    (features.angular_jitter >= 0.005 || features.line_deviation >= 0.001)
                } else {
                    // Active slider drag check: straight path but with human speed/jitter micro-adjustments
                    features.speed_var >= 0.08
                        && features.total_duration >= 0.075
                        && features.entropy >= 0.02
                        && (features.angular_jitter >= 0.001 || features.line_deviation >= 0.0005)
                }
            } else {
                false
            }
        } else {
            // Never learn a bot label merely because the current model rejected it;
            // that feedback loop would reinforce mistakes and is easy to poison.
            // Admit only independently bot-like movement into adaptive training.
            features_opt.as_ref().is_some_and(|features| {
                features.speed_var < 0.05
                    && features.angular_jitter < 0.01
                    && features.timing_jitter < 0.05
                    && features.entropy < 0.08
            })
        }
    };

    // Send log message to background channel for async persistence (non-blocking)
    let log_msg = LogMessage {
        point_count: client_data.points.len(),
        score,
        is_human,
        webdriver: client_data.webdriver,
        user_agent,
        ip_address: Some(ip_address),
        features_json,
        points_hash,
        is_high_confidence,
    };
    let _ = state.log_tx.try_send(log_msg);

    info!(
        "Verification result: Points: {}, Score: {:.4} ({}), Webdriver: {}, Time: {}ms, SolvedSlider: {}, Accessibility: {}",
        client_data.points.len(),
        score,
        if is_human { "Human" } else { "Bot" },
        client_data.webdriver,
        client_data.time_taken,
        solved_slider,
        is_accessibility
    );

    // 5. Generate token if verification succeeds, or request fallback if telemetry failed
    if is_human {
        // Mark challenge salt as fully consumed
        state.cache.insert(consumed_key, CHALLENGE_TTL_SECS);

        let token = crypto::generate_token(&state.secret_key, score);
        Json(VerifyResponse {
            success: true,
            score,
            token: Some(token),
            error: None,
            fallback_required: None,
            slider_target: None,
        })
    } else {
        // If they haven't tried the slider yet and failed passive check, prompt for fallback
        let fallback_needed = payload.slider_x.is_none() && !bot_flag;
        let mut target_val = None;
        if fallback_needed {
            // Passive check failed, transition to fallback state
            state.cache.insert(fallback_key, CHALLENGE_TTL_SECS);
            target_val = Some(derive_slider_target(&payload.salt, &state.secret_key));
        } else {
            // Obvious bot or failed slider check, mark as consumed
            state.cache.insert(consumed_key, CHALLENGE_TTL_SECS);
        }

        Json(VerifyResponse {
            success: false,
            score,
            token: None,
            error: Some(if fallback_needed {
                "Behavioral analysis uncertain. Fallback challenge required.".to_string()
            } else {
                "Telemetry profile classified as automated bot behavior.".to_string()
            }),
            fallback_required: Some(fallback_needed),
            slider_target: target_val,
        })
    }
}

/// API endpoint for backend servers to validate user verification tokens
async fn validate_handler(
    axum::extract::State(state): axum::extract::State<Arc<AppState>>,
    Json(payload): Json<ValidatePayload>,
) -> impl IntoResponse {
    // Verify token signature and timestamp before cache and database operations
    // to prevent database/cache pollution from invalid token spam.
    let score = match crypto::verify_token(&state.secret_key, &payload.token, CHALLENGE_TTL_SECS) {
        Ok(score) => score,
        Err(e) => {
            warn!("Token verification failed: {}", e);
            return Json(ValidateResponse {
                success: false,
                score: 0.0,
                error: Some(e),
            });
        }
    };

    // Store only a digest in replay-protection state so tokens are not retained in plaintext.
    let token_hash = hash_identifier(&state.secret_key, &payload.token);
    let cache_key = format!("token:{token_hash}");
    if !state
        .cache
        .check_and_insert(cache_key.clone(), CHALLENGE_TTL_SECS)
    {
        warn!("Replay validation check: Token has already been used (cache).");
        return Json(ValidateResponse {
            success: false,
            score: 0.0,
            error: Some("Token has already been validated.".to_string()),
        });
    }

    // Mark token as used in database atomically to prevent race conditions
    let state_clone = Arc::clone(&state);
    let mark_result = tokio::task::spawn_blocking(move || {
        let conn = state_clone
            .db_conn
            .lock()
            .unwrap_or_else(|e| e.into_inner());
        db::try_mark_token_used(&conn, &token_hash)
    })
    .await;
    match mark_result {
        Ok(Ok(true)) => {}
        Ok(Ok(false)) => {
            warn!("Replay validation check: token has already been used (database)");
            return Json(ValidateResponse {
                success: false,
                score: 0.0,
                error: Some("Token has already been validated.".to_string()),
            });
        }
        Ok(Err(error)) => {
            state.cache.remove(&cache_key);
            warn!("Could not persist token validation: {error}");
            return Json(ValidateResponse {
                success: false,
                score: 0.0,
                error: Some("Token validation is temporarily unavailable.".to_string()),
            });
        }
        Err(error) => {
            state.cache.remove(&cache_key);
            warn!("Token validation task failed: {error}");
            return Json(ValidateResponse {
                success: false,
                score: 0.0,
                error: Some("Token validation is temporarily unavailable.".to_string()),
            });
        }
    }

    // 3. Check if validation score meets human threshold
    if score >= state.min_score {
        info!("Token validation succeeded. Score: {:.4}", score);
        Json(ValidateResponse {
            success: true,
            score,
            error: None,
        })
    } else {
        warn!(
            "Token validation failed: Score {:.4} below threshold",
            score
        );
        Json(ValidateResponse {
            success: false,
            score: 0.0,
            error: Some("Token score does not meet human threshold.".to_string()),
        })
    }
}

fn compute_points_hash(points: &[features::TelemetryPoint]) -> String {
    use sha2::{Digest, Sha256};
    let mut hasher = Sha256::new();
    hasher.update((points.len() as u64).to_be_bytes());
    for point in points {
        hasher.update(point.x.to_bits().to_be_bytes());
        hasher.update(point.y.to_bits().to_be_bytes());
        hasher.update(point.t.to_bits().to_be_bytes());
    }
    format!("{:x}", hasher.finalize())
}

fn hash_identifier(secret: &[u8], value: &str) -> String {
    let mut mac =
        hmac::Hmac::<sha2::Sha256>::new_from_slice(secret).expect("HMAC identifier setup failed");
    mac.update(value.as_bytes());
    base64::engine::general_purpose::URL_SAFE_NO_PAD.encode(mac.finalize().into_bytes())
}

fn unix_time_secs() -> u64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs()
}

fn client_ip(headers: &HeaderMap, peer: SocketAddr) -> String {
    headers
        .get("fly-client-ip")
        .and_then(|value| value.to_str().ok())
        .and_then(|value| value.parse::<IpAddr>().ok())
        .unwrap_or_else(|| peer.ip())
        .to_string()
}

/// Middleware to restrict access only to the domain specified in the URL environment variable
async fn host_guard_middleware(
    axum::extract::State(state): axum::extract::State<Arc<AppState>>,
    request: axum::extract::Request,
    next: axum::middleware::Next,
) -> Result<axum::response::Response, axum::http::StatusCode> {
    // Fly's health checker addresses the machine directly, so keep this
    // non-sensitive endpoint independent of the public host restriction.
    if request.uri().path() != "/healthz" {
        if let Some(ref expected) = state.expected_host {
            let host_header = request.headers().get(header::HOST);
            let request_host = host_header
                .and_then(|value| value.to_str().ok())
                .and_then(|value| value.parse::<axum::http::uri::Authority>().ok())
                .map(|authority| authority.host().to_lowercase());

            if request_host.as_deref() != Some(expected.as_str()) {
                warn!("Forbidden request with unexpected Host header");
                return Err(StatusCode::FORBIDDEN);
            }
        }
    }

    Ok(next.run(request).await)
}

#[cfg(test)]
mod tests {
    use super::*;
    use axum::{
        Router,
        body::Body,
        http::{Request, StatusCode},
        routing::get,
    };
    use tower::ServiceExt;

    fn create_test_state() -> Arc<AppState> {
        let model = Arc::new(std::sync::RwLock::new(model::MiniTransformer::new_random()));
        let db_conn = db::init_db(":memory:").unwrap();
        let (log_tx, _) = tokio::sync::mpsc::channel(1);
        Arc::new(AppState {
            model,
            db_conn: std::sync::Mutex::new(db_conn),
            secret_key: vec![0u8; 32],
            min_score: 0.5,
            difficulty: 4.0,
            cache: cache::MemoryCache::new(),
            log_tx,
            expected_host: Some("null.dill.moe".to_string()),
            rate_limiter: std::sync::Mutex::new(RateLimiter::new(100, 60)),
        })
    }

    #[tokio::test]
    async fn test_host_guard_middleware_allowed() {
        let state = create_test_state();
        let app = Router::new()
            .route("/", get(|| async { "OK" }))
            .layer(axum::middleware::from_fn_with_state(
                Arc::clone(&state),
                host_guard_middleware,
            ))
            .with_state(state);

        let response = app
            .oneshot(
                Request::builder()
                    .uri("/")
                    .header("Host", "null.dill.moe")
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();

        assert_eq!(response.status(), StatusCode::OK);
    }

    #[tokio::test]
    async fn test_host_guard_middleware_allowed_with_port() {
        let state = create_test_state();
        let app = Router::new()
            .route("/", get(|| async { "OK" }))
            .layer(axum::middleware::from_fn_with_state(
                Arc::clone(&state),
                host_guard_middleware,
            ))
            .with_state(state);

        let response = app
            .oneshot(
                Request::builder()
                    .uri("/")
                    .header("Host", "null.dill.moe:443")
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();

        assert_eq!(response.status(), StatusCode::OK);
    }

    #[tokio::test]
    async fn test_host_guard_middleware_forbidden() {
        let state = create_test_state();
        let app = Router::new()
            .route("/", get(|| async { "OK" }))
            .layer(axum::middleware::from_fn_with_state(
                Arc::clone(&state),
                host_guard_middleware,
            ))
            .with_state(state);

        let response = app
            .oneshot(
                Request::builder()
                    .uri("/")
                    .header("Host", "evil-domain.com")
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();

        assert_eq!(response.status(), StatusCode::FORBIDDEN);
    }

    #[tokio::test]
    async fn test_host_guard_middleware_missing_host() {
        let state = create_test_state();
        let app = Router::new()
            .route("/", get(|| async { "OK" }))
            .layer(axum::middleware::from_fn_with_state(
                Arc::clone(&state),
                host_guard_middleware,
            ))
            .with_state(state);

        let response = app
            .oneshot(Request::builder().uri("/").body(Body::empty()).unwrap())
            .await
            .unwrap();

        assert_eq!(response.status(), StatusCode::FORBIDDEN);
    }

    #[test]
    fn test_verify_pow_fractional() {
        let salt = "test_salt";
        let mut nonce = 0;
        let mut solved = false;
        while nonce < 1_000_000 {
            if verify_pow(salt, nonce, 4.5) {
                solved = true;
                break;
            }
            nonce += 1;
        }
        assert!(solved, "Should find a solution for difficulty 4.5");
        assert!(verify_pow(salt, nonce, 4.0));
        assert!(!verify_pow(salt, nonce, f64::NAN));
        assert!(!verify_pow(salt, nonce, -1.0));

        use sha2::{Digest, Sha256};
        let mut hasher = Sha256::new();
        hasher.update(format!("{}{}", salt, nonce).as_bytes());
        let hash_bytes = hasher.finalize();
        let mut zero_bits = 0;
        for &byte in hash_bytes.iter() {
            let leading = byte.leading_zeros() as usize;
            zero_bits += leading;
            if leading < 8 {
                break;
            }
        }
        assert!(zero_bits >= 18);
    }
}
