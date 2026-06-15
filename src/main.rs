use axum::{
    http::{header, HeaderValue, HeaderMap},
    response::{Html, IntoResponse, Response},
    routing::{get, post},
    Json, Router,
    extract::ConnectInfo,
};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::net::SocketAddr;
use std::sync::Arc;
use tower_http::cors::CorsLayer;
use tracing::{info, warn, Level};
use tracing_subscriber::FmtSubscriber;
use hmac::Mac;
use base64::Engine;
use subtle::ConstantTimeEq;

mod crypto;
mod features;
mod model;
mod db;
mod cache;

// Embed frontend assets directly into the binary
const INDEX_HTML: &str = include_str!("static/index.html");
const NULL_JS: &str = include_str!("static/null.js");
const LLMS_TXT: &str = include_str!("static/llms.txt");

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
        let timestamps = self.buckets.entry(ip.to_string()).or_insert_with(Vec::new);
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
    _db_conn: std::sync::Mutex<rusqlite::Connection>,
    secret_key: Vec<u8>,
    min_score: f32,
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

#[tokio::main]
async fn main() {
    // Initialize tracing logger
    let subscriber = FmtSubscriber::builder()
        .with_max_level(Level::INFO)
        .finish();
    tracing::subscriber::set_global_default(subscriber)
        .expect("setting default subscriber failed");

    info!("Initializing Null CAPTCHA Server...");

    // 1. Initialize SQLite Database (statefulness)
    let db_path = std::env::var("DATABASE_URL").unwrap_or_else(|_| "captcha.db".to_string());
    info!("Connecting to SQLite database at {}...", db_path);
    let db_conn = db::init_db(&db_path).expect("Failed to initialize SQLite database");

    // 2. Load or train the Neural Network model
    info!("Checking database for existing trained neural network weights...");
    let raw_model = match db::load_model(&db_conn) {
        Ok(Some(loaded)) => {
            info!("Loaded trained neural network weights from database.");
            loaded
        }
        _ => {
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
    info!("Mini Transformer model ready. (Input: 8 features, Embedding: 16, FF: 32, Heads: 1, Output: 1)");

    let model = Arc::new(std::sync::RwLock::new(raw_model));

    // 3. Read Configuration
    let port = std::env::var("PORT")
        .unwrap_or_else(|_| "3000".to_string())
        .parse::<u16>()
        .expect("PORT must be a valid u16 port number");

    let secret_key = std::env::var("NULL_CAPTCHA_SECRET")
        .map(|s| s.into_bytes())
        .unwrap_or_else(|_| {
            warn!("NULL_CAPTCHA_SECRET environment variable not set. Generating a random key for this session.");
            warn!("Note: Verification tokens will invalidate if the server restarts!");
            let mut key = vec![0u8; 32];
            rand::Rng::fill(&mut rand::thread_rng(), &mut key[..]);
            key
        });

    let min_score = std::env::var("NULL_CAPTCHA_MIN_SCORE")
        .unwrap_or_else(|_| "0.5".to_string())
        .parse::<f32>()
        .unwrap_or(0.5);

    // Initialize Memory Cache for token and telemetry replay checks
    let cache = cache::MemoryCache::new();

    // Setup asynchronous background telemetry logger to prevent SQLite write lock blocks
    let (log_tx, mut log_rx) = tokio::sync::mpsc::channel::<LogMessage>(5000);
    let db_path_clone = db_path.clone();
    let model_clone = Arc::clone(&model);
    tokio::spawn(async move {
        if let Ok(conn) = rusqlite::Connection::open(&db_path_clone) {
            // Configure WAL and busy timeout for the background connection as well
            let _ = conn.execute_batch(
                "PRAGMA journal_mode = WAL;
                 PRAGMA busy_timeout = 5000;"
            );

            // Prune old tokens once on startup
            let _ = db::prune_old_tokens(&conn, 300);

            let mut last_prune = std::time::Instant::now();
            let mut logged_interactions_count = 0;

            while let Some(msg) = log_rx.recv().await {
                let _ = db::log_telemetry(
                    &conn,
                    msg.point_count,
                    msg.score,
                    msg.is_human,
                    msg.webdriver,
                    msg.user_agent.as_deref(),
                    msg.ip_address.as_deref(),
                    &msg.features_json,
                    &msg.points_hash,
                    msg.is_high_confidence,
                );

                // Increment logged interaction counter for adaptive training
                if !msg.features_json.is_empty() && msg.is_high_confidence {
                    logged_interactions_count += 1;
                }

                // Periodically retrain and adapt the model (every 10 new logged interactions)
                if logged_interactions_count >= 10 {
                    info!("Adapting neural network: retraining model on recent telemetry logs + synthetic dataset...");

                    // Fetch recent logs
                    match db::get_recent_telemetry(&conn, 500) {
                        Ok(mut recent_data) => {
                            info!("Fetched {} recent telemetry samples from database.", recent_data.len());

                            // Generate synthetic dataset
                            let mut synthetic_data = model::MiniTransformer::generate_synthetic_dataset();

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

                            // Fine-tune the current model clone in the background instead of training from scratch.
                            // This provides training stability and preserves learned boundaries.
                            let mut new_model = model_clone.read().unwrap().clone();
                            let loss = new_model.train(&train_dataset, 40, 0.015);

                            // Validate model accuracy and verify numerical sanity before promotion
                            let accuracy = new_model.validate(&val_dataset);
                            let is_sane = new_model.is_sane();

                            if is_sane && accuracy >= 0.90 {
                                info!("Model retraining succeeded. Loss: {:.6}, Validation Accuracy: {:.2}%", loss, accuracy * 100.0);

                                // Swap the model under a brief write lock (minimal block time)
                                {
                                    let mut model_guard = model_clone.write().unwrap();
                                    *model_guard = new_model;
                                }

                                // Save the updated model weights back to the database
                                {
                                    let model_guard = model_clone.read().unwrap();
                                    if let Err(e) = db::save_model(&conn, &model_guard) {
                                        warn!("Failed to save adapted model weights to database: {}", e);
                                    } else {
                                        info!("Adapted model weights successfully saved/persisted to database.");
                                    }
                                }
                            } else {
                                warn!(
                                    "Rejected retrained model update! Sanity check: {}, Validation Accuracy: {:.2}% (threshold: 90.0%)",
                                    if is_sane { "PASSED" } else { "FAILED" },
                                    accuracy * 100.0
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
                if last_prune.elapsed().as_secs() > 300 {
                    let _ = db::prune_old_tokens(&conn, 300);
                    last_prune = std::time::Instant::now();
                }
            }
        }
    });

    // Initialize expected host from URL environment variable if set
    let expected_host = std::env::var("URL").ok().and_then(|url| {
        let mut s = url.as_str();
        if s.starts_with("https://") {
            s = &s["https://".len()..];
        } else if s.starts_with("http://") {
            s = &s["http://".len()..];
        }
        if let Some(pos) = s.find('/') {
            s = &s[..pos];
        }
        let clean_host = if let Some(pos) = s.find(':') {
            &s[..pos]
        } else {
            s
        };
        if clean_host.is_empty() {
            None
        } else {
            Some(clean_host.to_string())
        }
    });

    if let Some(ref host) = expected_host {
        info!("URL restriction active: only allowing access via domain '{}'", host);
    }

    let state = Arc::new(AppState {
        model,
        _db_conn: std::sync::Mutex::new(db_conn),
        secret_key,
        min_score,
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
        .route("/api/challenge", get(challenge_handler))
        .route("/api/verify", post(verify_handler))
        .route("/api/validate", post(validate_handler))
        .layer(axum::middleware::from_fn_with_state(
            Arc::clone(&state),
            host_guard_middleware,
        ))
        .layer(CorsLayer::permissive())
        .with_state(state);

    // 5. Bind and run server
    let addr = SocketAddr::from(([0, 0, 0, 0], port));
    let listener = tokio::net::TcpListener::bind(addr)
        .await
        .unwrap_or_else(|e| panic!("Failed to bind to port {}: {}", port, e));

    info!("Null CAPTCHA server running at http://localhost:{}", port);
    axum::serve(listener, app.into_make_service_with_connect_info::<SocketAddr>()).await.unwrap();
}

/// Serve the gorgeous demo landing page
async fn serve_index() -> impl IntoResponse {
    Html(INDEX_HTML)
}

/// Serve the telemetry tracker JavaScript library with correct content type
async fn serve_js() -> impl IntoResponse {
    Response::builder()
        .header(header::CONTENT_TYPE, HeaderValue::from_static("application/javascript"))
        .body(NULL_JS.to_string())
        .unwrap()
}

/// Serve the llms.txt guide for AI agents
async fn serve_llms() -> impl IntoResponse {
    Response::builder()
        .header(header::CONTENT_TYPE, HeaderValue::from_static("text/plain; charset=utf-8"))
        .body(LLMS_TXT.to_string())
        .unwrap()
}

fn verify_pow(salt: &str, nonce: u64, difficulty: f64) -> bool {
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
) -> impl IntoResponse {
    use rand::distributions::Alphanumeric;
    use rand::Rng;
    use std::time::{SystemTime, UNIX_EPOCH};

    let now = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap()
        .as_secs();

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

    let difficulty = std::env::var("NULL_CAPTCHA_DIFFICULTY")
        .unwrap_or_else(|_| "4".to_string())
        .parse::<f64>()
        .unwrap_or(4.0);

    let sign_payload = format!("{}.{}.{}.{}", now, salt, difficulty, encryption_key);

    let mut mac = hmac::Hmac::<sha2::Sha256>::new_from_slice(&state.secret_key)
        .expect("HMAC sign failed");
    mac.update(sign_payload.as_bytes());
    let signature = base64::engine::general_purpose::URL_SAFE_NO_PAD.encode(mac.finalize().into_bytes());

    Json(ChallengeResponse {
        salt,
        difficulty,
        encryption_key,
        timestamp: now,
        signature,
    })
}

/// API endpoint to verify client-side telemetry
async fn verify_handler(
    axum::extract::State(state): axum::extract::State<Arc<AppState>>,
    headers: HeaderMap,
    ConnectInfo(addr): ConnectInfo<SocketAddr>,
    Json(payload): Json<VerifyPayload>,
) -> impl IntoResponse {
    // 1. Verify Challenge Signature
    let expected_payload = format!("{}.{}.{}.{}", payload.timestamp, payload.salt, payload.difficulty, payload.encryption_key);
    let mut mac = hmac::Hmac::<sha2::Sha256>::new_from_slice(&state.secret_key)
        .expect("HMAC verification signature setup failed");
    mac.update(expected_payload.as_bytes());
    let expected_signature = base64::engine::general_purpose::URL_SAFE_NO_PAD.encode(mac.finalize().into_bytes());

    // Constant-time signature verification to prevent timing attacks
    let sig_bytes = payload.signature.as_bytes();
    let expected_bytes = expected_signature.as_bytes();
    if sig_bytes.len() != expected_bytes.len() || sig_bytes.ct_eq(expected_bytes).unwrap_u8() != 1 {
        warn!("Verification challenge signature mismatch!");
        return Json(VerifyResponse {
            error: Some("Invalid challenge signature. Re-verification required.".to_string()),
            ..Default::default()
        });
    }

    // 2. Verify Challenge Expiration
    use std::time::{SystemTime, UNIX_EPOCH};
    let now = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap()
        .as_secs();
    if now < payload.timestamp || now - payload.timestamp > 300 {
        warn!("Verification challenge expired! now: {}, challenge: {}", now, payload.timestamp);
        return Json(VerifyResponse {
            error: Some("Verification challenge expired. Please retry.".to_string()),
            ..Default::default()
        });
    }

    // 3. Challenge State Enforcement (Strict Replay & Brute-Force prevention)
    let consumed_key = format!("salt:consumed:{}", payload.salt);
    let fallback_key = format!("salt:fallback:{}", payload.salt);

    // Check if challenge was already marked fully consumed
    if state.cache.contains(&consumed_key) {
        warn!("Verification failed: Salt already consumed.");
        return Json(VerifyResponse {
            error: Some("Challenge already completed. Re-verification required.".to_string()),
            ..Default::default()
        });
    }

    if payload.slider_x.is_some() {
        // If submitting a slider solve, the salt state MUST be "fallback_required"
        if !state.cache.contains(&fallback_key) {
            warn!("Verification failed: Attempted to submit slider solve without prior passive fail.");
            return Json(VerifyResponse {
                error: Some("Invalid challenge sequence. Re-verification required.".to_string()),
                ..Default::default()
            });
        }
        // Immediately invalidate the fallback state so it can never be retried/brute-forced
        state.cache.remove(&fallback_key);
        state.cache.insert(consumed_key.clone(), 300);
    } else {
        // If submitting a passive check and it was already processed, reject
        if state.cache.contains(&fallback_key) {
            warn!("Verification failed: Passive check already attempted and fallback is active.");
            return Json(VerifyResponse {
                error: Some("Please complete the slider fallback puzzle.".to_string()),
                fallback_required: Some(true),
                ..Default::default()
            });
        }
    }

    // 4. Rate limiting: per-IP sliding window (30 requests per 60 seconds)
    let ip_str = headers
        .get("x-forwarded-for")
        .and_then(|v| v.to_str().ok().map(|s| s.split(',').next().unwrap_or("").trim().to_string()))
        .unwrap_or_else(|| addr.ip().to_string());
    {
        let mut limiter = state.rate_limiter.lock().unwrap();
        if !limiter.check_and_record(&ip_str, now) {
            warn!("Rate limit exceeded for IP: {}", ip_str);
            return Json(VerifyResponse {
                error: Some("Too many requests. Please slow down.".to_string()),
                ..Default::default()
            });
        }
        // Periodically clean up stale entries (every ~10 requests as a simple heuristic)
        if limiter.buckets.len() > 1000 {
            limiter.remove_expired(now);
        }
    }

    // 5. Verify Proof of Work
    if !verify_pow(&payload.salt, payload.nonce, payload.difficulty) {
        warn!("Verification PoW mismatch! salt: {}, nonce: {}, diff: {}", payload.salt, payload.nonce, payload.difficulty);
        return Json(VerifyResponse {
            error: Some("Invalid Proof of Work solution.".to_string()),
            ..Default::default()
        });
    }

    // 6. PoW minimum time validation: check that the nonce wasn't found too quickly
    //    (pre-baked or instantly computed). Minimum 1 second of real wall time required.
    if now.saturating_sub(payload.timestamp) < 1 {
        warn!("PoW solved too quickly ({}s), possible pre-computed nonce", now.saturating_sub(payload.timestamp));
        return Json(VerifyResponse {
            error: Some("Proof of Work must take real computation time.".to_string()),
            ..Default::default()
        });
    }

    // 4. Decrypt and parse obfuscated payload
    let decrypted_str = match crypto::decrypt_payload(&payload.payload, &payload.salt, &payload.encryption_key) {
        Ok(s) => s,
        Err(e) => {
            warn!("Payload decryption failed: {}", e);
            return Json(VerifyResponse {
                error: Some("Payload validation failed. Ensure JS client is up to date.".to_string()),
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
        warn!("Verification failed: Telemetry salt mismatch!");
        return Json(VerifyResponse {
            error: Some("Telemetry payload is not bound to this challenge.".to_string()),
            ..Default::default()
        });
    }

    // Verify telemetry points are unique to prevent replay of the same movements
    let points_hash = compute_points_hash(&client_data.points);
    if !client_data.points.is_empty() {
        // Check in-memory cache first for speed
        if !state.cache.check_and_insert(format!("telemetry:{}", points_hash), 300) {
            warn!("Verification failed: Replayed telemetry detected in cache!");
            return Json(VerifyResponse {
                error: Some("Automated behavior detected (replayed telemetry).".to_string()),
                ..Default::default()
            });
        }

        // Check persistent database for replay checks
        let is_replay = {
            let conn = state._db_conn.lock().unwrap();
            db::is_telemetry_replayed(&conn, &points_hash).unwrap_or(false)
        };
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
        warn!("Automation detected: click happened too fast ({}ms)", client_data.time_taken);
        bot_flag = true;
    }
    if client_data.screen.ow == 0 || client_data.screen.oh == 0 {
        warn!("Automation detected: screen size report invalid (0x0 outer)");
        bot_flag = true;
    }

    // Check slider fallback puzzle solve status
    let mut solved_slider = false;
    let mut is_accessibility = false;
    let target = derive_slider_target(&payload.salt, &state.secret_key);

    if let Some(x) = payload.slider_x {
        if (x - target).abs() <= 6 {
            solved_slider = true;
            if client_data.accessibility_mode {
                is_accessibility = true;
            }
        }
    }

    // Filter out keyboard/accessibility events (y == -1.0) for mouse telemetry extraction
    let mouse_points: Vec<features::TelemetryPoint> = client_data.points.iter()
        .filter(|p| (p.y - -1.0).abs() > 0.001)
        .cloned()
        .collect();

    // 1. Passive checking (mouse pointer telemetry)
    let features_opt = features::extract_features(&mouse_points);

    let mut features_json = String::new();
    if let Some(ref features) = features_opt {
        features_json = serde_json::to_string(&features).unwrap_or_default();
        let feature_arr = features.to_array();
        let mut model_score = state.model.read().unwrap().predict(&feature_arr);

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
    if !is_human && !bot_flag {
        if solved_slider {
            if is_accessibility {
                // Keyboard / accessibility user successfully aligned the slider.
                // Verify that keyboard telemetry looks human to prevent simple script bypass.
                let kb_points: Vec<&features::TelemetryPoint> = client_data.points.iter()
                    .filter(|p| (p.y - -1.0).abs() < 0.001)
                    .collect();

                if kb_points.is_empty() {
                    warn!("Automation detected: accessibility mode claimed but no keyboard events recorded.");
                    bot_flag = true;
                } else {
                    // Check for automated rapid keystrokes (less than 15ms interval)
                    let mut too_fast = false;
                    for i in 0..(kb_points.len() - 1) {
                        let dt = kb_points[i+1].t - kb_points[i].t;
                        if dt < 15.0 {
                            too_fast = true;
                            break;
                        }
                    }
                    if too_fast {
                        warn!("Automation detected: keyboard events are too fast (less than 15ms interval)");
                        bot_flag = true;
                    } else {
                        // Check if the number of keypresses matches target distance roughly
                        // Initial slider position is 25. Each ArrowRight/ArrowLeft moves by 5.
                        let target_x = payload.slider_x.unwrap_or(25);
                        let expected_presses = ((target_x - 25).abs() as f32 / 5.0).ceil() as usize;

                        // Allow some flexibility but ensure they did at least half of the expected keystrokes
                        if kb_points.len() < expected_presses / 2 {
                            warn!("Automation detected: accessibility keypress count {} too low for target distance (expected ~{})", kb_points.len(), expected_presses);
                            bot_flag = true;
                        } else {
                            score = 1.0;
                            is_human = true;
                            info!("Accessibility validation succeeded via slider puzzle ({} keypresses).", kb_points.len());
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
                    let claimed_drag_dist = (payload.slider_x.unwrap_or(25) - 25).abs() as f32;

                    if span_x < claimed_drag_dist - 15.0 {
                        warn!("Automation detected: horizontal mouse movement span ({:.2}px) is less than claimed drag distance ({:.2}px)", span_x, claimed_drag_dist);
                        valid_drag_dist = false;
                    }
                } else {
                    valid_drag_dist = false;
                }

                if valid_drag_dist {
                    if let Some(features) = features::extract_features(&mouse_points) {
                        let feature_arr = features.to_array();
                        let drag_score = state.model.read().unwrap().predict(&feature_arr);
                        if drag_score >= 0.25 {
                            score = drag_score;
                            is_human = true;
                            info!("Slider puzzle verification succeeded with drag score: {:.4}", drag_score);
                        } else {
                            warn!("Slider puzzle verification failed: drag score {:.4} below human threshold", drag_score);
                        }
                    }
                } else {
                    bot_flag = true;
                }
            }
        }
    }

    // Get client details for logging
    let user_agent = headers
        .get("user-agent")
        .and_then(|v| v.to_str().ok());

    let ip_address = headers
        .get("x-forwarded-for")
        .and_then(|v| v.to_str().ok().map(|s| s.split(',').next().unwrap_or("").trim().to_string()))
        .unwrap_or_else(|| addr.ip().to_string());

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
                    features.speed_var >= 0.08 &&
                    features.total_duration >= 0.075 &&
                    features.entropy >= 0.02 &&
                    (features.angular_jitter >= 0.001 || features.line_deviation >= 0.0005)
                }
            } else {
                false
            }
        } else {
            // If they attempted the slider but drag features were classified as bot-like
            payload.slider_x.is_some() && !is_accessibility
        }
    };

    // Send log message to background channel for async persistence (non-blocking)
    let log_msg = LogMessage {
        point_count: client_data.points.len(),
        score,
        is_human,
        webdriver: client_data.webdriver,
        user_agent: user_agent.map(String::from),
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
        state.cache.insert(consumed_key, 300);

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
            state.cache.insert(fallback_key, 300);
            target_val = Some(derive_slider_target(&payload.salt, &state.secret_key));
        } else {
            // Obvious bot or failed slider check, mark as consumed
            state.cache.insert(consumed_key, 300);
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
    info!("Validating token: {:?}", payload.token);

    // 1. Verify token signature and timestamp BEFORE cache and database operations
    // to prevent database/cache pollution from invalid token spam.
    let score = match crypto::verify_token(&state.secret_key, &payload.token, 300) {
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

    // 2. Check if token was already validated to prevent token replay attacks
    // Check in-memory cache first for speed
    if !state.cache.check_and_insert(format!("token:{}", payload.token), 300) {
        warn!("Replay validation check: Token has already been used (cache).");
        return Json(ValidateResponse {
            success: false,
            score: 0.0,
            error: Some("Token has already been validated.".to_string()),
        });
    }

    // Mark token as used in database atomically to prevent race conditions
    let mark_success = {
        let conn = state._db_conn.lock().unwrap();
        db::try_mark_token_used(&conn, &payload.token).unwrap_or(false)
    };
    if !mark_success {
        warn!("Replay validation check: Token has already been used (database).");
        return Json(ValidateResponse {
            success: false,
            score: 0.0,
            error: Some("Token has already been validated.".to_string()),
        });
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
        warn!("Token validation failed: Score {:.4} below threshold", score);
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
    for p in points {
        hasher.update(format!("{},{},{}", p.x, p.y, p.t).as_bytes());
    }
    let hash_bytes = hasher.finalize();
    format!("{:x}", hash_bytes)
}

/// Middleware to restrict access only to the domain specified in the URL environment variable
async fn host_guard_middleware(
    axum::extract::State(state): axum::extract::State<Arc<AppState>>,
    request: axum::extract::Request,
    next: axum::middleware::Next,
) -> Result<axum::response::Response, axum::http::StatusCode> {
    if let Some(ref expected) = state.expected_host {
        let host_header = request
            .headers()
            .get(axum::http::header::HOST)
            .and_then(|v| v.to_str().ok());

        let matches = if let Some(host) = host_header {
            let clean_req_host = if let Some(pos) = host.find(':') {
                &host[..pos]
            } else {
                host
            };
            clean_req_host == expected
        } else {
            false
        };

        if !matches {
            warn!("Forbidden access from host: {:?}", host_header);
            return Err(axum::http::StatusCode::FORBIDDEN);
        }
    }

    Ok(next.run(request).await)
}

#[cfg(test)]
mod tests {
    use super::*;
    use axum::{
        body::Body,
        http::{Request, StatusCode},
        routing::get,
        Router,
    };
    use tower::ServiceExt;

    fn create_test_state() -> Arc<AppState> {
        let model = Arc::new(std::sync::RwLock::new(model::MiniTransformer::new_default()));
        let db_conn = db::init_db(":memory:").unwrap();
        let (log_tx, _) = tokio::sync::mpsc::channel(1);
        Arc::new(AppState {
            model,
            _db_conn: std::sync::Mutex::new(db_conn),
            secret_key: vec![0u8; 32],
            min_score: 0.5,
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
            .oneshot(
                Request::builder()
                    .uri("/")
                    .body(Body::empty())
                    .unwrap(),
            )
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

