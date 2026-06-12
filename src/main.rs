use axum::{
    http::{header, HeaderValue, HeaderMap},
    response::{Html, IntoResponse, Response},
    routing::{get, post},
    Json, Router,
    extract::ConnectInfo,
};
use serde::{Deserialize, Serialize};
use std::net::SocketAddr;
use std::sync::Arc;
use tower_http::cors::CorsLayer;
use tracing::{info, warn, Level};
use tracing_subscriber::FmtSubscriber;
use hmac::Mac;
use base64::Engine;

mod crypto;
mod features;
mod model;
mod db;

// Embed frontend assets directly into the binary
const INDEX_HTML: &str = include_str!("static/index.html");
const NULL_JS: &str = include_str!("static/null.js");

/// Application state containing neural network, database connection, and security configurations
struct AppState {
    model: model::MLP,
    db_conn: std::sync::Mutex<rusqlite::Connection>,
    secret_key: Vec<u8>,
    min_score: f32,
}

#[derive(Deserialize)]
struct VerifyPayload {
    payload: String,
    salt: String,
    signature: String,
    timestamp: u64,
    difficulty: u32,
    #[serde(rename = "encryptionKey")]
    encryption_key: u8,
    nonce: u64,
}

#[derive(Serialize)]
#[serde(rename_all = "camelCase")]
struct ChallengeResponse {
    salt: String,
    difficulty: u32,
    encryption_key: u8,
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
    points: Vec<features::TelemetryPoint>,
    webdriver: bool,
    plugins: usize,
    languages: usize,
    screen: ClientScreen,
    time_taken: u64,
}

#[derive(Serialize)]
struct VerifyResponse {
    success: bool,
    score: f32,
    token: Option<String>,
    error: Option<String>,
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
    let model = match db::load_model(&db_conn) {
        Ok(Some(loaded)) => {
            info!("Loaded trained neural network weights from database.");
            loaded
        }
        _ => {
            info!("No saved weights found. Training neural network from synthetic data...");
            let new_model = model::MLP::new_default();
            if let Err(e) = db::save_model(&db_conn, &new_model) {
                warn!("Failed to save trained neural network to database: {}", e);
            } else {
                info!("Saved trained neural network weights to database for persistence.");
            }
            new_model
        }
    };
    info!("Neural Network model ready. (Input: 8, Hidden1: 12, Hidden2: 8, Output: 1)");

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

    let state = Arc::new(AppState {
        model,
        db_conn: std::sync::Mutex::new(db_conn),
        secret_key,
        min_score,
    });

    // 4. Setup router with CORS
    let app = Router::new()
        .route("/", get(serve_index))
        .route("/js/null.js", get(serve_js))
        .route("/api/challenge", get(challenge_handler))
        .route("/api/verify", post(verify_handler))
        .route("/api/validate", post(validate_handler))
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

fn verify_pow(salt: &str, nonce: u64, difficulty: u32) -> bool {
    use sha2::{Digest, Sha256};
    let mut hasher = Sha256::new();
    hasher.update(format!("{}{}", salt, nonce).as_bytes());
    let hash_bytes = hasher.finalize();
    let hash_hex = format!("{:x}", hash_bytes);
    let prefix = "0".repeat(difficulty as usize);
    hash_hex.starts_with(&prefix)
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

    let encryption_key: u8 = rand::thread_rng().gen_range(1..=255);
    let difficulty: u32 = 4; // requires 4 leading hex zeros (~100-300ms to solve)

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

    if expected_signature != payload.signature {
        warn!("Verification challenge signature mismatch!");
        return Json(VerifyResponse {
            success: false,
            score: 0.0,
            token: None,
            error: Some("Invalid challenge signature. Re-verification required.".to_string()),
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
            success: false,
            score: 0.0,
            token: None,
            error: Some("Verification challenge expired. Please retry.".to_string()),
        });
    }

    // 3. Verify Proof of Work
    if !verify_pow(&payload.salt, payload.nonce, payload.difficulty) {
        warn!("Verification PoW mismatch! salt: {}, nonce: {}, diff: {}", payload.salt, payload.nonce, payload.difficulty);
        return Json(VerifyResponse {
            success: false,
            score: 0.0,
            token: None,
            error: Some("Invalid Proof of Work solution.".to_string()),
        });
    }

    // 4. Decrypt and parse obfuscated payload
    let decrypted_str = match crypto::decrypt_payload(&payload.payload, payload.encryption_key) {
        Ok(s) => s,
        Err(e) => {
            warn!("Payload decryption failed: {}", e);
            return Json(VerifyResponse {
                success: false,
                score: 0.0,
                token: None,
                error: Some("Payload validation failed. Ensure JS client is up to date.".to_string()),
            });
        }
    };

    let client_data: DecryptedPayload = match serde_json::from_str(&decrypted_str) {
        Ok(d) => d,
        Err(e) => {
            warn!("Failed to parse decrypted JSON: {}", e);
            return Json(VerifyResponse {
                success: false,
                score: 0.0,
                token: None,
                error: Some("Telemetry structure mismatch.".to_string()),
            });
        }
    };

    // 2. Extract features from telemetry points
    let features_opt = features::extract_features(&client_data.points);
    
    let features = match features_opt {
        Some(f) => f,
        None => {
            return Json(VerifyResponse {
                success: false,
                score: 0.0,
                token: None,
                error: Some("Insufficient interaction telemetry. Please move your cursor naturally.".to_string()),
            });
        }
    };

    // 3. Perform MLP Neural Network inference
    let feature_arr = features.to_array();
    let mut score = state.model.predict(&feature_arr);
    
    // 4. Apply anti-bot fingerprint rules (Bot Proof checks)
    let mut bot_flag = false;

    if client_data.webdriver {
        warn!("Automation detected: navigator.webdriver is true");
        score = 0.0;
        bot_flag = true;
    }

    if client_data.time_taken < 200 {
        warn!("Automation detected: click happened too fast ({}ms)", client_data.time_taken);
        score = 0.0;
        bot_flag = true;
    }

    if client_data.screen.ow == 0 || client_data.screen.oh == 0 {
        warn!("Automation detected: screen size report invalid (0x0 outer)");
        score = 0.0;
        bot_flag = true;
    }

    if client_data.plugins == 0 {
        // A common headless indicator, we penalize the score significantly
        info!("Anti-bot warning: 0 plugins detected. Applying penalty to classification.");
        score = (score - 0.25).max(0.0);
    }

    let is_human = score >= state.min_score && !bot_flag;

    info!(
        "Verification: Points: {}, Score: {:.4} ({}), Webdriver: {}, Time: {}ms, Plugins: {}",
        client_data.points.len(),
        score,
        if is_human { "Human" } else { "Bot" },
        client_data.webdriver,
        client_data.time_taken,
        client_data.plugins
    );

    // Get client details for logging
    let user_agent = headers
        .get("user-agent")
        .and_then(|v| v.to_str().ok());
    
    let ip_address = headers
        .get("x-forwarded-for")
        .and_then(|v| v.to_str().ok().map(|s| s.split(',').next().unwrap_or("").trim().to_string()))
        .unwrap_or_else(|| addr.ip().to_string());

    // Log telemetry features and fingerprint stats in database for diagnostics
    let features_json = serde_json::to_string(&features).unwrap_or_default();
    if let Ok(conn) = state.db_conn.lock() {
        if let Err(e) = db::log_telemetry(
            &conn,
            client_data.points.len(),
            score,
            is_human,
            client_data.webdriver,
            user_agent,
            Some(&ip_address),
            &features_json,
        ) {
            warn!("Failed to log telemetry into SQLite database: {}", e);
        }
    }

    // 5. Generate token if verification succeeds
    if is_human {
        let token = crypto::generate_token(&state.secret_key, score);
        Json(VerifyResponse {
            success: true,
            score,
            token: Some(token),
            error: None,
        })
    } else {
        Json(VerifyResponse {
            success: false,
            score,
            token: None,
            error: Some("Telemetry profile classified as automated bot behavior.".to_string()),
        })
    }
}

/// API endpoint for backend servers to validate user verification tokens
async fn validate_handler(
    axum::extract::State(state): axum::extract::State<Arc<AppState>>,
    Json(payload): Json<ValidatePayload>,
) -> impl IntoResponse {
    info!("Validating token: {:?}", payload.token);
    // 1. Check if token was already validated to prevent token replay attacks
    if let Ok(conn) = state.db_conn.lock() {
        match db::is_token_used(&conn, &payload.token) {
            Ok(true) => {
                warn!("Replay validation check: Token has already been used.");
                return Json(ValidateResponse {
                    success: false,
                    score: 0.0,
                    error: Some("Token has already been validated.".to_string()),
                });
            }
            Err(e) => {
                warn!("Database check for token replay failed: {}", e);
            }
            _ => {}
        }
    }

    // Max age for a token is set to 300 seconds (5 minutes)
    match crypto::verify_token(&state.secret_key, &payload.token, 300) {
        Ok(score) => {
            if score >= state.min_score {
                info!("Token validation succeeded. Score: {:.4}", score);

                // Mark token as used to prevent replay
                if let Ok(conn) = state.db_conn.lock() {
                    if let Err(e) = db::mark_token_used(&conn, &payload.token) {
                        warn!("Failed to mark token as used: {}", e);
                    }
                    // Prune old tokens to prevent unbounded DB growth
                    let _ = db::prune_old_tokens(&conn, 300);
                }

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
        Err(e) => {
            warn!("Token validation error: {}", e);
            Json(ValidateResponse {
                success: false,
                score: 0.0,
                error: Some(e),
            })
        }
    }
}
