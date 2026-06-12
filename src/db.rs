use rusqlite::{params, Connection, Result};
use crate::model::MLP;
use std::time::{SystemTime, UNIX_EPOCH};

pub fn init_db(db_path: &str) -> Result<Connection> {
    let conn = Connection::open(db_path)?;

    // 1. Create table for model weights
    conn.execute(
        "CREATE TABLE IF NOT EXISTS model_weights (
            key TEXT PRIMARY KEY,
            value TEXT NOT NULL,
            updated_at INTEGER NOT NULL
        )",
        [],
    )?;

    // 2. Create table for telemetry logs (for debugging, analytics, and future training)
    conn.execute(
        "CREATE TABLE IF NOT EXISTS telemetry_logs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp INTEGER NOT NULL,
            point_count INTEGER NOT NULL,
            score REAL NOT NULL,
            is_human INTEGER NOT NULL,
            webdriver INTEGER NOT NULL,
            user_agent TEXT,
            ip_address TEXT,
            features_json TEXT NOT NULL,
            points_hash TEXT NOT NULL DEFAULT ''
        )",
        [],
    )?;

    // Try to alter table to add the column if it doesn't exist (ignoring errors if it already exists)
    let _ = conn.execute(
        "ALTER TABLE telemetry_logs ADD COLUMN points_hash TEXT NOT NULL DEFAULT ''",
        [],
    );

    // Create index for fast points_hash lookups
    conn.execute(
        "CREATE INDEX IF NOT EXISTS idx_telemetry_points_hash ON telemetry_logs (points_hash)",
        [],
    )?;

    // 3. Create table for validated tokens to prevent replay/double-use attacks
    conn.execute(
        "CREATE TABLE IF NOT EXISTS used_tokens (
            token TEXT PRIMARY KEY,
            validated_at INTEGER NOT NULL
        )",
        [],
    )?;

    Ok(conn)
}

/// Save the current model weights and biases to the database
pub fn save_model(conn: &Connection, model: &MLP) -> Result<(), String> {
    let serialized = serde_json::to_string(model)
        .map_err(|e| format!("Failed to serialize model: {}", e))?;
    
    let now = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap()
        .as_secs() as i64;

    conn.execute(
        "INSERT OR REPLACE INTO model_weights (key, value, updated_at) VALUES ('current_model_v2', ?1, ?2)",
        params![serialized, now],
    ).map_err(|e| format!("Database save error: {}", e))?;

    Ok(())
}

/// Load model weights from the database, if they exist
pub fn load_model(conn: &Connection) -> Result<Option<MLP>, String> {
    let mut stmt = conn
        .prepare("SELECT value FROM model_weights WHERE key = 'current_model_v2'")
        .map_err(|e| format!("Prepare statement failed: {}", e))?;

    let mut rows = stmt
        .query([])
        .map_err(|e| format!("Query failed: {}", e))?;

    if let Some(row) = rows.next().map_err(|e| format!("Row fetching failed: {}", e))? {
        let val: String = row.get(0).map_err(|e| format!("Get column failed: {}", e))?;
        let model: MLP = serde_json::from_str(&val)
            .map_err(|e| format!("Failed to deserialize model: {}", e))?;
        Ok(Some(model))
    } else {
        Ok(None)
    }
}

/// Log telemetry data of a verification request
pub fn log_telemetry(
    conn: &Connection,
    point_count: usize,
    score: f32,
    is_human: bool,
    webdriver: bool,
    user_agent: Option<&str>,
    ip_address: Option<&str>,
    features_json: &str,
    points_hash: &str,
) -> Result<(), String> {
    let now = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap()
        .as_secs() as i64;

    conn.execute(
        "INSERT INTO telemetry_logs (timestamp, point_count, score, is_human, webdriver, user_agent, ip_address, features_json, points_hash)
         VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9)",
        params![
            now,
            point_count as i64,
            score as f64,
            if is_human { 1 } else { 0 },
            if webdriver { 1 } else { 0 },
            user_agent,
            ip_address,
            features_json,
            points_hash
        ],
    ).map_err(|e| format!("Failed to insert telemetry log: {}", e))?;

    Ok(())
}

/// Check if telemetry points have already been used to prevent replay attacks
#[allow(dead_code)]
pub fn is_telemetry_replayed(conn: &Connection, points_hash: &str) -> Result<bool, String> {
    if points_hash.is_empty() {
        return Ok(false);
    }
    let mut stmt = conn
        .prepare("SELECT 1 FROM telemetry_logs WHERE points_hash = ?1 LIMIT 1")
        .map_err(|e| format!("Prepare query failed: {}", e))?;
    
    let exists = stmt.exists(params![points_hash])
        .map_err(|e| format!("Query points_hash failed: {}", e))?;

    Ok(exists)
}

/// Check if a token has already been validated (to prevent reuse)
#[allow(dead_code)]
pub fn is_token_used(conn: &Connection, token: &str) -> Result<bool, String> {
    let mut stmt = conn
        .prepare("SELECT 1 FROM used_tokens WHERE token = ?1")
        .map_err(|e| format!("Prepare query failed: {}", e))?;
    
    let exists = stmt.exists(params![token])
        .map_err(|e| format!("Query token failed: {}", e))?;

    Ok(exists)
}

/// Mark a token as validated atomically. Returns Ok(true) if the token was successfully
/// marked as used, and Ok(false) if it was already marked as used (due to primary key constraint violation).
#[allow(dead_code)]
pub fn try_mark_token_used(conn: &Connection, token: &str) -> Result<bool, String> {
    let now = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap()
        .as_secs() as i64;

    match conn.execute(
        "INSERT INTO used_tokens (token, validated_at) VALUES (?1, ?2)",
        params![token, now],
    ) {
        Ok(_) => Ok(true),
        Err(rusqlite::Error::SqliteFailure(err, _)) => {
            if err.code == rusqlite::ErrorCode::ConstraintViolation {
                Ok(false)
            } else {
                Err(format!("Failed to record used token: {}", err))
            }
        }
        Err(e) => Err(format!("Failed to record used token: {}", e)),
    }
}

/// Clean up expired tokens (older than max_age_secs)
#[allow(dead_code)]
pub fn prune_old_tokens(conn: &Connection, max_age_secs: u64) -> Result<usize, String> {
    let now = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap()
        .as_secs() as i64;
    
    let cutoff = now - (max_age_secs as i64);

    let deleted = conn.execute(
        "DELETE FROM used_tokens WHERE validated_at < ?1",
        params![cutoff],
    ).map_err(|e| format!("Failed to prune tokens: {}", e))?;

    Ok(deleted)
}
