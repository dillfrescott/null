use crate::model::MiniTransformer;
use rusqlite::{Connection, Result, params};
use std::path::Path;
use std::time::{SystemTime, UNIX_EPOCH};

pub struct TelemetryRecord<'a> {
    pub point_count: usize,
    pub score: f32,
    pub is_human: bool,
    pub webdriver: bool,
    pub user_agent: Option<&'a str>,
    pub ip_address: Option<&'a str>,
    pub features_json: &'a str,
    pub points_hash: &'a str,
    pub is_high_confidence: bool,
}

pub fn init_db(db_path: &str) -> Result<Connection> {
    if db_path != ":memory:" {
        if let Some(parent) = Path::new(db_path).parent() {
            if !parent.as_os_str().is_empty() {
                std::fs::create_dir_all(parent)
                    .map_err(|_| rusqlite::Error::InvalidPath(parent.to_path_buf()))?;
            }
        }
    }
    let conn = Connection::open(db_path)?;

    // Enable WAL mode and set a busy timeout of 5 seconds to prevent database locked errors
    conn.execute_batch(
        "PRAGMA journal_mode = WAL;
         PRAGMA synchronous = NORMAL;
         PRAGMA busy_timeout = 5000;",
    )?;

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
            points_hash TEXT NOT NULL DEFAULT '',
            is_high_confidence INTEGER NOT NULL DEFAULT 0
        )",
        [],
    )?;

    // Try to alter table to add the columns if they don't exist (ignoring errors if they already exist)
    let _ = conn.execute(
        "ALTER TABLE telemetry_logs ADD COLUMN points_hash TEXT NOT NULL DEFAULT ''",
        [],
    );
    let _ = conn.execute(
        "ALTER TABLE telemetry_logs ADD COLUMN is_high_confidence INTEGER NOT NULL DEFAULT 0",
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

    // Create index on validated_at for used_tokens to speed up token pruning
    conn.execute(
        "CREATE INDEX IF NOT EXISTS idx_used_tokens_validated_at ON used_tokens (validated_at)",
        [],
    )?;

    Ok(conn)
}

/// Save the current model weights and biases to the database
pub fn save_model(conn: &Connection, model: &MiniTransformer) -> Result<(), String> {
    let serialized =
        serde_json::to_string(model).map_err(|e| format!("Failed to serialize model: {}", e))?;

    let now = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap()
        .as_secs() as i64;

    conn.execute(
        "INSERT OR REPLACE INTO model_weights (key, value, updated_at) VALUES ('current_model_v4_transformer', ?1, ?2)",
        params![serialized, now],
    ).map_err(|e| format!("Database save error: {}", e))?;

    Ok(())
}

/// Load model weights from the database, if they exist
pub fn load_model(conn: &Connection) -> Result<Option<MiniTransformer>, String> {
    let mut stmt = conn
        .prepare("SELECT value FROM model_weights WHERE key = 'current_model_v4_transformer'")
        .map_err(|e| format!("Prepare statement failed: {}", e))?;

    let mut rows = stmt.query([]).map_err(|e| format!("Query failed: {}", e))?;

    if let Some(row) = rows
        .next()
        .map_err(|e| format!("Row fetching failed: {}", e))?
    {
        let val: String = row
            .get(0)
            .map_err(|e| format!("Get column failed: {}", e))?;
        let model: MiniTransformer = serde_json::from_str(&val)
            .map_err(|e| format!("Failed to deserialize model: {}", e))?;
        Ok(Some(model))
    } else {
        Ok(None)
    }
}

/// Log telemetry data of a verification request
pub fn log_telemetry(conn: &Connection, record: &TelemetryRecord<'_>) -> Result<(), String> {
    let now = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap()
        .as_secs() as i64;

    conn.execute(
        "INSERT INTO telemetry_logs (timestamp, point_count, score, is_human, webdriver, user_agent, ip_address, features_json, points_hash, is_high_confidence)
         VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9, ?10)",
        params![
            now,
            record.point_count as i64,
            record.score as f64,
            if record.is_human { 1 } else { 0 },
            if record.webdriver { 1 } else { 0 },
            record.user_agent,
            record.ip_address,
            record.features_json,
            record.points_hash,
            if record.is_high_confidence { 1 } else { 0 }
        ],
    ).map_err(|e| format!("Failed to insert telemetry log: {}", e))?;

    Ok(())
}

/// Check if telemetry points have already been used to prevent replay attacks
pub fn is_telemetry_replayed(conn: &Connection, points_hash: &str) -> Result<bool, String> {
    if points_hash.is_empty() {
        return Ok(false);
    }
    let mut stmt = conn
        .prepare("SELECT 1 FROM telemetry_logs WHERE points_hash = ?1 LIMIT 1")
        .map_err(|e| format!("Prepare query failed: {}", e))?;

    let exists = stmt
        .exists(params![points_hash])
        .map_err(|e| format!("Query points_hash failed: {}", e))?;

    Ok(exists)
}

/// Mark a token as validated atomically. Returns Ok(true) if the token was successfully
/// marked as used, and Ok(false) if it was already marked as used (due to primary key constraint violation).
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
pub fn prune_old_tokens(conn: &Connection, max_age_secs: u64) -> Result<usize, String> {
    let now = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap()
        .as_secs() as i64;

    let cutoff = now - (max_age_secs as i64);

    let deleted = conn
        .execute(
            "DELETE FROM used_tokens WHERE validated_at < ?1",
            params![cutoff],
        )
        .map_err(|e| format!("Failed to prune tokens: {}", e))?;

    Ok(deleted)
}

/// Delete old telemetry to bound database growth and replay-retention time.
pub fn prune_old_telemetry(conn: &Connection, max_age_secs: u64) -> Result<usize, String> {
    let now = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs() as i64;
    let cutoff = now.saturating_sub(max_age_secs.min(i64::MAX as u64) as i64);
    conn.execute(
        "DELETE FROM telemetry_logs WHERE timestamp < ?1",
        params![cutoff],
    )
    .map_err(|e| format!("Failed to prune telemetry: {e}"))
}

/// Get the most recent telemetry logs from the database to use as training data.
pub fn get_recent_telemetry(
    conn: &Connection,
    limit: usize,
) -> Result<Vec<([f32; 13], f32)>, String> {
    let mut stmt = conn
        .prepare("SELECT features_json, is_human FROM telemetry_logs WHERE is_high_confidence = 1 ORDER BY id DESC LIMIT ?1")
        .map_err(|e| format!("Prepare query failed: {}", e))?;

    let rows = stmt
        .query_map(params![limit as i64], |row| {
            let features_json: String = row.get(0)?;
            let is_human_int: i32 = row.get(1)?;
            let target = if is_human_int > 0 { 1.0f32 } else { 0.0f32 };
            Ok((features_json, target))
        })
        .map_err(|e| format!("Query telemetry failed: {}", e))?;

    let mut data = Vec::new();
    for (features_json, target) in rows.flatten() {
        if let Ok(features) = serde_json::from_str::<crate::features::FeatureVector>(&features_json)
        {
            data.push((features.to_array(), target));
        }
    }
    Ok(data)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::features::FeatureVector;

    #[test]
    fn test_log_and_get_recent_telemetry_high_confidence() {
        let conn = init_db(":memory:").unwrap();

        let features = FeatureVector {
            straightness: 0.8,
            avg_speed: 0.4,
            speed_var: 0.2,
            angular_jitter: 0.1,
            total_duration: 0.5,
            line_deviation: 0.05,
            point_count: 0.6,
            entropy: 0.3,
            accel_var: 0.2,
            curvature_change: 0.1,
            overshoot: 0.02,
            dwell_ratio: 0.05,
            timing_jitter: 0.1,
        };
        let features_json = serde_json::to_string(&features).unwrap();

        // 1. Log a low confidence telemetry entry
        log_telemetry(
            &conn,
            &TelemetryRecord {
                point_count: 20,
                score: 0.9,
                is_human: true,
                webdriver: false,
                user_agent: Some("Mozilla"),
                ip_address: Some("hashed-ip"),
                features_json: &features_json,
                points_hash: "hash1",
                is_high_confidence: false,
            },
        )
        .unwrap();

        // 2. Log a high confidence telemetry entry
        log_telemetry(
            &conn,
            &TelemetryRecord {
                point_count: 25,
                score: 0.95,
                is_human: true,
                webdriver: false,
                user_agent: Some("Mozilla"),
                ip_address: Some("hashed-ip"),
                features_json: &features_json,
                points_hash: "hash2",
                is_high_confidence: true,
            },
        )
        .unwrap();

        // Retrieve telemetry for training
        let dataset = get_recent_telemetry(&conn, 10).unwrap();

        // Assert only the high confidence record is returned
        assert_eq!(dataset.len(), 1);
        assert_eq!(dataset[0].1, 1.0); // is_human target
    }
}
