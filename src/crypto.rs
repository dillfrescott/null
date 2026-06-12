use hmac::{Hmac, Mac};
use sha2::Sha256;
use rand::{distributions::Alphanumeric, Rng};
use std::time::{SystemTime, UNIX_EPOCH};
use base64::{engine::general_purpose::URL_SAFE_NO_PAD, engine::general_purpose::STANDARD, Engine as _};

type HmacSha256 = Hmac<Sha256>;

pub fn decrypt_payload(obfuscated_b64: &str, key: u8) -> Result<String, String> {
    let decoded_bytes = STANDARD
        .decode(obfuscated_b64.trim().as_bytes())
        .map_err(|e| format!("Base64 decode error: {}", e))?;
    
    let decrypted_bytes: Vec<u8> = decoded_bytes.into_iter().map(|b| b ^ key).collect();
    
    String::from_utf8(decrypted_bytes)
        .map_err(|e| format!("UTF8 decode error: {}", e))
}

pub fn generate_token(secret: &[u8], score: f32) -> String {
    let now = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap()
        .as_millis();
    
    let nonce: String = rand::thread_rng()
        .sample_iter(&Alphanumeric)
        .take(16)
        .map(char::from)
        .collect();

    let payload = format!("{}.{:.4}.{}", now, score, nonce);
    
    let mut mac = HmacSha256::new_from_slice(secret).expect("HMAC can take key of any size");
    mac.update(payload.as_bytes());
    let signature = URL_SAFE_NO_PAD.encode(mac.finalize().into_bytes());

    format!("{}.{}", payload, signature)
}

pub fn verify_token(secret: &[u8], token: &str, max_age_secs: u64) -> Result<f32, String> {
    let parts: Vec<&str> = token.split('.').collect();
    let (timestamp_str, score_str, nonce_str, signature_str) = match parts.len() {
        4 => (parts[0], parts[1].to_string(), parts[2], parts[3]),
        5 => (parts[0], format!("{}.{}", parts[1], parts[2]), parts[3], parts[4]),
        _ => return Err("Invalid token format. Must be timestamp.score.nonce.signature".to_string()),
    };

    // Reconstruct payload
    let payload = format!("{}.{}.{}", timestamp_str, score_str, nonce_str);

    // Verify HMAC signature
    let decoded_signature = URL_SAFE_NO_PAD
        .decode(signature_str.as_bytes())
        .map_err(|e| format!("Failed to decode signature: {}", e))?;

    let mut mac = HmacSha256::new_from_slice(secret).expect("HMAC can take key of any size");
    mac.update(payload.as_bytes());
    
    if mac.verify_slice(&decoded_signature).is_err() {
        return Err("Invalid signature".to_string());
    }

    // Verify timestamp (check for expiration / replay attacks)
    let token_time = timestamp_str
        .parse::<u64>()
        .map_err(|e| format!("Invalid timestamp: {}", e))?;
    
    let now = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap()
        .as_millis() as u64;

    if now < token_time {
        return Err("Token timestamp is in the future".to_string());
    }

    let age_secs = (now - token_time) / 1000;
    if age_secs > max_age_secs {
        return Err(format!("Token expired. Age: {}s, Max age: {}s", age_secs, max_age_secs));
    }

    // Parse score
    let score = score_str
        .parse::<f32>()
        .map_err(|e| format!("Invalid score: {}", e))?;

    Ok(score)
}
