use base64::{
    Engine as _, engine::general_purpose::STANDARD, engine::general_purpose::URL_SAFE_NO_PAD,
};
use hmac::{Hmac, Mac};
use rand::{Rng, distributions::Alphanumeric};
use sha2::Sha256;
use std::time::{SystemTime, UNIX_EPOCH};

type HmacSha256 = Hmac<Sha256>;
const MAX_ENCRYPTED_PAYLOAD_BYTES: usize = 48 * 1024;

pub fn decrypt_payload(obfuscated_b64: &str, salt: &str, key: &str) -> Result<String, String> {
    if obfuscated_b64.len() > MAX_ENCRYPTED_PAYLOAD_BYTES * 4 / 3 + 4 {
        return Err("Encrypted payload is too large".to_string());
    }
    let decoded_bytes = STANDARD
        .decode(obfuscated_b64.trim().as_bytes())
        .map_err(|e| format!("Base64 decode error: {}", e))?;
    if decoded_bytes.len() > MAX_ENCRYPTED_PAYLOAD_BYTES {
        return Err("Encrypted payload is too large".to_string());
    }

    use sha2::{Digest, Sha256};
    let mut hasher = Sha256::new();
    hasher.update(format!("{}{}", salt, key).as_bytes());
    let derived_key = hasher.finalize();

    let mut decrypted_bytes = Vec::with_capacity(decoded_bytes.len());

    for (block_idx, chunk) in decoded_bytes.chunks(32).enumerate() {
        let mut block_hasher = Sha256::new();
        block_hasher.update(derived_key);
        block_hasher.update((block_idx as u32).to_be_bytes());
        let block_key = block_hasher.finalize();

        for (i, &b) in chunk.iter().enumerate() {
            decrypted_bytes.push(b ^ block_key[i]);
        }
    }

    String::from_utf8(decrypted_bytes).map_err(|e| format!("UTF8 decode error: {}", e))
}

pub fn generate_token(secret: &[u8], score: f32) -> String {
    let now = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_millis();

    let nonce: String = rand::thread_rng()
        .sample_iter(&Alphanumeric)
        .take(16)
        .map(char::from)
        .collect();

    let score = if score.is_finite() {
        score.clamp(0.0, 1.0)
    } else {
        0.0
    };
    let payload = format!("{}|{:.4}|{}", now, score, nonce);

    let mut mac = HmacSha256::new_from_slice(secret).expect("HMAC can take key of any size");
    mac.update(payload.as_bytes());
    let signature = URL_SAFE_NO_PAD.encode(mac.finalize().into_bytes());

    format!("{}|{}", payload, signature)
}

pub fn verify_token(secret: &[u8], token: &str, max_age_secs: u64) -> Result<f32, String> {
    if token.len() > 256 {
        return Err("Invalid token length".to_string());
    }
    let mut parts = token.split('|');
    let (Some(timestamp_str), Some(score_str), Some(nonce_str), Some(signature_str)) =
        (parts.next(), parts.next(), parts.next(), parts.next())
    else {
        return Err("Invalid token format. Must be timestamp|score|nonce|signature".to_string());
    };
    if parts.next().is_some() || nonce_str.len() != 16 {
        return Err("Invalid token format. Must be timestamp|score|nonce|signature".to_string());
    }

    // Reconstruct payload
    let payload = format!("{}|{}|{}", timestamp_str, score_str, nonce_str);

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
        .unwrap_or_default()
        .as_millis() as u64;

    if now < token_time {
        return Err("Token timestamp is in the future".to_string());
    }

    let age_secs = (now - token_time) / 1000;
    if age_secs > max_age_secs {
        return Err(format!(
            "Token expired. Age: {}s, Max age: {}s",
            age_secs, max_age_secs
        ));
    }

    // Parse score
    let score = score_str
        .parse::<f32>()
        .map_err(|e| format!("Invalid score: {}", e))?;
    if !score.is_finite() || !(0.0..=1.0).contains(&score) {
        return Err("Token score is out of range".to_string());
    }

    Ok(score)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_obfuscation_roundtrip() {
        let original = "{\"points\":[{\"x\":1.0,\"y\":2.0,\"t\":3.0}],\"webdriver\":false}";
        let salt = "random_salt_12345";
        let key = "some_random_key_67890";

        // Emulate CTR mode encryption
        use sha2::{Digest, Sha256};
        let mut hasher = Sha256::new();
        hasher.update(format!("{}{}", salt, key).as_bytes());
        let derived_key = hasher.finalize();

        let mut obfuscated_bytes = Vec::with_capacity(original.len());

        for (block_idx, chunk) in original.as_bytes().chunks(32).enumerate() {
            let mut block_hasher = Sha256::new();
            block_hasher.update(derived_key);
            block_hasher.update((block_idx as u32).to_be_bytes());
            let block_key = block_hasher.finalize();

            for (i, &b) in chunk.iter().enumerate() {
                obfuscated_bytes.push(b ^ block_key[i]);
            }
        }

        let obfuscated_b64 = base64::engine::general_purpose::STANDARD.encode(obfuscated_bytes);

        let decrypted = decrypt_payload(&obfuscated_b64, salt, key).unwrap();
        assert_eq!(decrypted, original);
    }

    #[test]
    fn test_token_roundtrip() {
        let secret = b"my_secret_key_which_is_very_long";
        let score = 0.85;
        let token = generate_token(secret, score);

        let verified_score = verify_token(secret, &token, 10).unwrap();
        assert!((verified_score - score).abs() < 1e-4);
    }
}
