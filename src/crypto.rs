use hmac::{Hmac, Mac};
use sha2::Sha256;
use rand::{distributions::Alphanumeric, Rng};
use std::time::{SystemTime, UNIX_EPOCH};
use base64::{engine::general_purpose::URL_SAFE_NO_PAD, engine::general_purpose::STANDARD, Engine as _};

type HmacSha256 = Hmac<Sha256>;

pub fn decrypt_payload(obfuscated_b64: &str, salt: &str, key: &str) -> Result<String, String> {
    let decoded_bytes = STANDARD
        .decode(obfuscated_b64.trim().as_bytes())
        .map_err(|e| format!("Base64 decode error: {}", e))?;
    
    use sha2::{Digest, Sha256};
    let mut hasher = Sha256::new();
    hasher.update(format!("{}{}", salt, key).as_bytes());
    let derived_key = hasher.finalize();
    
    let decrypted_bytes: Vec<u8> = decoded_bytes
        .into_iter()
        .enumerate()
        .map(|(i, b)| b ^ derived_key[i % 32])
        .collect();
    
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_obfuscation_roundtrip() {
        let original = "{\"points\":[{\"x\":1.0,\"y\":2.0,\"t\":3.0}],\"webdriver\":false}";
        let salt = "random_salt_12345";
        let key = "some_random_key_67890";
        
        // Emulate client-side derivation and XOR
        use sha2::{Digest, Sha256};
        let mut hasher = Sha256::new();
        hasher.update(format!("{}{}", salt, key).as_bytes());
        let derived_key = hasher.finalize();
        
        let obfuscated_bytes: Vec<u8> = original.as_bytes()
            .iter()
            .enumerate()
            .map(|(i, b)| b ^ derived_key[i % 32])
            .collect();
        
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
