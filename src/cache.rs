use std::collections::HashMap;
use std::sync::RwLock;
use std::time::{SystemTime, UNIX_EPOCH};

pub struct MemoryCache {
    entries: RwLock<HashMap<String, u64>>,
}

impl MemoryCache {
    pub fn new() -> Self {
        Self {
            entries: RwLock::new(HashMap::new()),
        }
    }

    /// Checks if a key has been used/seen. 
    /// If not seen, inserts it with the given TTL (in seconds) and returns true.
    /// If already seen and not expired, returns false.
    pub fn check_and_insert(&self, key: String, ttl_secs: u64) -> bool {
        let now = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_secs();
        
        let expires_at = now + ttl_secs;

        let mut map = self.entries.write().unwrap();
        
        // Prune expired entries to keep memory footprint bounded
        map.retain(|_, &mut exp| exp > now);

        if map.contains_key(&key) {
            false
        } else {
            map.insert(key, expires_at);
            true
        }
    }
}
