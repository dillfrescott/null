use std::collections::HashMap;
use std::sync::RwLock;
use std::time::{SystemTime, UNIX_EPOCH};

const PRUNE_THRESHOLD: usize = 1_000;
const MAX_ENTRIES: usize = 100_000;

pub struct MemoryCache {
    entries: RwLock<HashMap<String, u64>>,
}

impl MemoryCache {
    pub fn new() -> Self {
        Self {
            entries: RwLock::new(HashMap::new()),
        }
    }

    /// Atomically inserts an unexpired key and returns true. Existing, unexpired
    /// keys return false. Expired entries can be reused immediately.
    pub fn check_and_insert(&self, key: String, ttl_secs: u64) -> bool {
        let now = unix_time_secs();
        let expires_at = now.saturating_add(ttl_secs);
        let mut map = self.entries.write().unwrap_or_else(|e| e.into_inner());

        if map.len() >= PRUNE_THRESHOLD {
            map.retain(|_, expiry| *expiry > now);
        }

        if map.get(&key).is_some_and(|expiry| *expiry > now) {
            return false;
        }

        // Fail closed rather than silently dropping replay-protection entries.
        if map.len() >= MAX_ENTRIES && !map.contains_key(&key) {
            return false;
        }

        map.insert(key, expires_at);
        true
    }

    pub fn contains(&self, key: &str) -> bool {
        let now = unix_time_secs();
        let map = self.entries.read().unwrap_or_else(|e| e.into_inner());
        map.get(key).is_some_and(|expiry| *expiry > now)
    }

    pub fn insert(&self, key: String, ttl_secs: u64) -> bool {
        let now = unix_time_secs();
        let mut map = self.entries.write().unwrap_or_else(|e| e.into_inner());
        if map.len() >= PRUNE_THRESHOLD {
            map.retain(|_, expiry| *expiry > now);
        }
        if map.len() >= MAX_ENTRIES && !map.contains_key(&key) {
            return false;
        }
        map.insert(key, now.saturating_add(ttl_secs));
        true
    }

    pub fn remove(&self, key: &str) -> bool {
        let mut map = self.entries.write().unwrap_or_else(|e| e.into_inner());
        map.remove(key).is_some()
    }
}

fn unix_time_secs() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn memory_cache_operations_and_expiry() {
        let cache = MemoryCache::new();
        let key = "test_key".to_string();

        assert!(!cache.contains(&key));
        assert!(cache.insert(key.clone(), 2));
        assert!(cache.contains(&key));

        let key2 = "test_key2".to_string();
        assert!(cache.check_and_insert(key2.clone(), 5));
        assert!(!cache.check_and_insert(key2, 5));

        let expired = "expired".to_string();
        assert!(cache.check_and_insert(expired.clone(), 0));
        assert!(!cache.contains(&expired));
        assert!(cache.check_and_insert(expired, 1));

        assert!(cache.remove(&key));
        assert!(!cache.contains(&key));
        assert!(!cache.remove(&key));
    }
}
