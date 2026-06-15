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
        
        // Prune expired entries to keep memory footprint bounded, but only
        // occasionally to avoid O(N) scan overhead on every insertion.
        if map.len() > 1000 || rand::random::<f64>() < 0.02 {
            map.retain(|_, &mut exp| exp > now);
        }

        if map.contains_key(&key) {
            false
        } else {
            map.insert(key, expires_at);
            true
        }
    }

    pub fn contains(&self, key: &str) -> bool {
        let now = SystemTime::now().duration_since(UNIX_EPOCH).unwrap().as_secs();
        let map = self.entries.read().unwrap();
        map.get(key).map(|&exp| exp > now).unwrap_or(false)
    }

    pub fn insert(&self, key: String, ttl_secs: u64) {
        let now = SystemTime::now().duration_since(UNIX_EPOCH).unwrap().as_secs();
        let mut map = self.entries.write().unwrap();
        map.insert(key, now + ttl_secs);
    }

    pub fn remove(&self, key: &str) -> bool {
        let mut map = self.entries.write().unwrap();
        map.remove(key).is_some()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_memory_cache_operations() {
        let cache = MemoryCache::new();
        let key = "test_key".to_string();

        // Check insert and contains
        assert!(!cache.contains(&key));
        cache.insert(key.clone(), 2);
        assert!(cache.contains(&key));

        // Check check_and_insert
        let key2 = "test_key2".to_string();
        assert!(cache.check_and_insert(key2.clone(), 5));
        assert!(!cache.check_and_insert(key2.clone(), 5));

        // Check remove
        assert!(cache.remove(&key));
        assert!(!cache.contains(&key));
        assert!(!cache.remove(&key));
    }
}
