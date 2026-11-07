use std::collections::HashMap;
use std::sync::{Arc, Mutex};
use std::time::Duration;

use tokio::task::AbortHandle;

use super::base::{AnyModel, InferenceEntry};
use crate::allocator;
use crate::config::clean_name;

type Key = (&'static str, String, Option<i64>);

struct Expiry {
    generation: u64,
    handle: AbortHandle,
}

#[derive(Default)]
struct Inner {
    models: HashMap<Key, Arc<dyn AnyModel>>,
    expiries: HashMap<Key, Expiry>,
    generation: u64,
}

#[derive(Clone, Default)]
pub struct ModelCache {
    inner: Arc<Mutex<Inner>>,
}

impl ModelCache {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn get(&self, entry: &InferenceEntry, ttl: Option<i64>) -> Arc<dyn AnyModel> {
        let key: Key = (
            entry.model.name(),
            clean_name(&entry.name),
            entry.model.graph(&entry.options),
        );
        let mut inner = self.inner.lock().expect("model cache lock");
        let mut ttl = ttl;
        if !inner.models.contains_key(&key) {
            inner
                .models
                .insert(key.clone(), entry.model.create(&entry.name, &entry.options));
        } else if !inner.expiries.contains_key(&key) {
            ttl = None;
        }
        if let Some(ttl) = ttl.filter(|ttl| *ttl != 0) {
            if let Some(previous) = inner.expiries.remove(&key) {
                previous.handle.abort();
            }
            inner.generation += 1;
            let generation = inner.generation;
            let cache = self.clone();
            let evicted = key.clone();
            let handle = tokio::spawn(async move {
                tokio::time::sleep(Duration::from_secs(ttl.max(0) as u64)).await;
                cache.evict(&evicted, generation);
            })
            .abort_handle();
            inner
                .expiries
                .insert(key.clone(), Expiry { generation, handle });
        }
        inner.models[&key].clone()
    }

    pub fn clear(&self) {
        let mut inner = self.inner.lock().expect("model cache lock");
        if inner.models.is_empty() {
            return;
        }
        inner.models.clear();
        for (_, expiry) in inner.expiries.drain() {
            expiry.handle.abort();
        }
        drop(inner);
        allocator::release();
    }

    fn evict(&self, key: &Key, generation: u64) {
        let mut inner = self.inner.lock().expect("model cache lock");
        if !inner.models.contains_key(key)
            || inner
                .expiries
                .get(key)
                .is_none_or(|expiry| expiry.generation != generation)
        {
            return;
        }
        inner.models.remove(key);
        inner.expiries.remove(key);
        drop(inner);
        allocator::release();
    }
}
