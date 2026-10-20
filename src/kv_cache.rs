//! Configurable KV-cache strategy interfaces.
//!
//! A cache strategy owns its storage layout and how that storage is written
//! and exposed to attention. This lets future strategies use paged storage,
//! quantized values, sliding windows, or shared/offloaded state without
//! teaching the model about their physical layout.

use crate::kernels::ops::{AttentionKernel, VectorKernels};
use cudarc::driver::{CudaSlice, CudaStream};
use std::sync::Arc;

/// Model dimensions required to create one request's KV cache.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct KVCacheShape {
    pub num_layers: usize,
    pub kv_dim: usize,
}

/// Factory for cache states. Implement this to add a selectable cache policy.
pub trait KVCacheStrategy: Send + Sync {
    fn name(&self) -> &'static str;

    /// Construct a request-local cache state using this strategy.
    fn new(
        &self,
        stream: &Arc<CudaStream>,
        shape: KVCacheShape,
        initial_capacity: usize,
    ) -> Result<Box<dyn KVCacheState>, String>;
}

/// Operations the model requires from one request's cache state.
///
/// Methods operate on logical token positions. Implementations own the
/// physical representation and can choose suitable append and attention
/// paths for that representation.
pub trait KVCacheState {
    fn strategy_name(&self) -> &'static str;
    fn num_layers(&self) -> usize;
    fn capacity(&self) -> usize;

    /// Ensure positions `[0, needed)` can be represented, retaining the first
    /// `preserve` positions already written. Paged strategies can reserve
    /// blocks here; sliding-window strategies may keep bounded storage.
    fn ensure_capacity(
        &mut self,
        stream: &Arc<CudaStream>,
        needed: usize,
        preserve: usize,
    ) -> Result<(), String>;

    /// Append one generated token's K/V values for a layer.
    fn append(
        &mut self,
        layer: usize,
        position: usize,
        key: &CudaSlice<f32>,
        value: &CudaSlice<f32>,
        vector: &VectorKernels,
        stream: &Arc<CudaStream>,
    ) -> Result<(), String>;

    fn append_prompt(
        &mut self,
        layer: usize,
        start_position: usize,
        batch: usize,
        keys: &CudaSlice<f32>,
        values: &CudaSlice<f32>,
        vector: &VectorKernels,
        stream: &Arc<CudaStream>,
    ) -> Result<(), String>;

    fn attend_decode(
        &mut self,
        layer: usize,
        query: &CudaSlice<f32>,
        sequence_length: usize,
        num_query_heads: usize,
        num_kv_heads: usize,
        head_dim: usize,
        attention: &AttentionKernel,
        stream: &Arc<CudaStream>,
    ) -> Result<CudaSlice<f32>, String>;

    fn attend_prefill(
        &mut self,
        layer: usize,
        query: &CudaSlice<f32>,
        start_position: usize,
        batch: usize,
        num_query_heads: usize,
        num_kv_heads: usize,
        head_dim: usize,
        attention: &AttentionKernel,
        stream: &Arc<CudaStream>,
    ) -> Result<CudaSlice<f32>, String>;
}

/// Runtime cache configuration, selected by the Rust caller at model load.
#[derive(Clone)]
pub struct KVCacheConfig {
    strategy: Arc<dyn KVCacheStrategy>,
    initial_capacity: usize,
}

impl KVCacheConfig {
    pub fn new(
        strategy: Arc<dyn KVCacheStrategy>,
        initial_capacity: usize,
    ) -> Result<Self, String> {
        if initial_capacity == 0 {
            return Err("KV cache initial capacity must be non-zero".into());
        }
        Ok(Self {
            strategy,
            initial_capacity,
        })
    }

    pub fn strategy_name(&self) -> &'static str {
        self.strategy.name()
    }

    pub fn initial_capacity(&self) -> usize {
        self.initial_capacity
    }

    pub fn create_cache(
        &self,
        stream: &Arc<CudaStream>,
        shape: KVCacheShape,
    ) -> Result<KVCache, String> {
        KVCache::new(self, stream, shape)
    }

    pub fn with_initial_capacity(&self, initial_capacity: usize) -> Result<Self, String> {
        Self::new(self.strategy.clone(), initial_capacity)
    }
}

/// Stable model-facing handle for a strategy-selected request cache.
///
/// The facade keeps cache construction and operations uniform while the
/// selected [`KVCacheState`] owns its physical representation.
pub struct KVCache {
    state: Box<dyn KVCacheState>,
}

impl KVCache {
    pub fn new(
        config: &KVCacheConfig,
        stream: &Arc<CudaStream>,
        shape: KVCacheShape,
    ) -> Result<Self, String> {
        let state = config
            .strategy
            .new(stream, shape, config.initial_capacity)?;
        Ok(Self { state })
    }

    pub fn strategy_name(&self) -> &'static str {
        self.state.strategy_name()
    }

    pub fn num_layers(&self) -> usize {
        self.state.num_layers()
    }

    pub fn capacity(&self) -> usize {
        self.state.capacity()
    }

    pub fn ensure_capacity(
        &mut self,
        stream: &Arc<CudaStream>,
        needed: usize,
        preserve: usize,
    ) -> Result<(), String> {
        self.state.ensure_capacity(stream, needed, preserve)
    }

    pub fn append(
        &mut self,
        layer: usize,
        position: usize,
        key: &CudaSlice<f32>,
        value: &CudaSlice<f32>,
        vector: &VectorKernels,
        stream: &Arc<CudaStream>,
    ) -> Result<(), String> {
        self.state
            .append(layer, position, key, value, vector, stream)
    }

    pub fn append_prompt(
        &mut self,
        layer: usize,
        start_position: usize,
        batch: usize,
        keys: &CudaSlice<f32>,
        values: &CudaSlice<f32>,
        vector: &VectorKernels,
        stream: &Arc<CudaStream>,
    ) -> Result<(), String> {
        self.state
            .append_prompt(layer, start_position, batch, keys, values, vector, stream)
    }

    pub fn attend_decode(
        &mut self,
        layer: usize,
        query: &CudaSlice<f32>,
        sequence_length: usize,
        num_query_heads: usize,
        num_kv_heads: usize,
        head_dim: usize,
        attention: &AttentionKernel,
        stream: &Arc<CudaStream>,
    ) -> Result<CudaSlice<f32>, String> {
        self.state.attend_decode(
            layer,
            query,
            sequence_length,
            num_query_heads,
            num_kv_heads,
            head_dim,
            attention,
            stream,
        )
    }

    pub fn attend_prefill(
        &mut self,
        layer: usize,
        query: &CudaSlice<f32>,
        start_position: usize,
        batch: usize,
        num_query_heads: usize,
        num_kv_heads: usize,
        head_dim: usize,
        attention: &AttentionKernel,
        stream: &Arc<CudaStream>,
    ) -> Result<CudaSlice<f32>, String> {
        self.state.attend_prefill(
            layer,
            query,
            start_position,
            batch,
            num_query_heads,
            num_kv_heads,
            head_dim,
            attention,
            stream,
        )
    }
}

impl Default for KVCacheConfig {
    fn default() -> Self {
        Self {
            strategy: Arc::new(NaiveKVCacheStrategy),
            initial_capacity: DEFAULT_KV_CACHE_CAPACITY,
        }
    }
}

/// Initial allocation for the current contiguous cache implementation.
pub const DEFAULT_KV_CACHE_CAPACITY: usize = 2048;

/// Current baseline: one contiguous K and V allocation per model layer.
#[derive(Clone, Copy, Debug, Default)]
pub struct NaiveKVCacheStrategy;

impl KVCacheStrategy for NaiveKVCacheStrategy {
    fn name(&self) -> &'static str {
        "naive"
    }

    fn new(
        &self,
        stream: &Arc<CudaStream>,
        shape: KVCacheShape,
        initial_capacity: usize,
    ) -> Result<Box<dyn KVCacheState>, String> {
        Ok(Box::new(NaiveKVCache::new(
            stream,
            shape.num_layers,
            initial_capacity,
            shape.kv_dim,
        )?))
    }
}

struct LayerKVCache {
    k: CudaSlice<f32>,
    v: CudaSlice<f32>,
}

struct NaiveKVCache {
    layers: Vec<LayerKVCache>,
    capacity: usize,
    kv_dim: usize,
}

impl NaiveKVCache {
    fn new(
        stream: &Arc<CudaStream>,
        num_layers: usize,
        capacity: usize,
        kv_dim: usize,
    ) -> Result<Self, String> {
        if num_layers == 0 || capacity == 0 || kv_dim == 0 {
            return Err("KV cache layers, capacity, and dimension must be non-zero".into());
        }
        let len = capacity
            .checked_mul(kv_dim)
            .ok_or("KV cache size overflow")?;
        let mut layers = Vec::with_capacity(num_layers);
        for _ in 0..num_layers {
            let k = stream
                .alloc_zeros::<f32>(len)
                .map_err(|e| format!("Allocating GPU key cache failed: {e}"))?;
            let v = stream
                .alloc_zeros::<f32>(len)
                .map_err(|e| format!("Allocating GPU value cache failed: {e}"))?;
            layers.push(LayerKVCache { k, v });
        }
        Ok(Self {
            layers,
            capacity,
            kv_dim,
        })
    }

    fn layer(&self, layer: usize) -> Result<&LayerKVCache, String> {
        self.layers
            .get(layer)
            .ok_or_else(|| format!("KV cache layer {layer} is out of range"))
    }

    fn layer_mut(&mut self, layer: usize) -> Result<&mut LayerKVCache, String> {
        self.layers
            .get_mut(layer)
            .ok_or_else(|| format!("KV cache layer {layer} is out of range"))
    }
}

impl KVCacheState for NaiveKVCache {
    fn strategy_name(&self) -> &'static str {
        "naive"
    }

    fn num_layers(&self) -> usize {
        self.layers.len()
    }

    fn capacity(&self) -> usize {
        self.capacity
    }

    fn ensure_capacity(
        &mut self,
        stream: &Arc<CudaStream>,
        needed: usize,
        preserve: usize,
    ) -> Result<(), String> {
        if needed <= self.capacity {
            return Ok(());
        }
        let old_capacity = self.capacity;
        if preserve > old_capacity {
            return Err(format!(
                "Cannot preserve {preserve} KV rows in a {old_capacity}-row cache"
            ));
        }
        let new_capacity = needed.max(
            old_capacity
                .checked_mul(2)
                .ok_or("KV cache capacity overflow")?,
        );
        let new_len = new_capacity
            .checked_mul(self.kv_dim)
            .ok_or("KV cache size overflow")?;
        let copy_len = preserve
            .checked_mul(self.kv_dim)
            .ok_or("KV cache copy size overflow")?;
        for layer in &mut self.layers {
            let mut k = stream
                .alloc_zeros::<f32>(new_len)
                .map_err(|e| format!("Growing GPU key cache failed: {e}"))?;
            let mut v = stream
                .alloc_zeros::<f32>(new_len)
                .map_err(|e| format!("Growing GPU value cache failed: {e}"))?;
            if copy_len > 0 {
                stream
                    .memcpy_dtod(&layer.k.slice(..copy_len), &mut k.slice_mut(..copy_len))
                    .map_err(|e| format!("Copying existing key cache failed: {e}"))?;
                stream
                    .memcpy_dtod(&layer.v.slice(..copy_len), &mut v.slice_mut(..copy_len))
                    .map_err(|e| format!("Copying existing value cache failed: {e}"))?;
            }
            layer.k = k;
            layer.v = v;
        }
        self.capacity = new_capacity;
        Ok(())
    }

    fn append(
        &mut self,
        layer: usize,
        position: usize,
        key: &CudaSlice<f32>,
        value: &CudaSlice<f32>,
        vector: &VectorKernels,
        stream: &Arc<CudaStream>,
    ) -> Result<(), String> {
        let kv_dim = self.kv_dim;
        let cache = self.layer_mut(layer)?;
        vector.append_cache(
            stream,
            key,
            value,
            &mut cache.k,
            &mut cache.v,
            position,
            kv_dim,
        )
    }

    fn append_prompt(
        &mut self,
        layer: usize,
        start_position: usize,
        batch: usize,
        keys: &CudaSlice<f32>,
        values: &CudaSlice<f32>,
        vector: &VectorKernels,
        stream: &Arc<CudaStream>,
    ) -> Result<(), String> {
        let kv_dim = self.kv_dim;
        let cache = self.layer_mut(layer)?;
        vector.append_cache_batch(
            stream,
            keys,
            values,
            &mut cache.k,
            &mut cache.v,
            start_position,
            batch,
            kv_dim,
        )
    }

    fn attend_decode(
        &mut self,
        layer: usize,
        query: &CudaSlice<f32>,
        sequence_length: usize,
        num_query_heads: usize,
        num_kv_heads: usize,
        head_dim: usize,
        attention: &AttentionKernel,
        stream: &Arc<CudaStream>,
    ) -> Result<CudaSlice<f32>, String> {
        let cache = self.layer(layer)?;
        attention.forward(
            stream,
            query,
            &cache.k,
            &cache.v,
            sequence_length,
            num_query_heads,
            num_kv_heads,
            head_dim,
        )
    }

    fn attend_prefill(
        &mut self,
        layer: usize,
        query: &CudaSlice<f32>,
        start_position: usize,
        batch: usize,
        num_query_heads: usize,
        num_kv_heads: usize,
        head_dim: usize,
        attention: &AttentionKernel,
        stream: &Arc<CudaStream>,
    ) -> Result<CudaSlice<f32>, String> {
        let cache = self.layer(layer)?;
        attention.forward_prefill(
            stream,
            query,
            &cache.k,
            &cache.v,
            start_position,
            batch,
            num_query_heads,
            num_kv_heads,
            head_dim,
        )
    }
}
