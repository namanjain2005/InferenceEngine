use serde::Deserialize;
use std::collections::HashMap;
use std::fs::File;
use std::io::{Read, Seek, SeekFrom};
use std::path::{Path, PathBuf};

#[derive(Debug, Deserialize, Clone)]
pub struct TensorMeta {
    pub dtype: String,
    pub shape: Vec<usize>,
    pub data_offsets: [usize; 2], // start  , end
}

#[derive(Debug, Clone)]
struct TensorLocation {
    meta: TensorMeta,
    shard: usize,
    data_start: u64,
}

pub struct SafeTensors {
    files: Vec<File>,
    tensors: HashMap<String, TensorLocation>,
}

impl SafeTensors {
    pub fn open<P: AsRef<Path>>(path: P) -> Result<Self, String> {
        Self::open_many(vec![path.as_ref().to_path_buf()])
    }

    /// Opens the standard Hugging Face single-file or sharded checkpoint layout.
    pub fn open_model_dir<P: AsRef<Path>>(dir: P) -> Result<Self, String> {
        let dir = dir.as_ref();
        let single = dir.join("model.safetensors");
        if single.is_file() {
            return Self::open(single);
        }
        let index_path = dir.join("model.safetensors.index.json");
        let index_file = File::open(&index_path).map_err(|e| {
            format!(
                "No model.safetensors or model.safetensors.index.json in {}: {e}",
                dir.display()
            )
        })?;
        let index: serde_json::Value = serde_json::from_reader(index_file)
            .map_err(|e| format!("Failed to parse {}: {e}", index_path.display()))?;
        let weight_map = index
            .get("weight_map")
            .and_then(serde_json::Value::as_object)
            .ok_or_else(|| format!("{} has no weight_map object", index_path.display()))?;
        let mut shard_names: Vec<String> = weight_map
            .values()
            .filter_map(serde_json::Value::as_str)
            .map(str::to_owned)
            .collect();
        shard_names.sort();
        shard_names.dedup();
        if shard_names.is_empty() {
            return Err("Safetensors index contains no shard files".into());
        }
        let paths = shard_names
            .iter()
            .map(|name| {
                let path = Path::new(name);
                if path.components().count() != 1
                    || path.file_name().and_then(|s| s.to_str()) != Some(name.as_str())
                {
                    return Err(format!("Invalid safetensors shard path in index: {name}"));
                }
                Ok(dir.join(path))
            })
            .collect::<Result<Vec<_>, _>>()?;
        Self::open_many(paths)
    }

    fn open_many(paths: Vec<PathBuf>) -> Result<Self, String> {
        let mut files = Vec::with_capacity(paths.len());
        let mut tensors = HashMap::new();
        for (shard, path) in paths.iter().enumerate() {
            let mut file = File::open(path)
                .map_err(|e| format!("Failed to open safetensors {}: {e}", path.display()))?;
            let mut header_len_bytes = [0u8; 8];
            file.read_exact(&mut header_len_bytes).map_err(|e| {
                format!("Failed to read header length from {}: {e}", path.display())
            })?;
            let header_len = u64::from_le_bytes(header_len_bytes);
            let header_size = usize::try_from(header_len)
                .map_err(|_| format!("Header too large in {}", path.display()))?;
            let mut json_bytes = vec![0u8; header_size];
            file.read_exact(&mut json_bytes)
                .map_err(|e| format!("Failed to read header JSON from {}: {e}", path.display()))?;
            let header: HashMap<String, serde_json::Value> = serde_json::from_slice(&json_bytes)
                .map_err(|e| format!("Failed to parse header JSON in {}: {e}", path.display()))?;
            for (name, value) in header {
                if name == "__metadata__" {
                    continue;
                }
                let meta: TensorMeta = serde_json::from_value(value).map_err(|e| {
                    format!(
                        "Failed to parse tensor metadata for '{name}' in {}: {e}",
                        path.display()
                    )
                })?;
                let location = TensorLocation {
                    meta,
                    shard,
                    data_start: 8 + header_len,
                };
                if tensors.insert(name.clone(), location).is_some() {
                    return Err(format!(
                        "Tensor '{name}' appears in more than one safetensors shard"
                    ));
                }
            }
            files.push(file);
        }
        Ok(Self { files, tensors })
    }

    pub fn tensor_names(&self) -> Vec<String> {
        let mut names: Vec<String> = self.tensors.keys().cloned().collect();
        names.sort();
        names
    }

    pub fn tensor_count(&self) -> usize {
        self.tensors.len()
    }

    pub fn get_meta(&self, name: &str) -> Option<&TensorMeta> {
        self.tensors.get(name).map(|location| &location.meta)
    }

    pub fn file_count(&self) -> usize {
        self.files.len()
    }

    pub fn read_tensor_f32(&mut self, name: &str) -> Result<Vec<f32>, String> {
        let meta = self
            .tensors
            .get(name)
            .ok_or_else(|| format!("Tensor '{}' not found in safetensors", name))?
            .clone();

        let start = meta.meta.data_offsets[0];
        let end = meta.meta.data_offsets[1];
        let byte_len = end - start;

        self.files[meta.shard]
            .seek(SeekFrom::Start(meta.data_start + start as u64))
            .map_err(|e| format!("Seek failed for '{}': {}", name, e))?;

        let mut raw_bytes = vec![0u8; byte_len];
        self.files[meta.shard]
            .read_exact(&mut raw_bytes)
            .map_err(|e| format!("Read failed for '{}': {}", name, e))?;

        match meta.meta.dtype.as_str() {
            // TODO : Dont just convert everyhting to F32
            "BF16" => {
                let count = byte_len / 2;
                let mut data = Vec::with_capacity(count);
                for i in 0..count {
                    let b = u16::from_le_bytes([raw_bytes[2 * i], raw_bytes[2 * i + 1]]);
                    data.push(bf16_to_f32(b));
                }
                Ok(data)
            }
            "F16" => {
                let count = byte_len / 2;
                let mut data = Vec::with_capacity(count);
                for i in 0..count {
                    let b = u16::from_le_bytes([raw_bytes[2 * i], raw_bytes[2 * i + 1]]);
                    data.push(f16_to_f32(b));
                }
                Ok(data)
            }
            "F32" => {
                let count = byte_len / 4;
                let mut data = Vec::with_capacity(count);
                for i in 0..count {
                    let b = [
                        raw_bytes[4 * i],
                        raw_bytes[4 * i + 1],
                        raw_bytes[4 * i + 2],
                        raw_bytes[4 * i + 3],
                    ];
                    data.push(f32::from_le_bytes(b));
                }
                Ok(data)
            }
            other => Err(format!(
                "Unsupported dtype '{}' for tensor '{}'",
                other, name
            )),
        }
    }

    /// Read a specific row (e.g., token embedding) without loading the entire tensor.
    pub fn read_embedding_row(
        &mut self,
        name: &str,
        token_id: usize,
        hidden_size: usize,
    ) -> Result<Vec<f32>, String> {
        let meta = self
            .tensors
            .get(name)
            .ok_or_else(|| format!("Embedding tensor '{}' not found", name))?
            .clone();

        let bytes_per_elem = match meta.meta.dtype.as_str() {
            "BF16" | "F16" => 2,
            "F32" => 4,
            other => return Err(format!("Unsupported dtype: {}", other)),
        };

        if meta.meta.shape.len() != 2
            || meta.meta.shape[1] != hidden_size
            || token_id >= meta.meta.shape[0]
        {
            return Err(format!(
                "Embedding row {token_id} is outside tensor '{name}' shape {:?}",
                meta.meta.shape
            ));
        }
        let row_bytes = hidden_size
            .checked_mul(bytes_per_elem)
            .ok_or("Embedding row size overflow")?;
        let offset = meta.meta.data_offsets[0]
            .checked_add(
                token_id
                    .checked_mul(row_bytes)
                    .ok_or("Embedding offset overflow")?,
            )
            .ok_or("Embedding offset overflow")?;

        self.files[meta.shard]
            .seek(SeekFrom::Start(meta.data_start + offset as u64))
            .map_err(|e| format!("Seek to embedding row {} failed: {}", token_id, e))?;

        let mut raw = vec![0u8; row_bytes];
        self.files[meta.shard]
            .read_exact(&mut raw)
            .map_err(|e| format!("Read embedding row {} failed: {}", token_id, e))?;

        let mut row = Vec::with_capacity(hidden_size);
        match meta.meta.dtype.as_str() {
            "BF16" => {
                for i in 0..hidden_size {
                    let b = u16::from_le_bytes([raw[2 * i], raw[2 * i + 1]]);
                    row.push(bf16_to_f32(b));
                }
            }
            "F16" => {
                for i in 0..hidden_size {
                    let b = u16::from_le_bytes([raw[2 * i], raw[2 * i + 1]]);
                    row.push(f16_to_f32(b));
                }
            }
            "F32" => {
                for i in 0..hidden_size {
                    let b = [raw[4 * i], raw[4 * i + 1], raw[4 * i + 2], raw[4 * i + 3]];
                    row.push(f32::from_le_bytes(b));
                }
            }
            _ => unreachable!(),
        }

        Ok(row)
    }
}

#[inline(always)]
pub fn bf16_to_f32(b: u16) -> f32 {
    f32::from_bits((b as u32) << 16)
}

#[inline(always)]
pub fn f16_to_f32(h: u16) -> f32 {
    let s = ((h >> 15) & 0x0001) as u32;
    let e = ((h >> 10) & 0x001f) as u32;
    let f = (h & 0x03ff) as u32;
    if e == 0 {
        if f == 0 {
            f32::from_bits(s << 31)
        } else {
            let mut e = 113;
            let mut f = f;
            while (f & 0x0400) == 0 {
                f <<= 1;
                e -= 1;
            }
            f &= 0x03ff;
            f32::from_bits((s << 31) | (e << 23) | (f << 13))
        }
    } else if e == 31 {
        f32::from_bits((s << 31) | 0x7f800000 | (f << 13))
    } else {
        f32::from_bits((s << 31) | ((e + 112) << 23) | (f << 13))
    }
}
