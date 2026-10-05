use serde::Deserialize;
use std::collections::HashMap;
use std::fs::File;
use std::io::{Read, Seek, SeekFrom};
use std::path::Path;

#[derive(Debug, Deserialize, Clone)]
pub struct TensorMeta {
    pub dtype: String,
    pub shape: Vec<usize>,
    pub data_offsets: [usize; 2],
}

pub struct SafeTensors {
    file: File,
    pub header_len: u64,
    pub tensors: HashMap<String, TensorMeta>,
    pub data_start: u64,
}

impl SafeTensors {
    pub fn open<P: AsRef<Path>>(path: P) -> Result<Self, String> {
        let mut file = File::open(path.as_ref()).map_err(|e| format!("Failed to open safetensors: {}", e))?;

        // 1. Read first 8 bytes (little endian u64 header length)
        let mut header_len_bytes = [0u8; 8];
        file.read_exact(&mut header_len_bytes)
            .map_err(|e| format!("Failed to read header length: {}", e))?;
        let header_len = u64::from_le_bytes(header_len_bytes);

        // 2. Read JSON header
        let mut json_bytes = vec![0u8; header_len as usize];
        file.read_exact(&mut json_bytes)
            .map_err(|e| format!("Failed to read header JSON: {}", e))?;

        let header_str = std::str::from_utf8(&json_bytes)
            .map_err(|e| format!("Header is not valid UTF-8: {}", e))?;

        let mut raw_map: HashMap<String, serde_json::Value> = serde_json::from_str(header_str)
            .map_err(|e| format!("Failed to parse header JSON: {}", e))?;

        // Remove __metadata__ if present
        raw_map.remove("__metadata__");

        let mut tensors = HashMap::new();
        for (name, val) in raw_map {
            let meta: TensorMeta = serde_json::from_value(val)
                .map_err(|e| format!("Failed to parse tensor meta for '{}': {}", name, e))?;
            println!("name -- {} , meta -- {} {} {} ",name,meta.data_offsets.to_vec()[0],meta.shape[0],meta.dtype);
            tensors.insert(name, meta);
        }

        let data_start = 8 + header_len;
        
        Ok(Self {
            file,
            header_len,
            tensors,
            data_start,
        })
    }

    pub fn tensor_names(&self) -> Vec<String> {
        let mut names: Vec<String> = self.tensors.keys().cloned().collect();
        names.sort();
        names
    }

    pub fn get_meta(&self, name: &str) -> Option<&TensorMeta> {
        self.tensors.get(name)
    }

    pub fn read_tensor_f32(&mut self, name: &str) -> Result<Vec<f32>, String> {
        let meta = self
            .get_meta(name)
            .ok_or_else(|| format!("Tensor '{}' not found in safetensors", name))?
            .clone();

        let start = meta.data_offsets[0];
        let end = meta.data_offsets[1];
        let byte_len = end - start;

        self.file
            .seek(SeekFrom::Start(self.data_start + start as u64))
            .map_err(|e| format!("Seek failed for '{}': {}", name, e))?;

        let mut raw_bytes = vec![0u8; byte_len];
        self.file
            .read_exact(&mut raw_bytes)
            .map_err(|e| format!("Read failed for '{}': {}", name, e))?;

        match meta.dtype.as_str() {
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
            other => Err(format!("Unsupported dtype '{}' for tensor '{}'", other, name)),
        }
    }

    /// Read a specific row (e.g., token embedding) without loading the entire tensor.
    pub fn read_embedding_row(&mut self, name: &str, token_id: usize, hidden_size: usize) -> Result<Vec<f32>, String> {
        let meta = self
            .get_meta(name)
            .ok_or_else(|| format!("Embedding tensor '{}' not found", name))?
            .clone();

        let bytes_per_elem = match meta.dtype.as_str() {
            "BF16" | "F16" => 2,
            "F32" => 4,
            other => return Err(format!("Unsupported dtype: {}", other)),
        };

        let row_bytes = hidden_size * bytes_per_elem;
        let offset = meta.data_offsets[0] + token_id * row_bytes;

        self.file
            .seek(SeekFrom::Start(self.data_start + offset as u64))
            .map_err(|e| format!("Seek to embedding row {} failed: {}", token_id, e))?;

        let mut raw = vec![0u8; row_bytes];
        self.file
            .read_exact(&mut raw)
            .map_err(|e| format!("Read embedding row {} failed: {}", token_id, e))?;

        let mut row = Vec::with_capacity(hidden_size);
        match meta.dtype.as_str() {
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
