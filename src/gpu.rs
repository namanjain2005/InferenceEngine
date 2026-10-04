use std::process::Command;
use std::sync::Arc;
use cudarc::driver::safe::{CudaContext, CudaStream};

#[derive(Debug, Clone)]
pub struct GpuDevice {
    pub index: usize,
    pub name: String,
    pub total_mb: usize,
    pub free_mb: usize,
    pub used_mb: usize,
    pub util_pct: usize,
}

pub fn query_nvidia_smi() -> Result<Vec<GpuDevice>, String> {
    let output = Command::new("nvidia-smi")
        .args([
            "--query-gpu=index,name,memory.total,memory.free,memory.used,utilization.gpu",
            "--format=csv,noheader,nounits",
        ])
        .output()
        .map_err(|e| format!("Failed to execute nvidia-smi: {}", e))?;

    if !output.status.success() {
        return Err(format!(
            "nvidia-smi exited with status {}: {}",
            output.status,
            String::from_utf8_lossy(&output.stderr)
        ));
    }

    let text = String::from_utf8_lossy(&output.stdout);
    let mut gpus = Vec::new();

    for line in text.lines() {
        let parts: Vec<&str> = line.split(',').map(|s| s.trim()).collect();
        if parts.len() >= 6 {
            let index = parts[0].parse::<usize>().unwrap_or(0);
            let name = parts[1].to_string();
            let total_mb = parts[2].parse::<usize>().unwrap_or(0);
            let free_mb = parts[3].parse::<usize>().unwrap_or(0);
            let used_mb = parts[4].parse::<usize>().unwrap_or(0);
            let util_pct = parts[5].parse::<usize>().unwrap_or(0);

            gpus.push(GpuDevice {
                index,
                name,
                total_mb,
                free_mb,
                used_mb,
                util_pct,
            });
        }
    }

    Ok(gpus)
}

pub fn print_gpu_table(gpus: &[GpuDevice], selected_idx: usize) {
    println!("\n+-----+-----------------------+---------------------+------------+----------+");
    println!("| GPU | Name                  | Memory (Free/Total) | Memory %   | Selected |");
    println!("+-----+-----------------------+---------------------+------------+----------+");
    for g in gpus {
        let sel = if g.index == selected_idx { "  --> YES" } else { "         " };
        let pct = if g.total_mb > 0 {
            (g.used_mb as f64 / g.total_mb as f64) * 100.0
        } else {
            0.0
        };
        println!(
            "| {:3} | {:21} | {:5} MiB / {:5} MiB | {:5.1}% used|{}|",
            g.index,
            if g.name.len() > 21 { &g.name[..21] } else { &g.name },
            g.free_mb,
            g.total_mb,
            pct,
            sel
        );
    }
    println!("+-----+-----------------------+---------------------+------------+----------+\n");
}

pub fn select_gpu(requested: Option<usize>, gpus: &[GpuDevice]) -> Result<usize, String> {
    if gpus.is_empty() {
        return Err("No NVIDIA GPUs detected via nvidia-smi.".to_string());
    }

    if let Some(req) = requested {
        if let Some(gpu) = gpus.iter().find(|g| g.index == req) {
            println!("Requested GPU {}: {} ({:.1} GB free)", req, gpu.name, gpu.free_mb as f64 / 1024.0);
            return Ok(req);
        } else {
            return Err(format!("Requested GPU {} is not in range 0..{}", req, gpus.len() - 1));
        }
    }

    // Auto-select GPU with highest free memory
    let best = gpus.iter().max_by_key(|g| g.free_mb).unwrap();
    println!("Auto-selected GPU {}: {} with highest free memory ({:.1} GB free)", best.index, best.name, best.free_mb as f64 / 1024.0);
    Ok(best.index)
}

pub fn init_cuda_device(gpu_idx: usize) -> Result<(Arc<CudaContext>, Arc<CudaStream>), String> {
    println!("Initializing CUDA Context on GPU {}...", gpu_idx);
    let ctx = CudaContext::new(gpu_idx).map_err(|e| format!("CUDA context creation failed on GPU {}: {:?}", gpu_idx, e))?;
    let stream = ctx.default_stream();
    println!("CUDA Context and default stream ready on GPU {}.", gpu_idx);
    Ok((ctx, stream))
}
