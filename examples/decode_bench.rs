// Microbenchmark: cuBLAS GEMV (OP_T) vs cuBLAS GEMM with M=1 (the previous
// decode fallback) for the exact per-token decode projection shapes. Kernel-only
// timing (output preallocated) so the comparison isolates the projection itself.
use cudarc::cublas::safe::{CudaBlas, Gemm, GemmConfig, Gemv, GemvConfig};
use cudarc::cublas::sys;
use cudarc::driver::CudaContext;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let gpu: usize = std::env::args().nth(1).and_then(|s| s.parse().ok()).unwrap_or(4);
    let iters: usize = std::env::args().nth(2).and_then(|s| s.parse().ok()).unwrap_or(50);
    let ctx = CudaContext::new(gpu)?;
    let stream = ctx.default_stream();
    let blas = CudaBlas::new(stream.clone())?;

    // (in, out, label) for Qwen2.5 0.5B / 1.5B / 3B decode projections.
    let shapes: [(usize, usize, &str); 12] = [
        (896, 896, "0.5B q/o"),
        (896, 128, "0.5B k/v"),
        (896, 4864, "0.5B gate/up"),
        (4864, 896, "0.5B down"),
        (896, 151936, "0.5B logits"),
        (1536, 1536, "1.5B q/o"),
        (1536, 1536, "1.5B k/v"),
        (1536, 8960, "1.5B gate/up"),
        (8960, 1536, "1.5B down"),
        (1536, 151936, "1.5B logits"),
        (2048, 11008, "3B gate/up"),
        (2048, 151936, "3B logits"),
    ];

    println!("{:<12} {:>7} {:>7}  {:>10} {:>10}  {:>8}  {:>8} {:>8}", "shape", "in", "out", "gemv ms", "sgemm1 ms", "faster", "gemv GB/s", "sgemm GB/s");
    for (inf, outf, label) in shapes {
        let x: Vec<f32> = (0..inf).map(|i| ((i % 31) as f32) * 0.01 - 0.1).collect();
        let w: Vec<f32> = (0..outf * inf).map(|i| ((i % 97) as f32) * 0.001 - 0.05).collect();
        let x_dev = stream.clone_htod(&x)?;
        let w_dev = stream.clone_htod(&w)?;
        let mut y_gemv = stream.alloc_zeros::<f32>(outf)?;
        let mut y_gemm = stream.alloc_zeros::<f32>(outf)?;

        let gv = GemvConfig {
            trans: sys::cublasOperation_t::CUBLAS_OP_T,
            m: inf as i32,
            n: outf as i32,
            alpha: 1.0,
            lda: inf as i32,
            incx: 1,
            beta: 0.0,
            incy: 1,
        };
        let gm = GemmConfig {
            transa: sys::cublasOperation_t::CUBLAS_OP_T,
            transb: sys::cublasOperation_t::CUBLAS_OP_N,
            m: outf as i32,
            n: 1,
            k: inf as i32,
            alpha: 1.0,
            lda: inf as i32,
            ldb: inf as i32,
            beta: 0.0,
            ldc: outf as i32,
        };

        // Warmup.
        for _ in 0..5 {
            unsafe { blas.gemv(gv, &w_dev, &x_dev, &mut y_gemv) }.unwrap();
            unsafe { blas.gemm(gm, &w_dev, &x_dev, &mut y_gemm) }.unwrap();
        }
        stream.synchronize()?;

        let t0 = std::time::Instant::now();
        for _ in 0..iters {
            unsafe { blas.gemv(gv, &w_dev, &x_dev, &mut y_gemv) }.unwrap();
        }
        stream.synchronize()?;
        let gemv_ms = t0.elapsed().as_secs_f64() * 1000.0 / iters as f64;

        let t1 = std::time::Instant::now();
        for _ in 0..iters {
            unsafe { blas.gemm(gm, &w_dev, &x_dev, &mut y_gemm) }.unwrap();
        }
        stream.synchronize()?;
        let gemm_ms = t1.elapsed().as_secs_f64() * 1000.0 / iters as f64;

        // Byte traffic: weights + input + output (minimum for a GEMV).
        let bytes = ((outf * inf) + inf + outf) as f64 * 4.0;
        let gv_bw = bytes / (gemv_ms * 1e-3) / 1e9;
        let gm_bw = bytes / (gemm_ms * 1e-3) / 1e9;

        let gv_v = stream.clone_dtoh(&y_gemv)?;
        let gm_v = stream.clone_dtoh(&y_gemm)?;
        let maxdiff = gv_v.iter().zip(gm_v.iter()).map(|(a, b)| (a - b).abs()).fold(0.0f32, f32::max);
        let rel = if gemv_ms > gemm_ms { format!("gemm {:.2}x", gemv_ms / gemm_ms) } else { format!("gemv {:.2}x", gemm_ms / gemv_ms) };
        println!("{:<12} {:>7} {:>7}  {:>10.4} {:>10.4}  {:>8}  {:>8.1} {:>8.1}   maxdiff={maxdiff:.2e}", label, inf, outf, gemv_ms, gemm_ms, rel, gv_bw, gm_bw);
    }
    Ok(())
}
