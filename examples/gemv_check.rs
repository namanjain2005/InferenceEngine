// Standalone cublasSgemv check with the exact projection shapes used during
// decode. Config mirrors CublasLinear::gemv: weights row-major [out, in]
// passed as column-major [in, out] with OP_T, m=in, n=out, lda=in.
use cudarc::cublas::safe::{CudaBlas, Gemv, GemvConfig};
use cudarc::cublas::sys;
use cudarc::driver::CudaContext;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let gpu: usize = std::env::args().nth(1).and_then(|s| s.parse().ok()).unwrap_or(4);
    let ctx = CudaContext::new(gpu)?;
    let stream = ctx.default_stream();
    let blas = CudaBlas::new(stream.clone())?;

    // (in_features, out_features) hit by Qwen2.5 decode: q, k, v, o, gate, up,
    // down, and the final vocab projection (K != N and K == N cases).
    let shapes: [(usize, usize); 7] = [
        (896, 896),
        (896, 128),
        (896, 4864),
        (4864, 896),
        (896, 151936),
        (1536, 1536),
        (2048, 2048),
    ];
    for (inf, outf) in shapes {
        let x: Vec<f32> = (0..inf).map(|i| (i as f32 * 0.001) % 1.0).collect();
        let w: Vec<f32> = (0..outf * inf).map(|i| ((i % 97) as f32) * 0.01).collect();
        let x_dev = stream.clone_htod(&x)?;
        let w_dev = stream.clone_htod(&w)?;
        let mut y_dev = stream.alloc_zeros::<f32>(outf)?;
        let cfg = GemvConfig {
            trans: sys::cublasOperation_t::CUBLAS_OP_T,
            m: inf as i32,
            n: outf as i32,
            alpha: 1.0,
            lda: inf as i32,
            incx: 1,
            beta: 0.0,
            incy: 1,
        };
        unsafe { blas.gemv(cfg, &w_dev, &x_dev, &mut y_dev) }
            .map_err(|e| format!("sgemv {inf}->{outf}: {e:?}"))?;
        stream.synchronize()?;
        let y = stream.clone_dtoh(&y_dev)?;
        // Reference for the first two outputs on the CPU.
        for n in 0..2 {
            let expect: f32 = (0..inf).map(|k| w[n * inf + k] * x[k]).sum();
            assert!(
                (expect - y[n]).abs() <= 1e-3 * expect.abs().max(1.0),
                "sgemv {inf}->{outf} row {n}: got {} expected {expect}",
                y[n]
            );
        }
        let finite = y.iter().all(|v| v.is_finite());
        println!("sgemv {inf:>6} -> {outf:<6} ok={finite} y[0]={:.4}", y[0]);
    }
    println!("ALL SGEMMV SHAPES OK");
    Ok(())
}
