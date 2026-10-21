use std::env;
use std::fs;
use std::path::{Path, PathBuf};
use std::process::Command;

fn track_tree(path: &Path) {
    println!("cargo:rerun-if-changed={}", path.display());
    if let Ok(entries) = fs::read_dir(path) {
        for entry in entries.flatten() {
            let child = entry.path();
            if child.is_dir() {
                track_tree(&child);
            } else {
                println!("cargo:rerun-if-changed={}", child.display());
            }
        }
    }
}

fn main() {
    let root = PathBuf::from(env::var_os("CARGO_MANIFEST_DIR").expect("manifest dir"));
    let out = PathBuf::from(env::var_os("OUT_DIR").expect("OUT_DIR"));
    let cuda = root.join("src/kernels/layer/cutlass_attention.cu");
    let cutlass = root.join("vendor/cutlass");
    let cutlass_include = cutlass.join("include");
    let fmha_example = cutlass.join("examples/41_fused_multi_head_attention");
    if !cutlass_include.join("cutlass/numeric_types.h").is_file()
        || !fmha_example.join("kernel_forward.h").is_file()
    {
        panic!(
            "CUTLASS submodule is missing; run `git submodule update --init --recursive` (see CUTLASS.md)"
        );
    }
    let ptx = out.join("cutlass_attention.ptx");

    println!("cargo:rerun-if-changed={}", cuda.display());
    track_tree(&cutlass_include);
    track_tree(&fmha_example);

    // CUTLASS's optional single-query MQA shortcut assumes every query head
    // shares a KV head. This wrapper supplies a GQA batch hook, so compile a
    // tiny overlay that keeps the normal per-query-head path in that case.
    let upstream_forward = fmha_example.join("kernel_forward.h");
    let upstream_source = fs::read_to_string(&upstream_forward)
        .unwrap_or_else(|error| panic!("reading {} failed: {error}", upstream_forward.display()));
    let original = "if (num_queries == 1 && k_strideH == 0 && v_strideH == 0) {";
    let patched = "if (!kToBatchHook && num_queries == 1 && k_strideH == 0 && v_strideH == 0) {";
    if upstream_source.matches(original).count() != 1 {
        panic!("CUTLASS kernel_forward.h no longer matches the GQA patch expected by this wrapper");
    }
    let overlay = out.join("cutlass-overlay");
    fs::create_dir_all(&overlay).expect("create CUTLASS overlay directory");
    fs::write(
        overlay.join("kernel_forward.h"),
        upstream_source.replacen(original, patched, 1),
    )
    .expect("write patched CUTLASS kernel_forward.h");

    let nvcc = env::var("NVCC").unwrap_or_else(|_| "nvcc".to_owned());
    let status = Command::new(&nvcc)
        .arg("-ptx")
        .arg("-arch=compute_70")
        .arg("-std=c++17")
        .arg("-O3")
        .arg("-lineinfo")
        .arg("-Xcompiler=-fPIC")
        .arg(format!("-I{}", cutlass_include.display()))
        .arg(format!("-I{}", overlay.display()))
        .arg(format!("-I{}", fmha_example.display()))
        .arg(&cuda)
        .arg("-o")
        .arg(&ptx)
        .status()
        .unwrap_or_else(|error| panic!("failed to invoke {nvcc}: {error}"));

    if !status.success() {
        panic!("CUTLASS attention PTX compilation failed with {status}");
    }
    println!("cargo:rustc-env=CUTLASS_ATTENTION_PTX={}", ptx.display());
}
