use std::env;
use std::path::PathBuf;
use std::process::Command;

fn main() {
    let root = PathBuf::from(env::var_os("CARGO_MANIFEST_DIR").expect("manifest dir"));
    let out = PathBuf::from(env::var_os("OUT_DIR").expect("OUT_DIR"));
    let cuda = root.join("src/kernels/layer/cutlass_attention.cu");
    let cutlass = root.join("vendor/cutlass");
    let ptx = out.join("cutlass_attention.ptx");

    println!("cargo:rerun-if-changed={}", cuda.display());
    println!(
        "cargo:rerun-if-changed={}",
        cutlass.join("UPSTREAM.md").display()
    );
    println!(
        "cargo:rerun-if-changed={}",
        cutlass
            .join("examples/41_fused_multi_head_attention")
            .display()
    );
    println!(
        "cargo:rerun-if-changed={}",
        cutlass.join("tools/util/include").display()
    );
    println!(
        "cargo:rerun-if-changed={}",
        cutlass.join("include/cutlass").display()
    );

    let nvcc = env::var("NVCC").unwrap_or_else(|_| "nvcc".to_owned());
    let status = Command::new(&nvcc)
        .arg("-ptx")
        .arg("-arch=compute_70")
        .arg("-std=c++17")
        .arg("-O3")
        .arg("-lineinfo")
        .arg("-Xcompiler=-fPIC")
        .arg(format!("-I{}", cutlass.join("include").display()))
        .arg(format!(
            "-I{}",
            cutlass
                .join("examples/41_fused_multi_head_attention")
                .display()
        ))
        .arg(format!(
            "-I{}",
            cutlass.join("tools/util/include").display()
        ))
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
