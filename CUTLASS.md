# CUTLASS dependency

The prefill attention kernel uses the fused multi-head attention example from
NVIDIA CUTLASS. CUTLASS is a Git submodule pinned by this repository to
v3.9.2 (`ad7b2f5e84fcfa124cb02b91d5bd26d238c0459e`). The build compiles the
kernel to SM70 PTX with NVCC. A small build-time overlay preserves the GQA
behavior needed by this wrapper without modifying the upstream submodule.

## Clone the repository

With modern Git, initialize CUTLASS while cloning:

```bash
git clone --recurse-submodules https://github.com/namanjain2005/InferenceEngine.git
```

For older Git versions, `--recursive` is the equivalent option:

```bash
git clone --recursive https://github.com/namanjain2005/InferenceEngine.git
```

If the repository is already cloned, or the submodule directory is empty, run
this from the repository root:

```bash
git submodule update --init --recursive
```

## Pull project changes

Pull the project commit, then check out the CUTLASS commit recorded by it:

```bash
git pull
git submodule update --init --recursive
```

The submodule is pinned; pulling the project does not float CUTLASS to its
latest upstream version. Verify the checked-out commit with:

```bash
git submodule status
```

## Intentionally update CUTLASS

Only update the submodule when changing the dependency version deliberately.
Fetch and check out the desired upstream tag, then commit the new submodule
pointer together with any compatibility changes:

```bash
cd vendor/cutlass
git fetch --tags
git checkout v3.9.2
cd ../..
git add vendor/cutlass
git commit -m "Update pinned CUTLASS version"
```

Replace `v3.9.2` with the reviewed target tag when upgrading. After pulling a
commit that updates the pointer, run `git submodule update --init --recursive`
to check out that exact version.
