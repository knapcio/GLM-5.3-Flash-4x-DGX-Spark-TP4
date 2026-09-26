# Weights used by LVKP-S-L2

The target starts from NVIDIA's ModelOpt NVFP4 checkpoint, not the older RedHat
checkpoint. Download this fixed revision on each node (or distribute verified
files once). Keep the base and converted target on the same filesystem: assembly
uses hardlinks, so moving only the overlay shard is insufficient.

```sh
hf download nvidia/GLM-5.3-Flash-NVFP4 --revision 09b04e5e74bca08ca8549fc736d4cdd8624bfde3 --local-dir "$HOME/models/nvidia/GLM-5.3-Flash-NVFP4"
hf download incoai/GLM-5.3-Flash-DFlash2 --revision bf582e4eacc1810f76656d1811693ff6c6737d2a --local-dir "$HOME/models/incoai/GLM-5.3-Flash-DFlash2"
```

These revisions were recovered from the accepted deployment's Hugging Face
config download receipts. Base config SHA256:
`e23c5d98f53e861d49a51bd3c68591621c5482ce829e42c31724152322fba03d`;
original drafter config:
`c4aeac0101196a6e26705b34c45230bcd0c7c68ee2d2d1efdb242087f3712573`.
A config receipt is not a checksum of all weight shards. Record HF/LFS hashes
for the complete downloaded snapshot before conversion.

Use Python with the pinned image's torch/numpy/safetensors packages; no GPU is
needed. Do this offline, not concurrently with serving or loading. The target
converter processes fused groups but retains quantized output bytes until writing;
peak RAM was not independently bounded. The drafter loads its entire state
dictionary. The16GiB container limit is a cap, not a guarantee of completion. Run both inside a
bounded CPU container if host memory is shared with other services.

```sh
# Run from the repository, only on an idle node. Fresh destination directories
# and a unique retained container name are required; no GPU is attached.
docker run --name glm53-convert-lvkp-20260926 --network none --memory 16g --cpus 4 \
  --user "$(id -u):$(id -g)" -e HOME=/tmp \
  -v "$PWD:/recipe:ro" -v "$HOME/models:/models" \
  --entrypoint bash glm53-roce:v11-b58f34ea -c '
    set -e
    bash /recipe/scripts/build_lossless8.sh /models/nvidia/GLM-5.3-Flash-NVFP4 /models/glm-quant-mix
    python3 /recipe/scripts/drafter_fp8.py /models/incoai/GLM-5.3-Flash-DFlash2 /models/incoai/GLM-5.3-Flash-DFlash2-fp8blk
  '
```

The entire models tree uses one bind mount so base and derived hardlinks share a
filesystem. Preserve the stopped converter container and stdout as provenance.
No automatic deletion or retry occurs; inspect partial outputs after failure.

`lossless8` is the historical recipe name, **not a mathematical lossless claim**.
The routed NVFP4 experts remain unchanged. KDA projections use MXFP8; MLA/shared
projections use block128 FP8; selected sensitive/absorbed weights remain BF16.
The conversion preserves the model's released8-bit grids approximately, with
small nonzero re-encoding error. `QMIX_FP8_BLOCK=1` and the included loader filter
are essential: original base shards retain superseded BF16 tensors, while the
index and sentinel ensure the overlay is used. Do not remove the sentinel.

The same incoai drafter is re-encoded on block128 FP8 for decoder projections,
with its fc/conv/norm/codebook portions retained. This is also a lossy conversion;
the target still verifies proposed tokens. The upstream drafter license applies
to downloading, local conversion and distribution; this repository ships the
converter, not derived weights. Consult the pinned model card before redistribution.

Accepted derived config hashes are target
`f14dc13ce3bef88a5539c9e61b3e4f3dbef6958ebc3acda4b7f05f418642c3b0`
and drafter
`15bc842939ff7ca3ccc5aad36d25fff854da05a60086a4f55672d804fc6dbe7b`.
Compare generated configs; `assemble` invokes validation with its zero-copy
index/sentinel context. Do not run standalone `verify` on this hardlinked layout:
it would misinterpret retained superseded base tensors as duplicates. Full numeric
quality validation is still required for a rebuilt image/checkpoint; matching
config hashes alone does not validate tensor values. The portable scripts have
CPU contract tests; conversion of all weights was not rerun for publication.
