"""glm-quant-mix runtime patch (imported by overlay/sitecustomize.py when QMIX_FP8_BLOCK=1).

Adds a per-layer quant_algo "FP8_BLOCK" to vLLM's ModelOpt MIXED_PRECISION config: DeepSeek-style block-128 FP8
(`weight` float8_e4m3fn + `weight_scale_inv` float32 [N/128, K/128]) served by vLLM's own Fp8LinearMethod, the path
the official FP8 GLM checkpoint already runs on GB10. With VLLM_TEST_FORCE_FP8_MARLIN=1 (and --linear-backend marlin)
that is Marlin W8A16: weight-only, BF16 activations. ModelOpt's own FP8 entries are per-tensor W8A8 with a calibrated
static activation scale, which this pack does not have.

Also carries the `_try_load_fp8_attn_proj` guard (same as patches/model.py) in case the mounted model.py lacks it,
and, only with QMIX_DRAFT_HEAD_FP8=1, a draft-only FP8 copy of the borrowed lm_head (consult lever 4).
Same content as patches/modelopt.py, applied at import time instead of by a file mount.
"""
import importlib.abc
import importlib.util
import sys

_TARGETS = {
    "vllm.model_executor.layers.quantization.modelopt": "_patch_modelopt",
    "vllm.models.glm5next.nvidia.model": "_patch_glm_model",
    "vllm.v1.spec_decode.llm_base_proposer": "_patch_proposer",
}


def _patch_modelopt(mod):
    import torch  # noqa: F401
    from vllm.model_executor.layers.linear import LinearBase
    from vllm.model_executor.layers.vocab_parallel_embedding import ParallelLMHead

    C = mod.ModelOptMixedPrecisionConfig
    if getattr(C, "_qmix_patched", False):
        return
    orig_from = C._from_config.__func__
    orig_gqm = C.get_quant_method

    def _from_config(cls, **kw):
        obj = orig_from(cls, **kw)
        from vllm.model_executor.layers.quantization.fp8 import Fp8Config

        obj.fp8_block_config = Fp8Config(
            is_checkpoint_fp8_serialized=True, activation_scheme="dynamic", weight_block_size=[128, 128]
        )
        return obj

    def get_quant_method(self, layer, prefix):
        if isinstance(layer, (LinearBase, ParallelLMHead)) and not self.is_layer_excluded(prefix):
            if self._resolve_quant_algo(prefix) == "FP8_BLOCK":
                from vllm.model_executor.layers.quantization.fp8 import Fp8LinearMethod

                return Fp8LinearMethod(self.fp8_block_config)
        return orig_gqm(self, layer, prefix)

    C._from_config = classmethod(_from_config)
    C.get_quant_method = get_quant_method
    C._qmix_patched = True
    sys.stderr.write("qmix: ModelOpt MIXED_PRECISION accepts FP8_BLOCK\n")


def _patch_glm_model(mod):
    import torch

    orig = mod._try_load_fp8_attn_proj
    if getattr(orig, "_qmix", False):
        return

    def guarded(name, tensor, buf, params_dict, loaded_params, kv_a_pad_size):
        for suffix, (_key, target_base, _sid, _kva) in mod._FP8_ATTN_PROJS.items():
            if suffix in name:
                tw = f"{name.rsplit(suffix, 1)[0]}.{target_base}.weight"
                p = params_dict.get(tw)
                if p is not None and p.dtype not in (torch.bfloat16, torch.float16, torch.float32):
                    return False
                break
        return orig(name, tensor, buf, params_dict, loaded_params, kv_a_pad_size)

    guarded._qmix = True
    mod._try_load_fp8_attn_proj = guarded

    # Zero-copy arms (tools/glm_quant_mix.py assemble without --common): the arm directory hardlinks the untouched
    # nvidia shards, which still carry BF16 copies of the re-encoded weights. Drop exactly those copies (listed in
    # qmix-manifest.json, recognised by BF16 dtype) and the sentinel tensor; refuse to finish if the sentinel was
    # not seen, so a wrong directory cannot load half-filtered.
    C = mod.Glm5NextForConditionalGeneration
    if getattr(C, "_qmix_filter", False):
        return
    orig_init, orig_lw = C.__init__, C.load_weights

    import functools

    # functools.wraps keeps the original signature visible to inspect.signature(): vLLM's model loader checks
    # for a `vllm_config` parameter and otherwise calls the legacy positional constructor (boot failures
    # 2026-09-25 23:53 and 2026-09-26 01:32: "__init__() missing 1 required keyword-only argument: 'vllm_config'").
    @functools.wraps(orig_init)
    def __init__(self, *args, **kw):
        orig_init(self, *args, **kw)
        vc = kw.get("vllm_config")
        object.__setattr__(self, "_qmix_model_path", getattr(getattr(vc, "model_config", None), "model", None))

    def _layer_report(model):
        """QMIX_DEBUG_LAYERS=1: per re-encodable module family, count (quant method, weight dtype) and the bytes
        this rank holds, right after the checkpoint is loaded (before Marlin repack, which keeps the dtype)."""
        import collections
        import os
        import re

        if os.environ.get("QMIX_DEBUG_LAYERS") != "1":
            return
        lt = getattr(getattr(model, "config", None), "text_config", None) or getattr(model, "config", None)
        types = getattr(lt, "layer_types", None) or []
        fam = collections.defaultdict(lambda: collections.Counter())
        nbytes = collections.defaultdict(float)
        for name, mod in model.named_modules():
            w = getattr(mod, "weight", None)
            qm = getattr(mod, "quant_method", None)
            if w is None or qm is None or not isinstance(w, torch.Tensor):
                continue
            m = re.search(r"layers\.(\d+)\.(self_attn|mlp)\.(.*)$", name)
            if m:
                L, leaf = int(m.group(1)), m.group(3)
                if leaf.startswith("experts") or leaf == "gate" or leaf.startswith("indexer"):
                    continue
                kind = ("kda." if L < len(types) and types[L] == "linear_attention" else "mla.") + leaf \
                    if m.group(2) == "self_attn" else "mlp." + leaf
            elif name.endswith("lm_head"):
                kind = "draft.lm_head" if "draft" in name else "lm_head"
            else:
                continue
            key = (kind, type(qm).__name__, str(w.dtype).replace("torch.", ""))
            fam[kind][key[1:]] += 1
            nbytes[key] += w.numel() * w.element_size()
            for sname in ("weight_scale", "weight_scale_inv", "weight_scale_2"):
                sc = getattr(mod, sname, None)
                if isinstance(sc, torch.Tensor):
                    nbytes[key] += sc.numel() * sc.element_size()
        tot = collections.Counter()
        lines = []
        for kind in sorted(fam):
            for (meth, dt), n in sorted(fam[kind].items()):
                b = nbytes[(kind, meth, dt)]
                tot["bf16" if dt == "bfloat16" else "low"] += b
                lines.append(f"  {kind:34s} {meth:36s} {dt:14s} x{n:3d} {b / 2**20:8.1f} MiB")
        sys.stderr.write("qmix: layer report (this rank)\n" + "\n".join(lines) +
                         f"\nqmix: non-expert BF16 {tot['bf16'] / 2**30:.2f} GiB, low-precision {tot['low'] / 2**30:.2f} GiB\n")

    @functools.wraps(orig_lw)
    def load_weights(self, weights):
        import json
        import os

        path = getattr(self, "_qmix_model_path", None)
        mf = os.path.join(path, "qmix-manifest.json") if path else None
        man = json.load(open(mf)) if mf and os.path.exists(mf) else {}
        if not man.get("zero_copy"):
            out = orig_lw(self, weights)
            _layer_report(self)
            return out
        qn, sentinel = set(man["quantized_names"]), man["sentinel"]
        st = {"sentinel": 0, "dropped": 0}

        def gen():
            for item in weights:
                name, t = item[0], item[1]
                if name == sentinel:
                    st["sentinel"] += 1
                    continue
                if name in qn and t.dtype == torch.bfloat16:
                    st["dropped"] += 1
                    continue
                yield item

        out = orig_lw(self, gen())
        sys.stderr.write(f"qmix: zero-copy arm {man.get('arm')}: dropped {st['dropped']} BF16 duplicates "
                         f"of {len(qn)} re-encoded weights, sentinel {st['sentinel']}\n")
        if st["sentinel"] < 1:
            raise RuntimeError("qmix: sentinel tensor not seen; refusing a possibly half-filtered load")
        if st["dropped"] != len(qn):
            sys.stderr.write("qmix: WARNING dropped count differs from the manifest\n")
        _layer_report(self)
        return out

    C.__init__, C.load_weights, C._qmix_filter = __init__, load_weights, True


def _patch_proposer(mod):
    """QMIX_DRAFT_HEAD_FP8=1: after the drafter borrows the target lm_head, give the drafter its own FP8 copy
    (per-row scale, Marlin W8A16). Only draft candidates change; the target head that verifies stays BF16, so
    outputs are unchanged by construction and only acceptance can move. Fails safe: any error keeps the shared
    BF16 head."""
    import copy
    import os

    import torch

    if os.environ.get("QMIX_DRAFT_HEAD_FP8") != "1":
        return
    B = mod.SpecDecodeBaseProposer
    if getattr(B, "_qmix_patched", False):
        return
    orig = B._maybe_share_lm_head

    def share(self, target_language_model):
        orig(self, target_language_model)
        try:
            head = getattr(self.model, "lm_head", None)
            if head is None or head is not getattr(target_language_model, "lm_head", None):
                return
            from vllm.model_executor.layers.quantization.utils.marlin_utils_fp8 import (
                apply_fp8_marlin_linear, prepare_fp8_layer_for_marlin)

            w = head.weight.data
            n, k = w.shape
            scale = (w.float().abs().amax(dim=1).clamp(min=1e-12) / 448.0)
            q = (w.float() / scale[:, None]).clamp(-448.0, 448.0).to(torch.float8_e4m3fn)
            h = torch.nn.Module()
            h.weight = torch.nn.Parameter(q, requires_grad=False)
            h.weight_scale = torch.nn.Parameter(scale.to(torch.float32), requires_grad=False)
            h.output_size_per_partition, h.input_size_per_partition = n, k
            h.orig_dtype, h.logical_widths = w.dtype, [n]
            prepare_fp8_layer_for_marlin(h, size_k_first=False)

            class _M:
                def apply(self_, layer, x, bias=None):
                    return apply_fp8_marlin_linear(input=x, weight=h.weight, weight_scale=h.weight_scale,
                                                   workspace=h.workspace, size_n=n, size_k=k, bias=bias)

            dh = copy.copy(head)            # same shard metadata, same (shared) BF16 weight object
            object.__setattr__(dh, "quant_method", _M())
            object.__setattr__(dh, "_qmix_fp8", h)   # keep the repacked tensors alive; plain attribute, so the
                                                     # shallow copy's shared _modules dict (= target head's) is untouched
            self.model.lm_head = dh
            sys.stderr.write(f"qmix: draft lm_head -> FP8 Marlin copy [{n}x{k}]\n")
        except Exception as exc:  # noqa: BLE001
            sys.stderr.write(f"qmix: draft head fp8 skipped: {exc!r}\n")

    B._maybe_share_lm_head = share
    B._qmix_patched = True


class _Hook(importlib.abc.MetaPathFinder):
    def find_spec(self, name, path, target=None):
        if name not in _TARGETS:
            return None
        sys.meta_path.remove(self)
        try:
            spec = importlib.util.find_spec(name)
        finally:
            sys.meta_path.insert(0, self)
        if spec is None or spec.loader is None:
            return None
        exec_module = spec.loader.exec_module
        fn = globals()[_TARGETS[name]]

        def patched(module):
            exec_module(module)
            try:
                fn(module)
            except Exception as exc:  # noqa: BLE001
                sys.stderr.write(f"qmix: patch {name} failed: {exc!r}\n")
                raise

        spec.loader.exec_module = patched
        return spec


def register():
    if not any(isinstance(h, _Hook) for h in sys.meta_path):
        sys.meta_path.insert(0, _Hook())
