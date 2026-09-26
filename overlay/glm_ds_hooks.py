# SPDX-License-Identifier: Apache-2.0
"""Import hooks and drift guard for the DS4.1 -> GLM transfer adapters (diagnostics/glm-ds-transfer).

  glm_ds_split    GLM_DS_SPLIT=fc:128[,qkv_a:64,wq_b:16]
                  replicated BF16 linears split by output columns over TP + all-gather
                  (port of our DS4.1 DSV41_REPLICATED_SPLIT)
  glm_ds_draft    GLM_DS_DRAFT_HEAD_FP8=1       draft-only fp8 copy of the draft's LM head, DFlash2 (shared target
                                                head) or DSpark (own head) (DS DSV41_DRAFT_HEAD_FP8)
                  GLM_DS_TOPK_ONE_GATHER=1      DFlash2 vocab-parallel top-k: one packed all-gather instead of two
                  GLM_DS_DRAFT_TAU=0.7          proposal temperature for probabilistic DFlash2 drafts (DS DSV41_DRAFT_TAU)
  glm_ds_cpu_pin  GLM_DS_CPU_PIN=1              pin engine core / TP worker / API server to CPU sets (X925 cores)

register() is called from overlay/sitecustomize.py (one marked block, see sitecustomize.snippet) in
every Python process of the container. With every GLM_DS_* variable unset or 0 it registers nothing
and imports nothing. Otherwise each adapter's install(module) runs right after the engine module it
patches has executed. A failing install raises, so the boot fails loudly instead of serving a
half-patched engine. Before patching, the text of every engine function an adapter replaces or
relies on is hashed and compared with the table below (taken from the Tony v11 image,
ghcr.io/tonyd2wild/vllm-glm53-flash@sha256:4def0ef6..., vLLM 0.1.dev20051+g487ecf187); a mismatch
refuses the adapter (GLM_DS_ALLOW_DRIFT=1 overrides, for a qualified newer image only).

Self-contained: no dependency on the separate prefill hooks.
"""
from __future__ import annotations

import ast
import hashlib
import importlib.abc
import importlib.util
import os
import sys
import textwrap

# function text sha256[:16] in the qualified image
EXPECTED = {
    "vllm.model_executor.model_loader.base_loader:BaseModelLoader.load_model": "9c24cbd9e097ad35",
    "vllm.model_executor.layers.linear:ReplicatedLinear.forward": "0509484a019dc9a6",
    "vllm.model_executor.layers.linear:ColumnParallelLinear.forward": "a83e0a998236fbef",
    "vllm.model_executor.layers.linear:UnquantizedLinearMethod.apply": "218ee489ba7f57bd",
    "vllm.model_executor.models.deepseek_v2:DeepSeekV2FusedQkvAProjLinear.forward": "798a7005cd37ff4a",
    "vllm.model_executor.layers.mla:MultiHeadLatentAttentionWrapper.forward": "4a3b9e6f74c94a46",
    "vllm.model_executor.models.qwen3_dflash:DFlashQwen3ForCausalLM.combine_hidden_states": "8bbefec5535cc3b1",
    "vllm.model_executor.layers.logits_processor:LogitsProcessor.get_top_k_tokens": "d5c541467fd52166",
    "vllm.model_executor.layers.logits_processor:LogitsProcessor._apply_head": "a3d5c2eec69e2448",
    "vllm.model_executor.models.qwen3_dflash2:DFlash2Qwen3ForCausalLM.compute_candidates": "c3ab78b7f9645cd0",
    "vllm.v1.worker.gpu.spec_decode.dflash2.speculator:DFlash2Speculator._sample_path": "d67af3e6de394fa9",
    "vllm.v1.worker.gpu.spec_decode.dflash2.speculator:DFlash2Speculator._cache_draft_logits": "975f688d4367d37a",
    "vllm.v1.worker.gpu.spec_decode.dflash.speculator:DFlashSpeculator.load_draft_model": "f80a17509f09ccb8",
    "vllm.utils.system_utils:set_process_title": "b064f569374188ce",
    "vllm.model_executor.models.qwen3_dspark:Qwen3DSparkForCausalLM.compute_draft_logits": "ab6c3182b52c19f4",
    "vllm.v1.worker.gpu.spec_decode.dspark.speculator:DSparkSpeculator.load_draft_model": "8f8132dbd56d6d6c",
    "vllm.model_executor.layers.logits_processor:LogitsProcessor._get_logits": "2cb2548df232bbb1",
    "vllm.model_executor.layers.logits_processor:LogitsProcessor.forward": "997b8331aec63267",
}

# which adapter depends on which engine functions
NEEDS = {
    "split": [
        "vllm.model_executor.model_loader.base_loader:BaseModelLoader.load_model",
        "vllm.model_executor.layers.linear:ReplicatedLinear.forward",
        "vllm.model_executor.layers.linear:ColumnParallelLinear.forward",
        "vllm.model_executor.layers.linear:UnquantizedLinearMethod.apply",
        "vllm.model_executor.models.deepseek_v2:DeepSeekV2FusedQkvAProjLinear.forward",
        "vllm.model_executor.layers.mla:MultiHeadLatentAttentionWrapper.forward",
        "vllm.model_executor.models.qwen3_dflash:DFlashQwen3ForCausalLM.combine_hidden_states",
    ],
    "draft": [
        "vllm.model_executor.layers.logits_processor:LogitsProcessor.get_top_k_tokens",
        "vllm.model_executor.layers.logits_processor:LogitsProcessor._apply_head",
        "vllm.model_executor.models.qwen3_dflash2:DFlash2Qwen3ForCausalLM.compute_candidates",
        "vllm.v1.worker.gpu.spec_decode.dflash2.speculator:DFlash2Speculator._sample_path",
        "vllm.v1.worker.gpu.spec_decode.dflash2.speculator:DFlash2Speculator._cache_draft_logits",
        "vllm.v1.worker.gpu.spec_decode.dflash.speculator:DFlashSpeculator.load_draft_model",
        "vllm.model_executor.models.qwen3_dspark:Qwen3DSparkForCausalLM.compute_draft_logits",
        "vllm.v1.worker.gpu.spec_decode.dspark.speculator:DSparkSpeculator.load_draft_model",
        "vllm.model_executor.layers.logits_processor:LogitsProcessor._get_logits",
        "vllm.model_executor.layers.logits_processor:LogitsProcessor.forward",
    ],
    "cpu_pin": ["vllm.utils.system_utils:set_process_title"],
}


def _on(env, name) -> bool:
    return env.get(name, "0").strip().lower() not in ("", "0", "off", "false", "no", "1.0")


def wanted(env=None) -> dict:
    env = os.environ if env is None else env
    return {
        "split": _on(env, "GLM_DS_SPLIT"),
        "draft": any(_on(env, v) for v in ("GLM_DS_DRAFT_HEAD_FP8", "GLM_DS_TOPK_ONE_GATHER", "GLM_DS_DRAFT_TAU")),
        "cpu_pin": _on(env, "GLM_DS_CPU_PIN"),
    }


# ------------------------------------------------------------------------------------------
# drift guard (same scheme as glm_prefill_hooks: ast-extracted function text, sha256[:16])
# ------------------------------------------------------------------------------------------
def func_source(path: str, qualname: str) -> str:
    with open(path, encoding="utf-8") as f:
        src = f.read()
    node = ast.parse(src)
    for part in qualname.split("."):
        for child in ast.iter_child_nodes(node):
            if isinstance(child, (ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)) and child.name == part:
                node = child
                break
        else:
            raise KeyError(f"{qualname} not found in {path}")
    lines = src.splitlines(keepends=True)[node.lineno - 1:node.end_lineno]
    return textwrap.dedent("".join(lines))


def src_hash(path: str, qualname: str) -> str:
    return hashlib.sha256(func_source(path, qualname).encode()).hexdigest()[:16]


def hashes_at(root: str, keys) -> dict:
    out = {}
    for key in keys:
        modname, qual = key.split(":", 1)
        try:
            out[key] = src_hash(os.path.join(root, *modname.split(".")) + ".py", qual)
        except (KeyError, OSError) as exc:
            out[key] = f"missing ({type(exc).__name__})"
    return out


def vllm_root() -> str:
    spec = importlib.util.find_spec("vllm")
    return os.path.dirname(os.path.dirname(spec.origin))


def check_sources(adapter: str, root: str | None = None) -> None:
    keys = NEEDS[adapter]
    got = hashes_at(root or vllm_root(), keys)
    bad = {k: (EXPECTED[k], got[k]) for k in keys if got[k] != EXPECTED[k]}
    if bad and os.environ.get("GLM_DS_ALLOW_DRIFT", "0") != "1":
        lines = "\n".join(f"  {k}: expected {e}, image {g}" for k, (e, g) in sorted(bad.items()))
        raise RuntimeError(f"glm-ds {adapter}: engine sources differ from the qualified image; refusing to "
                           f"patch (unset the GLM_DS_* flag to run stock, or GLM_DS_ALLOW_DRIFT=1):\n{lines}")


# ------------------------------------------------------------------------------------------
# after-import hooks
# ------------------------------------------------------------------------------------------
class _AfterImport(importlib.abc.MetaPathFinder):
    def __init__(self):
        self.hooks: dict[str, list] = {}
        self._busy: set[str] = set()

    def add(self, modname, fn):
        mod = sys.modules.get(modname)
        if mod is not None:  # already imported: patch now
            fn(mod)
            return
        self.hooks.setdefault(modname, []).append(fn)

    def find_spec(self, name, path, target=None):
        if name not in self.hooks or name in self._busy:
            return None
        self._busy.add(name)
        try:
            spec = importlib.util.find_spec(name)
        finally:
            self._busy.discard(name)
        if spec is None or spec.loader is None:
            return None
        callbacks = self.hooks.pop(name)
        exec_module = spec.loader.exec_module

        def patched_exec(module, _exec=exec_module, _cbs=callbacks):
            _exec(module)
            for cb in _cbs:
                cb(module)

        spec.loader.exec_module = patched_exec
        return spec


_FINDER = None


def after_import(modname: str, fn) -> None:
    global _FINDER
    if _FINDER is None:
        _FINDER = _AfterImport()
        sys.meta_path.insert(0, _FINDER)
    _FINDER.add(modname, fn)


def register(env=None) -> dict:
    """Entry point from sitecustomize. Returns what was registered (for tests and logs)."""
    w = wanted(env)
    if w["split"]:
        check_sources("split")

        def _split(mod):
            import glm_ds_split
            glm_ds_split.install_loader(mod)
        after_import("vllm.model_executor.model_loader.base_loader", _split)
    if w["draft"]:
        check_sources("draft")

        def _lp(mod):
            import glm_ds_draft
            glm_ds_draft.install_logits_processor(mod)

        def _model(mod):
            import glm_ds_draft
            glm_ds_draft.install_dflash2_model(mod)

        def _spec2(mod):
            import glm_ds_draft
            glm_ds_draft.install_dflash2_speculator(mod)

        def _spec(mod):
            import glm_ds_draft
            glm_ds_draft.install_dflash_speculator(mod)
        after_import("vllm.model_executor.layers.logits_processor", _lp)
        after_import("vllm.model_executor.models.qwen3_dflash2", _model)
        after_import("vllm.v1.worker.gpu.spec_decode.dflash2.speculator", _spec2)
        after_import("vllm.v1.worker.gpu.spec_decode.dflash.speculator", _spec)

        def _dsm(mod):
            import glm_ds_draft
            glm_ds_draft.install_dspark_model(mod)

        def _dss(mod):
            import glm_ds_draft
            glm_ds_draft.install_dspark_speculator(mod)
        after_import("vllm.model_executor.models.qwen3_dspark", _dsm)
        after_import("vllm.v1.worker.gpu.spec_decode.dspark.speculator", _dss)
    if w["cpu_pin"]:
        check_sources("cpu_pin")
        import glm_ds_cpu_pin
        glm_ds_cpu_pin.pin_main_process()

        def _pin(mod):
            glm_ds_cpu_pin.install(mod)
        after_import("vllm.utils.system_utils", _pin)
    return w


def _cli(argv) -> int:
    """python3 glm_ds_hooks.py --check ROOT   (ROOT = the image's dist-packages, or an extracted tree)"""
    if len(argv) != 2 or argv[0] != "--check":
        print(_cli.__doc__)
        return 2
    got = hashes_at(argv[1], EXPECTED)
    bad = 0
    for k, v in EXPECTED.items():
        ok = got[k] == v
        bad += not ok
        print(f"{'OK  ' if ok else 'DIFF'} {k}  {got[k]}{'' if ok else '  expected ' + v}")
    print(f"{bad} differences")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(_cli(sys.argv[1:]))
