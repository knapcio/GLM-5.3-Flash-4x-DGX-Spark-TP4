# SPDX-License-Identifier: Apache-2.0
"""Import hooks + drift guard for the exact GLM decode wins (diagnostics/glm-exact-wins).

  glm_target_argmax  GLM_TARGET_VOCAB_ARGMAX=1|check  vocab-parallel greedy target selection
  glm_kda_nocopy     GLM_KDA_NOCOPY=1                 KDA recurrent kernel reads q/k/v/g/beta in place

register() is called from overlay/sitecustomize.py (one appended block, see sitecustomize.snippet) in every
Python process of the container. With both variables unset or 0 it registers nothing and imports nothing
beyond this file. Each adapter's install(module) runs right after the engine module it patches has executed.
Before patching, the text of every engine function the adapter replaces or relies on is hashed (ast-extracted
source, sha256[:16], read from the file without importing it) and compared with the table below, taken from
the Tony v11 image (ghcr.io/tonyd2wild/vllm-glm53-flash:sm121-v11-dflash2 = 4def0ef6..., vLLM
0.1.dev20051+g487ecf187; the glm53-roce:v11-b58f34ea image does not touch these files). A mismatch refuses
to boot (GLM_EXACT_ALLOW_DRIFT=1 overrides, for a re-qualified image only). A failing install raises.

  python3 glm_exact_hooks.py --hashes [VLLM_PARENT_DIR]   prints the table for an image (run inside it)
"""
from __future__ import annotations

import ast
import hashlib
import importlib.abc
import importlib.util
import os
import sys
import textwrap

RUNNER = "vllm.v1.worker.gpu.model_runner"
KDA = "vllm.models.glm5next.nvidia.kda"

EXPECTED = {
    # --- GLM_TARGET_VOCAB_ARGMAX --------------------------------------------------------------------
    "vllm.v1.worker.gpu.model_runner:GPUModelRunner.sample": "d58876dc84455870",
    "vllm.v1.worker.gpu.sample.sampler:Sampler.__call__": "5ab01c7bbef92256",
    "vllm.v1.worker.gpu.sample.sampler:Sampler._requires_logits_processing": "49aa83c34acadedd",
    "vllm.v1.worker.gpu.sample.sampler:Sampler.sample": "0a9891bf03af6e71",
    "vllm.v1.worker.gpu.sample.sampler:Sampler.apply_sampling_params": "56558b1afa204b45",
    "vllm.v1.worker.gpu.sample.states:SamplingStates.max_num_logprobs": "ba695e7561fd9e70",
    "vllm.v1.worker.gpu.sample.gumbel:gumbel_sample": "ded980830618e668",
    "vllm.v1.worker.gpu.sample.gumbel:gumbel_block_argmax": "96bf9de0671d9e25",
    "vllm.v1.worker.gpu.spec_decode.rejection_sampler:RejectionSampler.__init__": "468fc771339b0472",
    "vllm.v1.worker.gpu.spec_decode.rejection_sampler:RejectionSampler.__call__": "5a297f779d7f774f",
    "vllm.v1.worker.gpu.spec_decode.rejection_sampler:RejectionSampler._verify": "e8e2dbd71fa67f53",
    "vllm.v1.worker.gpu.spec_decode.rejection_sampler:RejectionSampler._verify_in_chunks": "4e1a2f1099e88bc4",
    "vllm.v1.worker.gpu.spec_decode.rejection_sampler_utils:rejection_sample": "b08a507a5566fc51",
    "vllm.v1.worker.gpu.spec_decode.rejection_sampler_utils:_compute_local_logits_stats_kernel": "aad5d1e6ecf2390f",
    "vllm.v1.worker.gpu.spec_decode.rejection_sampler_utils:_compute_global_target_argmax": "7fc79ca11948c7f5",
    "vllm.v1.worker.gpu.spec_decode.rejection_sampler_utils:_rejection_kernel": "8f6e445f172306a9",
    "vllm.v1.worker.gpu.spec_decode.rejection_sampler_utils:_resample_kernel": "c9a3c8169da455fd",
    "vllm.v1.worker.gpu.spec_decode.rejection_sampler_utils:_insert_resampled_kernel": "486363149c679495",
    "vllm.v1.worker.gpu.input_batch:get_num_sampled_and_rejected": "57da8d694e3213c2",
    "vllm.model_executor.layers.logits_processor:LogitsProcessor.forward": "997b8331aec63267",
    "vllm.model_executor.layers.logits_processor:LogitsProcessor._get_logits": "2cb2548df232bbb1",
    "vllm.model_executor.layers.logits_processor:LogitsProcessor._gather_logits": "2ac2ccc43ad6af0a",
    "vllm.model_executor.layers.logits_processor:LogitsProcessor._apply_head": "a3d5c2eec69e2448",
    "vllm.model_executor.layers.logits_processor:LogitsProcessor.get_top_tokens": "54274b40c9850a2a",
    "vllm.models.glm5next.nvidia.model:Glm5NextForCausalLM.compute_logits": "d50f0cd6fd8c6537",
    "vllm.model_executor.models.glm4_1v:Glm4vForConditionalGeneration.compute_logits": "b13585b1fb288261",
    # --- GLM_KDA_NOCOPY -------------------------------------------------------------------------------
    "vllm.models.glm5next.nvidia.kda:Glm5NextLinearAttention.forward": "30746ac56387a7ef",
    "vllm.models.glm5next.nvidia.kda:Glm5NextLinearAttention._forward": "335c9ad1a3d5f0de",
    "vllm.third_party.flash_linear_attention.ops.kda:fused_recurrent_kda": "ea1877f6e20d8483",
    "vllm.third_party.flash_linear_attention.ops.kda:fused_recurrent_kda_fwd": "4848420ffb47640d",
    "vllm.third_party.flash_linear_attention.ops.fused_recurrent:fused_recurrent_gated_delta_rule_fwd_kernel": "3b8d2c84135d8d38",
}

NEEDS = {
    "argmax": [k for k in EXPECTED if not k.startswith(("vllm.models.glm5next.nvidia.kda",
                                                          "vllm.third_party.flash_linear_attention"))],
    "kda": [k for k in EXPECTED if k.startswith(("vllm.models.glm5next.nvidia.kda",
                                                 "vllm.third_party.flash_linear_attention"))],
}


def _on(name: str, env=None) -> bool:
    env = os.environ if env is None else env
    return env.get(name, "0").strip().lower() not in ("", "0", "off", "false", "no")


def wanted(env=None) -> dict:
    return {"argmax": _on("GLM_TARGET_VOCAB_ARGMAX", env), "kda": _on("GLM_KDA_NOCOPY", env)}


# ------------------------------------------------------------------------------------------------------
# drift guard
# ------------------------------------------------------------------------------------------------------
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


def vllm_parent() -> str:
    spec = importlib.util.find_spec("vllm")  # top-level package only; imports nothing below it
    return os.path.dirname(os.path.dirname(spec.origin))


def src_hash(root: str, key: str) -> str:
    mod, qual = key.split(":")
    path = os.path.join(root, *mod.split(".")) + ".py"
    return hashlib.sha256(func_source(path, qual).encode()).hexdigest()[:16]


def check(adapter: str, root: str | None = None) -> None:
    root = root or vllm_parent()
    bad = []
    for key in NEEDS[adapter]:
        want = EXPECTED[key]
        try:
            got = src_hash(root, key)
        except (OSError, KeyError) as exc:
            got = f"missing ({exc})"
        if want is not None and got != want:
            bad.append(f"{key}: {got} != {want}")
    if bad:
        msg = "glm-exact: engine drift for %s:\n  %s" % (adapter, "\n  ".join(bad))
        if os.environ.get("GLM_EXACT_ALLOW_DRIFT") == "1":
            sys.stderr.write(msg + "\n(GLM_EXACT_ALLOW_DRIFT=1: patching anyway)\n")
        else:
            raise RuntimeError(msg + "\nrefusing to patch (GLM_EXACT_ALLOW_DRIFT=1 overrides)")


# ------------------------------------------------------------------------------------------------------
# hooks
# ------------------------------------------------------------------------------------------------------
def _install_argmax(module) -> None:
    check("argmax")
    import glm_target_argmax
    glm_target_argmax.install(module)


def _install_kda(module) -> None:
    check("kda")
    import glm_kda_nocopy
    glm_kda_nocopy.install(module)


HOOKS: dict = {}


class _Finder(importlib.abc.MetaPathFinder):
    def find_spec(self, name, path, target=None):
        if name not in HOOKS:
            return None
        sys.meta_path.remove(self)
        try:
            spec = importlib.util.find_spec(name)  # lets the finders behind this one wrap too
        finally:
            sys.meta_path.insert(0, self)
        if spec is None or spec.loader is None:
            return None
        loader = spec.loader
        orig_exec = loader.exec_module
        fns = HOOKS.pop(name)

        def exec_module(module, _orig=orig_exec):
            _orig(module)
            for fn in fns:
                fn(module)

        loader.exec_module = exec_module
        return spec


def register(env=None) -> None:
    w = wanted(env)
    if w["argmax"]:
        HOOKS.setdefault(RUNNER, []).append(_install_argmax)
    if w["kda"]:
        HOOKS.setdefault(KDA, []).append(_install_kda)
    if not HOOKS:
        return
    for name in list(HOOKS):
        if name in sys.modules:  # already imported: patch in place
            for fn in HOOKS.pop(name):
                fn(sys.modules[name])
    if HOOKS:
        sys.meta_path.insert(0, _Finder())


if __name__ == "__main__" and len(sys.argv) > 1 and sys.argv[1] == "--hashes":
    r = sys.argv[2] if len(sys.argv) > 2 else vllm_parent()
    for k in EXPECTED:
        try:
            print(f'    "{k}": "{src_hash(r, k)}",')
        except (OSError, KeyError) as exc:
            print(f"    # {k}: {exc}")
