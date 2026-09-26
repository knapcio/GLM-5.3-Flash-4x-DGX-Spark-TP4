"""LVKP-S-L2 startup hooks. Diagnostic and unqualified registrations omitted.
Active registration statements preserve the accepted deployment's ordering.
"""
import importlib.abc
import importlib.util
import os
import sys

_orig = "/usr/lib/python3.12/sitecustomize.py"
if os.path.exists(_orig):
    try:
        spec = importlib.util.spec_from_file_location("_distro_sitecustomize", _orig)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
    except Exception:
        pass



_POST = {}  # module name -> list of callables(module)




if any(os.environ.get(k, "0").strip().lower() not in ("", "0", "off", "false", "no")
       for k in ("GLM_LV_CGSTAT", "GLM_LV_SPLIT_FIX", "GLM_LV_ARGMAX_MINTOK", "GLM_LV_DRAFT_FP8_KV")):
    try:
        import glm_levers as _glv
        _glv.register_early()
        for _name, _fn in _glv.HOOKS.items():
            _POST.setdefault(_name, []).append(_fn)
    except Exception as exc:  # noqa: BLE001
        sys.stderr.write("glm-levers: import failed: %r\n" % (exc,))

if _POST:

    class _Hook(importlib.abc.MetaPathFinder):
        def find_spec(self, name, path, target=None):
            if name not in _POST:
                return None
            sys.meta_path.remove(self)
            try:
                spec = importlib.util.find_spec(name)
            finally:
                sys.meta_path.insert(0, self)
            if spec is None or spec.loader is None:
                return None
            loader = spec.loader
            exec_module = loader.exec_module
            fns = _POST.pop(name)

            def patched_exec(module):
                exec_module(module)
                for fn in fns:
                    try:
                        fn(module)
                    except Exception as exc:  # noqa: BLE001
                        sys.stderr.write("overlay hook for %s failed: %r\n" % (name, exc))
                        raise

            loader.exec_module = patched_exec
            return spec

    sys.meta_path.insert(0, _Hook())


if any(os.environ.get(k) == "1" for k in ("QMIX_FP8_BLOCK", "QMIX_DRAFT_HEAD_FP8", "QMIX_DEBUG_LAYERS")):
    import qmix_patch
    qmix_patch.register()

if os.environ.get("GLM_FAST_LOAD", "0").strip().lower() in ("1", "on", "true"):
    import glm_fast_load
    glm_fast_load.register()
if any(k.startswith("GLM_DS_") and os.environ[k].strip() not in ("", "0", "off") for k in os.environ):
    import glm_ds_hooks
    glm_ds_hooks.register()

if "1" in (os.environ.get("GLM_KDA_STASH"), os.environ.get("GLM_ROUTER_FP32OUT")):
    import glm_kda_stash
    glm_kda_stash.register()
if any(os.environ.get(k, "0").strip().lower() not in ("", "0", "off", "false", "no")
       for k in ("GLM_TARGET_VOCAB_ARGMAX", "GLM_KDA_NOCOPY")):
    import glm_exact_hooks
    glm_exact_hooks.register()
if any(os.environ.get(k, "0").strip().lower() not in ("", "0", "off", "false", "no")
       for k in ("GLM_KDA_STASH_NOCOPY", "GLM_KDA_FLAG_FUSED")):
    import glm_kda_stash_fast
    glm_kda_stash_fast.register()
if os.environ.get("GLM_L2_PREFETCH", "0").strip().lower() not in ("", "0", "off", "false", "no"):
    import glm_l2_prefetch
    glm_l2_prefetch.register()
