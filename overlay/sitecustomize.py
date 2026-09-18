"""Overlay sitecustomize: run the distro sitecustomize, then (only if VLLM_SUFFIX_DRAFT=1)
install suffix_draft after vllm.v1.worker.gpu.model_runner is imported, in whichever
process imports it (engine core / TP workers). No effect otherwise."""
import importlib.abc, importlib.machinery, importlib.util, os, sys

_orig = "/usr/lib/python3.12/sitecustomize.py"
if os.path.exists(_orig):
    try:
        spec = importlib.util.spec_from_file_location("_distro_sitecustomize", _orig)
        mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)
    except Exception:
        pass

if os.environ.get("VLLM_SUFFIX_DRAFT") == "1":
    TARGET = "vllm.v1.worker.gpu.model_runner"

    class _Hook(importlib.abc.MetaPathFinder):
        def find_spec(self, name, path, target=None):
            if name != TARGET:
                return None
            sys.meta_path.remove(self)
            spec = importlib.util.find_spec(name)
            if spec is None or spec.loader is None:
                return None
            loader = spec.loader
            exec_module = loader.exec_module

            def patched_exec(module):
                exec_module(module)
                try:
                    import suffix_draft
                    suffix_draft.install()
                except Exception as exc:  # noqa: BLE001
                    sys.stderr.write("suffix-draft: install failed: %r\n" % (exc,))

            loader.exec_module = patched_exec
            return spec

    sys.meta_path.insert(0, _Hook())
