#!/usr/bin/env python3
"""CPU tests for overlay/glm_roce_proxy_pin.py: core choice, thread discovery, off by default."""
import os
import sys
import tempfile
import types
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "overlay"))
import glm_roce_proxy_pin as pp  # noqa: E402


class Pin(unittest.TestCase):
    def test_choose(self):
        aff, big = set(range(20)), set(range(5, 10)) | set(range(15, 20))
        self.assertEqual(pp.choose("", aff, big), set())
        self.assertEqual(pp.choose("off", aff, big), set())
        self.assertEqual(pp.choose("auto", aff, big), {19})
        self.assertEqual(pp.choose("auto", set(range(10)), big), {9})
        self.assertEqual(pp.choose("auto", {0, 1}, big), {1})          # no big core allowed: highest allowed
        self.assertEqual(pp.choose("9,19", aff, big), {9, 19})
        self.assertEqual(pp.choose("9,19", set(range(10)), big), {9})  # clipped to the affinity

    def test_parse_and_tids(self):
        self.assertEqual(pp.parse_cpus("5-9,15"), {5, 6, 7, 8, 9, 15})
        self.assertEqual(pp.new_tids({1, 2}, {1, 2, 77}), {77})

    def test_ring_target(self):
        self.assertEqual(pp.target_module({}), "b12x.comm.roce._proxy")
        self.assertEqual(pp.target_module({"GLM_ROCE_RING": "0"}), "b12x.comm.roce._proxy")
        self.assertEqual(pp.target_module({"GLM_ROCE_RING": "1"}), "b12x.comm.roce_ring._proxy")
        self.assertEqual(pp.target_module({"GLM_ROCE_RING": " 1 "}), "b12x.comm.roce_ring._proxy")

    def test_ring_module_pinned_when_selected(self):
        os.environ.pop(pp.ENV, None)
        os.environ[pp.ENV] = "auto"
        os.environ[pp.ENV_RING] = "1"
        target = "b12x.comm.roce_ring._proxy"
        mod = types.ModuleType(target)

        class Proxy:
            def start(self):
                return None

        mod.Proxy = Proxy
        sys.modules[target] = mod
        try:
            pp.register()
            self.assertTrue(getattr(Proxy, "_glm_pin", False))
        finally:
            sys.modules.pop(target, None)
            os.environ.pop(pp.ENV, None)
            os.environ.pop(pp.ENV_RING, None)

    def test_big_cores_from_sysfs(self):
        with tempfile.TemporaryDirectory() as d:
            for c, cap in ((0, 512), (1, 512), (2, 1024), (3, 1024)):
                os.makedirs(os.path.join(d, f"cpu{c}"))
                with open(os.path.join(d, f"cpu{c}", "cpu_capacity"), "w") as f:
                    f.write(str(cap))
            self.assertEqual(pp.big_cores(d), {2, 3})
            self.assertEqual(pp.big_cores(os.path.join(d, "missing"), "5-6"), {5, 6})

    def test_off_by_default(self):
        os.environ.pop(pp.ENV, None)
        mod = types.ModuleType(pp.TARGET)

        class Proxy:
            def start(self):
                return None
        mod.Proxy = Proxy
        sys.modules[pp.TARGET] = mod
        try:
            pp.register()
            self.assertFalse(getattr(Proxy, "_glm_pin", False))
            os.environ[pp.ENV] = "auto"
            pp.register()
            self.assertTrue(Proxy._glm_pin)
        finally:
            sys.modules.pop(pp.TARGET, None)
            os.environ.pop(pp.ENV, None)


if __name__ == "__main__":
    unittest.main(verbosity=2)
