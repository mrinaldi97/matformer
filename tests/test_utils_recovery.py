import subprocess
import sys
import unittest


class UtilitiesRecoveryTests(unittest.TestCase):
    def test_import_orders_and_cpu_backward(self):
        for first_import in (
            "matformer.models",
            "matformer.transformer_blocks",
            "matformer.matformer_registry",
        ):
            with self.subTest(first_import=first_import):
                code = f"""
import {first_import}
import torch
from matformer.matformer_registry import registry

layer = registry.create(
    "linear", "linear", preferred="torch",
    in_features=4, out_features=3,
)
x = torch.randn(2, 4, requires_grad=True)
output = layer(x)
assert output.shape == (2, 3)
output.square().mean().backward()
assert x.grad is not None
assert torch.isfinite(x.grad).all()
"""
                subprocess.run([sys.executable, "-c", code], check=True)

    def test_cache_helper_and_causal_mask(self):
        import torch
        from matformer.utils.matformer_cache import (
            CachedStuff, ensure_cache_and_registry,
        )
        from matformer.matformer_registry import registry

        cache = CachedStuff()
        self.assertIs(ensure_cache_and_registry(cache), cache)
        self.assertIs(cache.registry, registry)

        q = torch.zeros(1, 2, 4, 8)
        mask = cache.get_attention_mask(
            query=q, kv=q, nheads=2, causal=True,
        )
        self.assertEqual(tuple(mask.shape), (1, 2, 4, 4))
        self.assertTrue(torch.isneginf(mask[0, 0, 0, 1]).item())
        self.assertEqual(mask[0, 0, 1, 0].item(), 0.0)


if __name__ == "__main__":
    unittest.main()
