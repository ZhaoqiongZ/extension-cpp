import torch
from torch.testing._internal.common_utils import TestCase
import unittest
import sycl_extension

def reference_muladd(a, b, c):
    return a * b + c

class TestMyMulAdd(TestCase):
    def sample_inputs(self, device, *, requires_grad=False):
        def make_tensor(*size):
            return torch.randn(size, device=device, requires_grad=requires_grad)

        def make_nondiff_tensor(*size):
            return torch.randn(size, device=device, requires_grad=False)

        return [
            [make_tensor(3), make_tensor(3), 1],
            [make_tensor(20), make_tensor(20), 3.14],
            [make_tensor(20), make_nondiff_tensor(20), -123],
            [make_nondiff_tensor(2, 3), make_tensor(2, 3), -0.3],
        ]

    def _test_correctness(self, device):
        samples = self.sample_inputs(device)
        for args in samples:
            result = sycl_extension.ops.mymuladd(*args)
            expected = reference_muladd(*args)
            torch.testing.assert_close(result, expected)

    @unittest.skipIf(not torch.xpu.is_available(), "requires Intel GPU")
    def test_correctness_xpu(self):
        self._test_correctness("xpu")

if __name__ == "__main__":
    unittest.main()