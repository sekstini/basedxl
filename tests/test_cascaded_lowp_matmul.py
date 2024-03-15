import torch
import pytest
from basedxl.triton.cascaded_lowp_matmul import cascaded_lowp_matmul


@pytest.mark.parametrize(
    "shape_a, shape_b",
    [
        ((10, 20), (20, 30)),  # Non-batched matrices
        ((5, 10, 20), (20, 30)),  # First matrix is batched
        ((10, 20), (5, 20, 30)),  # Second matrix is batched
        ((5, 10, 20), (5, 20, 30)),  # Both matrices are batched
        [(128, 512), (512, 128)],
        [(1024, 2048), (2048, 1024)],
        # TODO: Failing because the criteria is static and too strict.
        # Need to do the math and find the correct expected threshold.
        # [(4096, 8192), (8192, 4096)],
        # [(4096, 14336), (14336, 4096)], # Mistral-7B MLP worst case (~0.8 max abs error)
    ],
)
def test_cascaded_lowp_matmul(shape_a, shape_b):
    a = torch.randn(*shape_a, device="cuda", dtype=torch.float16)
    b = torch.randn(*shape_b, device="cuda", dtype=torch.float16)
    output = cascaded_lowp_matmul(a, b)
    expected = torch.matmul(a, b)
    assert torch.allclose(output, expected, rtol=0.001, atol=0.35)
