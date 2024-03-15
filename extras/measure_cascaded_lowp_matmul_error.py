import pandas as pd
import torch
import torch.nn.functional as F
import numpy as np
from basedxl.triton.cascaded_lowp_matmul import cascaded_lowp_matmul

DUMP_TABLE: bool = True
EPS: float = 1e-6
QUANTILES: list[float] = [0.01, 0.2, 0.5, 0.8, 0.99]

rows = []
for M, N, K in [
    (4096, 4096, 4096),
    (4096, 14336, 4096),
    (14336, 4096, 4096),
    (4096, 4096, 14336),
]:
    torch.manual_seed(42)
    A = torch.randn(M, K, device="cuda", dtype=torch.float16)
    B = torch.randn(K, N, device="cuda", dtype=torch.float16)
    expected = A.double() @ B.double()

    lowp_out = cascaded_lowp_matmul(A, B)
    base_out = A @ B

    for acc_type, out in [("fp32", base_out), ("fp16", lowp_out)]:
        l1, mse = F.l1_loss(expected, out), F.mse_loss(expected, out)
        max_err = (expected - out).abs().max()
        rel_err = (expected - out).abs().div(expected.abs() + EPS)
        rel_err_cpu = rel_err.to("cpu", torch.float64).numpy()
        rel_err_quantiles = np.quantile(rel_err_cpu, QUANTILES)

        print(M, K, N, "-" * 20, acc_type, "acc", "-" * 20)
        print(f"\tL1 error: {l1.item():.3f}")
        print(f"\tMSE error: {mse.item():.3f}")
        print(f"\tMax error: {max_err.item():.3f}")
        print(f"\t          PERCENTILES: {' '.join(f'{q:10}' for q in QUANTILES)}")
        print(f"\t       relative error: {' '.join(f'{q:10.7f}' for q in rel_err_quantiles)}")

    if DUMP_TABLE:
        rows.append([M, K, N, l1.item(), mse.item(), max_err.item(), rel_err.median().item()])


if DUMP_TABLE:
    df = pd.DataFrame(rows, columns=["M", "K", "N", "L1", "MSE", "Max", "Median rel"])
    print("-" * 80)
    print(df)
    df.to_csv("results.csv", index=False)
    df.to_markdown("results.md", index=False)
