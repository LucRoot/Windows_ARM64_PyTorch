import sys, time, statistics, platform
import torch, torch.nn as nn, torch.nn.functional as F

dtype = {"bf16": torch.bfloat16, "fp32": torch.float32}[sys.argv[1] if len(sys.argv) > 1 else "bf16"]
iters = int(sys.argv[2]) if len(sys.argv) > 2 else 20

torch.manual_seed(0)
print(f"interpreter: {platform.machine()} | torch {torch.__version__} | threads={torch.get_num_threads()} | dtype={dtype} iters={iters}")

class Block(nn.Module):
    def __init__(self, d=1024):
        super().__init__()
        self.l1 = nn.Linear(d, d * 4)
        self.l2 = nn.Linear(d * 4, d)
        self.norm = nn.LayerNorm(d)
    def forward(self, x):
        h = F.gelu(self.l1(x))
        return self.norm(x + self.l2(h))

model = nn.Sequential(*[Block() for _ in range(4)]).to(dtype)
opt = torch.optim.AdamW(model.parameters(), lr=1e-4)
x = torch.randn(16, 512, 1024, dtype=dtype)

t0 = time.perf_counter()
opt.zero_grad(); loss = model(x).sum(); loss.backward(); opt.step()
print(f"  first iter: {time.perf_counter()-t0:.3f}s (warmup, includes JIT/alloc)")

times = []
for i in range(iters):
    t0 = time.perf_counter()
    opt.zero_grad(); loss = model(x).sum(); loss.backward(); opt.step()
    dt = time.perf_counter() - t0
    times.append(dt)
    print(f"  iter {i+1}: {dt:.3f}s")

print(f"RESULT {platform.machine()} {dtype}: mean={statistics.mean(times):.3f}s "
      f"min={min(times):.3f}s max={max(times):.3f}s loss={loss.item():.1f}")
