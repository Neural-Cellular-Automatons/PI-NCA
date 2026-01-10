# ============================================================
# MULTI-STEP FUSED PHYSICS-INFORMED NCA (FULLY FIXED)
# ============================================================

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
import imageio
from torch.cuda.amp import autocast, GradScaler
import torch.nn.functional as F

# ============================================================
# 1. SETUP
# ============================================================
torch.manual_seed(42)
np.random.seed(42)

torch.backends.cudnn.benchmark = True
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device:", device)

# ============================================================
# 2. CONFIG
# ============================================================
CONFIG = {
    "train_size": 64,
    "test_size": 128,
    "batch_size": 64,

    "epochs": 900,
    "min_steps": 20,
    "max_steps": 2000,

    "lr": 5e-4,
    "weight_decay": 1e-5,

    "alpha": 0.5,
    "dt": 0.1,

    "truncate_depth": 64,
    "solver_chunk": 8,
    "val_freq": 50,

    "final_test_steps": 8000
}

# ============================================================
# 3. HEAT EQUATION SOLVER (PERIODIC + FUSED)
# ============================================================
class HeatEquationSolver(nn.Module):
    def __init__(self, alpha, dt):
        super().__init__()
        self.alpha_dt = alpha * dt

        kernel = torch.tensor(
            [[0, 1, 0],
             [1, -4, 1],
             [0, 1, 0]],
            dtype=torch.float32
        ).view(1, 1, 3, 3)

        self.laplace = nn.Conv2d(
            1, 1, 3, padding=1, padding_mode="circular", bias=False
        )
        self.laplace.weight.data = kernel
        self.laplace.requires_grad_(False)

    def step(self, u):
        return u + self.alpha_dt * self.laplace(u)

    def k_steps(self, u, k):
        for _ in range(k):
            u = self.step(u)
        return u

# ============================================================
# 4. ENERGY CONSERVATION (FIXED SHAPE)
# ============================================================
def conserve_energy(u, target_sum):
    diff = (target_sum - u.sum(dim=(1,2,3), keepdim=True)) / (u.shape[-1] ** 2)
    return u + diff

# ============================================================
# 5. DEEP FLUX NCA
# ============================================================
class DeepFluxNCA(nn.Module):
    def __init__(self):
        super().__init__()
        self.perceive = nn.Conv2d(
            1, 32, 3, padding=1, padding_mode="circular"
        )
        self.process = nn.Sequential(
            nn.ReLU(),
            nn.Conv2d(32, 64, 1),
            nn.ReLU(),
            nn.Conv2d(64, 32, 1),
            nn.ReLU(),
            nn.Conv2d(32, 2, 1, bias=False)
        )
        with torch.no_grad():
            self.process[-1].weight.zero_()

    def forward(self, x):
        flux = self.process(self.perceive(x))
        fx, fy = flux[:, 0:1], flux[:, 1:2]
        dx = (torch.roll(fx, 1, 3) - fx) + (torch.roll(fy, 1, 2) - fy)
        return x + dx

# ============================================================
# 6. DATA GENERATION (PERIODIC BLOBS)
# ============================================================
def make_state(batch, size):
    grid = torch.zeros(batch, 1, size, size, device=device)
    x = torch.arange(size, device=device).float()
    y = torch.arange(size, device=device).float()
    yy, xx = torch.meshgrid(y, x, indexing="ij")

    for i in range(batch):
        for _ in range(np.random.randint(3, 6)):
            cx, cy = np.random.randint(0, size, 2)
            sigma = size * 0.08
            amp = np.random.uniform(5.0, 10.0)

            dx = torch.min(torch.abs(xx - cx), size - torch.abs(xx - cx))
            dy = torch.min(torch.abs(yy - cy), size - torch.abs(yy - cy))

            grid[i, 0] += amp * torch.exp(-(dx**2 + dy**2) / (2 * sigma**2))
    return grid

# ============================================================
# 7. TRAINING
# ============================================================
model = DeepFluxNCA().to(device)
solver = HeatEquationSolver(CONFIG["alpha"], CONFIG["dt"]).to(device)
solver.k_steps = torch.compile(solver.k_steps)

optimizer = optim.Adam(model.parameters(), lr=CONFIG["lr"], weight_decay=CONFIG["weight_decay"])
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=300, gamma=0.5)
loss_fn = nn.MSELoss()
scaler = GradScaler()

train_loss, val_loss = [], []

print("\n--- TRAINING STARTED ---\n")

for epoch in range(CONFIG["epochs"]):
    model.train()

    state = make_state(CONFIG["batch_size"], CONFIG["train_size"])
    target_sum = state.sum(dim=(1,2,3), keepdim=True)

    horizon = int(
        CONFIG["min_steps"] +
        epoch * (CONFIG["max_steps"] - CONFIG["min_steps"]) / CONFIG["epochs"]
    )
    steps = np.random.randint(CONFIG["min_steps"], horizon + 1)

    pred, tgt = state.clone(), state.clone()
    acc_loss = torch.tensor(0.0, device=device)

    t = 0
    while t < steps:
        k = min(CONFIG["solver_chunk"], steps - t)

        with torch.no_grad():
            tgt = solver.k_steps(tgt, k)

        with autocast():
            for _ in range(k):
                pred = model(pred)
                pred = conserve_energy(pred, target_sum)
            acc_loss = acc_loss + loss_fn(pred, tgt)

        if (t + k) % CONFIG["truncate_depth"] == 0:
            optimizer.zero_grad()
            scaler.scale(acc_loss).backward()
            scaler.step(optimizer)
            scaler.update()
            pred, tgt = pred.detach(), tgt.detach()
            acc_loss = torch.tensor(0.0, device=device)

        t += k

    if acc_loss.item() > 0:
        optimizer.zero_grad()
        scaler.scale(acc_loss).backward()
        scaler.step(optimizer)
        scaler.update()

    scheduler.step()
    train_loss.append(acc_loss.item())

    if epoch % CONFIG["val_freq"] == 0:
        model.eval()
        with torch.no_grad():
            v = make_state(1, CONFIG["train_size"])
            vt, vp = v.clone(), v.clone()
            for _ in range(500):
                vt = solver.step(vt)
                vp = model(vp)
            v_loss = loss_fn(vp, vt).item()
        val_loss.append((epoch, v_loss))
        print(f"Epoch {epoch:4d} | Train {train_loss[-1]:.2e} | Val {v_loss:.2e}")

# ============================================================
# 8. FINAL TEST + GIFS (FIXED)
# ============================================================
print("\n--- FINAL TEST & GIFS ---\n")

model.eval()
u0 = make_state(1, CONFIG["test_size"])
ut, um = u0.clone(), u0.clone()
energy0 = u0.sum(dim=(1,2,3), keepdim=True)

frames_ana, frames_mod, frames_err = [], [], []
errors, energies = [], []

for t in range(CONFIG["final_test_steps"]):
    with torch.no_grad():
        ut = solver.step(ut)
        um = model(um)
        um = conserve_energy(um, energy0)

    errors.append(torch.mean((ut - um) ** 2).item())
    energies.append(um.sum().item())

    if t % 200 == 0:
        vmax = u0.max().item()
        fig, ax = plt.subplots(1, 3, figsize=(15, 5))

        ax[0].imshow(ut[0,0].cpu(), cmap="inferno", vmin=0, vmax=vmax)
        ax[0].set_title("Analytical")

        ax[1].imshow(um[0,0].cpu(), cmap="inferno", vmin=0, vmax=vmax)
        ax[1].set_title("Model")

        ax[2].imshow(torch.abs(ut-um)[0,0].cpu(),
                     cmap="viridis", vmin=0, vmax=vmax*0.1)
        ax[2].set_title("Error")

        for a in ax:
            a.axis("off")

        fig.canvas.draw()
        frame = np.asarray(fig.canvas.buffer_rgba())[:, :, :3]

        w = frame.shape[1] // 3
        frames_ana.append(frame[:, :w])
        frames_mod.append(frame[:, w:2*w])
        frames_err.append(frame[:, 2*w:])

        plt.close(fig)

imageio.mimsave("analytic.gif", frames_ana, fps=10)
imageio.mimsave("model.gif", frames_mod, fps=10)
imageio.mimsave("error.gif", frames_err, fps=10)

# ============================================================
# 9. INFERENCE PLOTS
# ============================================================
plt.figure(figsize=(18,4))

plt.subplot(1,3,1)
plt.plot(train_loss)
plt.yscale("log")
plt.title("Training Loss")

plt.subplot(1,3,2)
e, v = zip(*val_loss)
plt.plot(e, v, "o-")
plt.yscale("log")
plt.title("Validation Loss")

plt.subplot(1,3,3)
plt.plot(errors)
plt.yscale("log")
plt.title("Long-Horizon MSE")

plt.tight_layout()
plt.show()

plt.figure()
plt.plot(np.array(energies) - energy0.item())
plt.title("Energy Drift")
plt.show()

print("\nDONE.")