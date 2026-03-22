#!/usr/bin/env python3
"""
Visualize particle trajectories for a single channel case.
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# Folder and file with particle data (adjust as needed)
base = Path(__file__).parent
data_file = base / "particles_les.npz"   # e.g. rename your file to this

# Load data
data = np.load(data_file)
t = data["times"]         # (Nsaved,)
xp = data["xp"]           # (Nsaved, Nparticles, 3)

# Number of particles
Nparticles = xp.shape[1]

# Choose subset of particles to plot
n_show = min(20, Nparticles)
indices = np.linspace(0, Nparticles - 1, n_show, dtype=int)

fig, ax = plt.subplots(figsize=(6, 4))

for idx in indices:
    x = xp[:, idx, 0]
    y = xp[:, idx, 1]
    ax.plot(x, y, linewidth=1.0, alpha=0.8)

ax.set_xlabel("x [m]")
ax.set_ylabel("y [m]")
ax.set_title("Particle trajectories")
ax.grid(True)

plt.tight_layout()
plt.show()
