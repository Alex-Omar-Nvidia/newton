"""Plot the mesh and interfaces exported as CSV by example_brick_CB.m."""

from pathlib import Path

import matplotlib  # noqa: TID253
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402, TID253


example_dir = Path(__file__).resolve().parent
points = np.loadtxt(example_dir / "cb_rom_points.csv", delimiter=",", ndmin=2)
faces = np.loadtxt(
    example_dir / "cb_rom_faces.csv", delimiter=",", dtype=int, ndmin=2
) - 1  # MATLAB indices are one-based.
interfaces = np.loadtxt(example_dir / "cb_rom_interfaces.csv", delimiter=",", ndmin=2)

figure = plt.figure(figsize=(10, 6.5))
axes = figure.add_subplot(projection="3d")
axes.plot_trisurf(
    points[:, 0],
    points[:, 1],
    points[:, 2],
    triangles=faces,
    color="#bfd1e6",
    edgecolor="#404040",
    linewidth=0.5,
    alpha=0.9,
)

for interface, color, label in zip(
    interfaces,
    ("tab:red", "tab:blue"),
    ("Left rigid interface", "Right rigid interface"),
    strict=True,
):
    samples = np.isclose(points[:, 0], interface[0])
    axes.scatter(*points[samples].T, color=color, s=24)
    axes.scatter(*interface, color=color, marker="s", s=80, label=label)

axes.set_title("Craig–Bampton brick finite-element mesh")
axes.set_xlabel("x [m]", labelpad=14)
axes.set_ylabel("y [m]", labelpad=8)
axes.set_zlabel("z [m]", labelpad=8)
axes.set_xticks(np.linspace(points[:, 0].min(), points[:, 0].max(), 5))
axes.set_yticks(np.linspace(points[:, 1].min(), points[:, 1].max(), 3))
axes.set_zticks(np.linspace(points[:, 2].min(), points[:, 2].max(), 3))
axes.set_box_aspect(np.ptp(points, axis=0))
axes.view_init(elev=24, azim=-55)
axes.legend()
figure.savefig(example_dir / "example_brick_CB.png", dpi=180, bbox_inches="tight")
