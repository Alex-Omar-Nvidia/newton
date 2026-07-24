# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Render the three-interface T-bracket and its reference probes."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

EXAMPLE_DIR = Path(__file__).resolve().parent


def main() -> None:
    points = np.loadtxt(EXAMPLE_DIR / "cb_rom_points.csv", delimiter=",", ndmin=2)
    faces = np.loadtxt(EXAMPLE_DIR / "cb_rom_faces.csv", delimiter=",", dtype=int, ndmin=2) - 1
    interfaces = np.loadtxt(EXAMPLE_DIR / "cb_rom_interfaces.csv", delimiter=",", ndmin=2)

    figure = plt.figure(figsize=(7.2, 5.2), dpi=160)
    axes = figure.add_subplot(projection="3d")
    surface = Poly3DCollection(
        points[faces],
        facecolor="#76a9dc",
        edgecolor="#24496b",
        linewidth=0.25,
        alpha=0.88,
    )
    axes.add_collection3d(surface)
    marker_offsets = np.array([[-0.035, 0.0, 0.0], [0.035, 0.0, 0.0], [0.0, 0.0, 0.035]])
    marker_points = interfaces + marker_offsets
    axes.scatter(
        marker_points[:, 0],
        marker_points[:, 1],
        marker_points[:, 2],
        color=["#ef6351", "#ef6351", "#f4b942"],
        edgecolor="#4a251f",
        linewidth=0.5,
        s=80,
        depthshade=False,
    )
    label_offsets = (
        np.array([-0.10, -0.015, 0.03]),
        np.array([0.02, -0.015, 0.03]),
        np.array([0.02, -0.015, 0.03]),
    )
    for label, point, offset in zip(("left", "right", "top"), marker_points, label_offsets, strict=True):
        axes.text(*(point + offset), label, fontsize=9, color="#301612")

    axes.set(
        xlabel="x [m]",
        ylabel="y [m]",
        zlabel="z [m]",
        xlim=(-0.7, 0.7),
        ylim=(-0.3, 0.3),
        zlim=(-0.2, 0.7),
        title="Three-interface Craig-Bampton T-bracket",
    )
    axes.set_box_aspect((1.4, 0.6, 0.9))
    axes.view_init(elev=22, azim=-62)
    figure.tight_layout()
    figure.savefig(EXAMPLE_DIR / "example_cb_multi_interface.png", bbox_inches="tight")
    figure.set_size_inches(4.0, 4.0)
    figure.savefig(
        EXAMPLE_DIR.parents[3] / "docs/images/examples/example_cb_multi_interface.jpg",
        dpi=80,
        facecolor="white",
    )


if __name__ == "__main__":
    main()
