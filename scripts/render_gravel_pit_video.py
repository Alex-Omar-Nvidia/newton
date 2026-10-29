# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Render a headless MP4 and measurements for one uniform gravel friction value.

Example: uv run python scripts/render_gravel_pit_video.py --friction 0.3 \
    --output /tmp/gravel_friction_0.3.mp4
"""

import argparse
import csv
import subprocess
from pathlib import Path

import numpy as np
import warp as wp
from PIL import Image, ImageDraw, ImageFont  # noqa: TID253 - standalone renderer requires Pillow

import newton.viewer
from newton.examples.multiphysics.example_vbd_mpm_gravel_pit import Example


def draw_plot(draw, values, bounds, color):
    x0, y0, x1, y1 = bounds
    draw.rectangle(bounds, outline=(80, 88, 100))
    if len(values) < 2:
        return
    maximum = max(*values, 1.0e-9)
    points = [
        (x0 + (x1 - x0) * i / (len(values) - 1), y1 - (y1 - y0) * max(0.0, value) / maximum)
        for i, value in enumerate(values)
    ]
    draw.line(points, fill=color, width=2)


def annotate(frame, example, energies, penetrations, font):
    image = Image.fromarray(frame).convert("RGB")
    overlay = Image.new("RGBA", image.size)
    draw = ImageDraw.Draw(overlay)
    draw.rounded_rectangle((12, 12, 395, 236), radius=8, fill=(15, 20, 28, 220))
    lines = [
        f"Friction {example.material['friction']:.4f} | t={example.sim_time:.2f} s",
        f"MPM stiffness {example.material['young_modulus'] / 1.0e6:g} MPa",
        f"MPM yield pressure {example.material['yield_pressure'] / 1.0e6:g} MPa",
        f"Tire penetration {penetrations[-1] * 100:.1f} cm",
        f"Floor clearance {example.metrics['Wheel bottom clearance [m]'] * 100:.1f} cm",
    ]
    for i, line in enumerate(lines):
        draw.text((22, 18 + 25 * i), line, font=font, fill=(245, 245, 245))
    draw.text((22, 146), "Gravel energy", font=font, fill=(220, 200, 155))
    draw_plot(draw, energies, (165, 153, 380, 174), (226, 190, 110))
    draw.text((22, 186), "Tire depth", font=font, fill=(255, 125, 95))
    draw_plot(draw, penetrations, (165, 193, 380, 214), (255, 125, 95))
    return Image.alpha_composite(image.convert("RGBA"), overlay).convert("RGB")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--friction", type=float, required=True, help="Uniform MPM gravel friction coefficient")
    parser.add_argument("--output", type=Path, required=True, help="Output MP4 path")
    parser.add_argument("--seconds", type=float, default=20.0)
    parser.add_argument("--fps", type=int, default=20)
    parser.add_argument("--substeps", type=int, default=2)
    parser.add_argument(
        "--playback-speed",
        type=int,
        default=1,
        help="Show every Nth simulation frame at the original playback FPS (N-times faster)",
    )
    parser.add_argument("--width", type=int, default=960)
    parser.add_argument("--height", type=int, default=720)
    parser.add_argument("--device", default="cuda:0")
    options = parser.parse_args()
    if options.seconds <= 0 or options.fps <= 0 or options.substeps <= 0 or options.playback_speed <= 0:
        parser.error("seconds, fps, substeps, and playback-speed must be positive")
    if options.width % 2 or options.height % 2:
        parser.error("width and height must be even for H.264 output")
    simulation_frames = round(options.seconds * options.fps)
    if simulation_frames % options.playback_speed:
        parser.error("simulation frame count must be divisible by playback-speed")

    options.output.parent.mkdir(parents=True, exist_ok=True)
    wp.init()
    wp.set_device(options.device)
    args = Example.create_parser().parse_args([])
    args.mpm_friction = options.friction
    args.simulation_fps = options.fps
    args.simulation_substeps = options.substeps
    viewer = newton.viewer.ViewerGL(width=options.width, height=options.height, headless=True)
    example = Example(viewer, args)
    friction = example.model.mpm.friction.numpy()[example.mpm_particles]
    np.testing.assert_allclose(friction, options.friction)
    try:
        font = ImageFont.truetype("DejaVuSans.ttf", 18)
    except OSError:
        font = ImageFont.load_default(size=18)
    ffmpeg = subprocess.Popen(
        [
            "ffmpeg",
            "-y",
            "-loglevel",
            "error",
            "-f",
            "rawvideo",
            "-pixel_format",
            "rgb24",
            "-video_size",
            f"{options.width}x{options.height}",
            "-framerate",
            str(options.fps),
            "-i",
            "-",
            "-an",
            "-c:v",
            "libx264",
            "-preset",
            "veryfast",
            "-crf",
            "22",
            "-pix_fmt",
            "yuv420p",
            "-movflags",
            "+faststart",
            str(options.output),
        ],
        stdin=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    energies = []
    penetrations = []
    metrics_path = options.output.with_suffix(".csv")
    try:
        with metrics_path.open("w", newline="") as csv_file:
            writer = csv.writer(csv_file)
            writer.writerow(
                [
                    "time_s",
                    "mpm_friction",
                    "mpm_young_modulus_Pa",
                    "mpm_yield_pressure_Pa",
                    "gravel_kinetic_energy_J",
                    "gravel_rms_speed_m_s",
                    "tire_penetration_m",
                    "tire_bottom_clearance_m",
                ]
            )
            for frame in range(simulation_frames):
                example.step()
                metrics = example.metrics
                energies.append(metrics["Gravel kinetic energy [J]"])
                penetrations.append(metrics["Wheel penetration below release surface [m]"])
                writer.writerow(
                    [
                        example.sim_time,
                        example.material["friction"],
                        example.material["young_modulus"],
                        example.material["yield_pressure"],
                        energies[-1],
                        metrics["Gravel RMS speed [m/s]"],
                        penetrations[-1],
                        metrics["Wheel bottom clearance [m]"],
                    ]
                )
                if (frame + 1) % options.playback_speed == 0:
                    example.render()
                    image = annotate(viewer.get_frame().numpy(), example, energies, penetrations, font)
                    ffmpeg.stdin.write(image.tobytes())
                if (frame + 1) % options.fps == 0:
                    if not np.isfinite(example.state_0.particle_q.numpy()).all():
                        raise RuntimeError("Simulation produced non-finite particle positions")
                    print(
                        f"friction={options.friction:g} t={example.sim_time:.1f}s "
                        f"depth={penetrations[-1]:.3f}m clearance={metrics['Wheel bottom clearance [m]']:.3f}m",
                        flush=True,
                    )
    finally:
        ffmpeg.stdin.close()
        error = ffmpeg.stderr.read().decode()
        result = ffmpeg.wait()
        viewer.close()
    if result:
        raise RuntimeError(f"ffmpeg failed with status {result}: {error}")
    print(f"Saved {options.output} and {metrics_path}", flush=True)


if __name__ == "__main__":
    main()
