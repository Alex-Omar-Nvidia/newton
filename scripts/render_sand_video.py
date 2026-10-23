# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Capture one sand experiment from a saved fit as a headless MP4."""

import argparse
import json
import subprocess
from pathlib import Path

import numpy as np
import warp as wp

import newton.viewer
from newton.examples.mpm.example_mpm_sand_tests import Example
from scripts.fit_sand import build_observe_command


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--best", type=Path, required=True)
    parser.add_argument(
        "--material-config", type=Path, help="Override the saved fit material from a JSON material record"
    )
    parser.add_argument("--experiment", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--frames", type=int, default=None)
    parser.add_argument("--width", type=int, default=960)
    parser.add_argument("--height", type=int, default=720)
    parser.add_argument("--video-fps", type=int, default=30)
    options = parser.parse_args()

    command = build_observe_command(options.best, experiment=options.experiment)
    args = Example.create_parser().parse_args(command[4:])
    if options.material_config is not None:
        reference = json.loads(options.best.read_text())["material"]
        material = json.loads(options.material_config.read_text())["material"]
        if set(material) != set(reference):
            parser.error("material config must contain the same material fields as the best fit")
        for name, value in material.items():
            setattr(args, name, value)
    if options.frames is not None:
        args.num_frames = options.frames
    if args.fps % options.video_fps:
        parser.error("simulation fps must be divisible by video fps")
    if options.width % 2 or options.height % 2:
        parser.error("video dimensions must be even")
    if options.output.exists():
        parser.error(f"refusing to overwrite {options.output}")
    options.output.parent.mkdir(parents=True, exist_ok=True)
    args.results = str(options.output.with_suffix(".json"))

    wp.init()
    wp.set_device(args.device)
    viewer = newton.viewer.ViewerGL(width=options.width, height=options.height, headless=True)
    encoder = None
    try:
        example = Example(viewer, args)
        encoder = subprocess.Popen(
            [
                "ffmpeg",
                "-n",
                "-loglevel",
                "error",
                "-f",
                "rawvideo",
                "-pixel_format",
                "rgb24",
                "-video_size",
                f"{options.width}x{options.height}",
                "-framerate",
                str(options.video_fps),
                "-i",
                "-",
                "-an",
                "-c:v",
                "libx264",
                "-preset",
                "veryfast",
                "-crf",
                "20",
                "-pix_fmt",
                "yuv420p",
                "-movflags",
                "+faststart",
                str(options.output),
            ],
            stdin=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        stride = args.fps // options.video_fps
        captured = 0
        for frame in range(args.num_frames):
            example.step()
            if (frame + 1) % stride == 0:
                example.render()
                pixels = viewer.get_frame().numpy()
                if pixels.shape[:2] != (options.height, options.width):
                    raise RuntimeError(f"unexpected frame shape {pixels.shape}")
                rgb = np.ascontiguousarray(pixels[:, :, :3], dtype=np.uint8)
                encoder.stdin.write(rgb.tobytes())
                captured += 1
            if (frame + 1) % args.fps == 0:
                print(
                    f"{options.experiment}: {(frame + 1) / args.fps:.0f}/{args.num_frames / args.fps:.0f}s", flush=True
                )
        encoder.stdin.close()
        error = encoder.stderr.read().decode()
        status = encoder.wait()
        if status:
            raise RuntimeError(f"ffmpeg exited {status}: {error}")
        print(json.dumps({"video": str(options.output), "frames": captured, "results": args.results}), flush=True)
    finally:
        if encoder is not None and encoder.poll() is None:
            encoder.kill()
            encoder.wait()
        viewer.close()


if __name__ == "__main__":
    main()
