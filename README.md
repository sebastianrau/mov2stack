# mov2stack

Stack every frame from a MOV/video file into one still image.

The script is useful for long-exposure style composites, frame averaging, maximum
brightness stacks, and lightning photos extracted from video. It keeps the source
video resolution, shows a progress counter, supports optional motion compensation,
and can trim the video by start/stop seconds.

## Install

```bash
python3 -m pip install -r requirements.txt
```

Or install it as a local command:

```bash
python3 -m pip install .
```

## Basic Usage

```bash
python3 mov2stack.py input.mov
```

If no output file is given, the script writes:

```text
input_stacked.png
```

You can also choose the output path:

```bash
python3 mov2stack.py input.mov output.png
```

If installed with `pip install .`, you can run:

```bash
mov2stack input.mov --method lightning
```

## Examples

Mean stack:

```bash
python3 mov2stack.py input.mov --method mean
```

Maximum brightness stack:

```bash
python3 mov2stack.py input.mov --method max
```

Lightning optimized stack:

```bash
python3 mov2stack.py input.mov --method lightning
```

Trim the input video from 4.5s to 12s:

```bash
python3 mov2stack.py input.mov --method lightning --start 4.5 --stop 12
```

Use tracked translation motion compensation:

```bash
python3 mov2stack.py input.mov --method lightning --movement-compensation translation
```

For shaky lightning footage, try the stable foreground/lower-image region:

```bash
python3 mov2stack.py input.mov \
  --method lightning \
  --movement-compensation translation \
  --alignment-region bottom \
  --max-shift 20
```

## CLI Options

```text
python3 mov2stack.py [options] input [output]
```

| Option | Description |
| --- | --- |
| `input` | Input video path, for example `clip.mov`. |
| `output` | Optional output image path. Defaults to `<input name>_stacked.png`. |
| `-m`, `--method` | Stacking method: `mean`, `average`, `max`, `min`, `median`, or `lightning`. Default: `mean`. |
| `--movement-compensation`, `--motion-compensation` | Motion compensation method: `none`, `translation`, `phase`, `affine`, or `ecc`. Default: `none`. |
| `-j`, `--workers` | Worker threads for resize/alignment batches. Default: CPU count minus one. |
| `--chunk-size` | Frames to read before parallel preprocessing. Default: `32`. |
| `--alignment-region` | Region used by tracked translation: `bottom`, `center`, or `full`. Default: `bottom`. |
| `--max-shift` | Maximum accepted frame-to-frame translation in pixels. Larger shifts are ignored. Default: `20`. |
| `--start` | Start time in seconds. Default: `0`. |
| `--stop` | Stop time in seconds. Default: end of video. |

## Stacking Methods

| Method | Use case |
| --- | --- |
| `mean`, `average` | Smooth noise and create a normal averaged composite. |
| `max` | Keep the brightest value from all frames. Good for light trails. |
| `min` | Keep the darkest value from all frames. |
| `median` | Remove transient objects, but uses more memory because all frames are kept. |
| `lightning` | Two-pass lightning extractor that chooses a base frame, builds a quiet background, and overlays detected lightning detail. |

## Motion Compensation

| Method | Notes |
| --- | --- |
| `none` | Fastest. Use when the camera is locked off. |
| `translation` | Best first choice for handheld or slightly moving lightning footage. Tracks frame-to-frame shift and masks invalid warped borders. |
| `phase` | Fast global translation estimate. Can work on simple shifts, but may be less stable on dark footage. |
| `affine` | ECC-based alignment with shift, rotation, and scale. Slower and can fail on low-texture frames. |
| `ecc` | ECC translation alignment against the reference frame. Slower than `translation`. |

For lightning videos, `translation` with `--alignment-region bottom` is usually
the most useful because clouds and lightning change rapidly, while foreground
features often provide more stable tracking.

## Notes

- Output resolution matches the source video resolution reported by OpenCV.
- `--start` and `--stop` are in seconds and require OpenCV to read the video FPS.
- The progress bar reports processed frames within the selected trim range.
- GPU acceleration depends on your local OpenCV build. The standard
  `opencv-python` package used here normally runs these operations on CPU.

## Release

The current project version is defined in both `mov2stack.py` and
`pyproject.toml`.

```bash
python3 mov2stack.py --version
```

Build source and wheel distributions with:

```bash
python3 -m pip install build
python3 -m build
```
