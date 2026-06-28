#!/usr/bin/env python3
"""Stack frames from a MOV/video file into a single image."""

from __future__ import annotations

import argparse
import os
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any


STREAMING_METHODS = {"mean", "average", "max", "min"}
ALL_FRAME_METHODS = {"median"}
SPECIAL_METHODS = {"lightning"}
METHODS = sorted(STREAMING_METHODS | ALL_FRAME_METHODS | SPECIAL_METHODS)
MOTION_COMPENSATION_METHODS = {"none", "translation", "phase", "affine", "ecc"}
ALIGNMENT_REGIONS = {"full", "bottom", "center"}
Size = tuple[int, int]
__version__ = "0.1.0"


class ProgressBar:
    def __init__(self, total: int | None, width: int = 32) -> None:
        self.total = total if total and total > 0 else None
        self.width = width
        self.last_text = ""
        self.current = 0

    def update(self, current: int) -> None:
        self.current = current
        if self.total is None:
            text = f"Processed frames: {current}"
        else:
            percent = min(current / self.total, 1.0)
            filled = int(round(self.width * percent))
            bar = "#" * filled + "-" * (self.width - filled)
            text = (
                f"Processed frames: {current}/{self.total} "
                f"[{bar}] {percent * 100:6.0f}%"
            )

        padding = " " * max(len(self.last_text) - len(text), 0)
        print(f"\r{text}{padding}", end="", file=sys.stderr, flush=True)
        self.last_text = text

    def finish(self, current: int) -> None:
        if current != self.current:
            self.update(current)
        print(file=sys.stderr)


def load_dependencies() -> tuple[Any, Any]:
    try:
        import cv2
        import numpy as np
    except ModuleNotFoundError as exc:
        missing = exc.name or "a required package"
        raise SystemExit(
            f"Missing dependency: {missing}. Install dependencies with:\n"
            "  python3 -m pip install opencv-python numpy"
        ) from exc

    return cv2, np


def source_size(video: Any, cv2: Any) -> Size | None:
    width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
    if width <= 0 or height <= 0:
        return None

    video.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    video.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    return width, height


def trim_frame_range(
    video: Any,
    cv2: Any,
    start_seconds: float,
    stop_seconds: float | None,
) -> tuple[int, int | None]:
    fps = float(video.get(cv2.CAP_PROP_FPS) or 0.0)
    total_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT) or 0)

    if fps <= 0.0:
        if start_seconds > 0.0 or stop_seconds is not None:
            raise ValueError(
                "Could not determine video FPS for --start/--stop trimming."
            )
        return 0, total_frames if total_frames > 0 else None

    start_frame = int(round(start_seconds * fps))
    stop_frame = (
        int(round(stop_seconds * fps)) if stop_seconds is not None else total_frames
    )

    if total_frames > 0:
        start_frame = min(start_frame, total_frames)
        stop_frame = min(stop_frame, total_frames)

    frame_limit = max(stop_frame - start_frame, 0)
    return start_frame, frame_limit


def seek_to_frame(video: Any, cv2: Any, frame_index: int) -> None:
    if frame_index > 0:
        video.set(cv2.CAP_PROP_POS_FRAMES, frame_index)


def ensure_size(frame: Any, size: Size | None, cv2: Any) -> Any:
    if size is None:
        return frame

    width, height = size
    frame_height, frame_width = frame.shape[:2]
    if (frame_width, frame_height) == (width, height):
        return frame

    return cv2.resize(frame, (width, height), interpolation=cv2.INTER_LANCZOS4)


def frame_size(frame: Any) -> Size:
    height, width = frame.shape[:2]
    return width, height


def grayscale_float(frame: Any, cv2: Any, np: Any) -> Any:
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    return gray.astype(np.float32)


def tracking_gray(
    frame: Any,
    cv2: Any,
    np: Any,
    alignment_region: str,
    max_dimension: int = 960,
) -> tuple[Any, float, int, int]:
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    gray = clahe.apply(gray)
    height, width = gray.shape[:2]

    offset_x = 0
    offset_y = 0
    if alignment_region == "bottom":
        offset_y = height // 2
        gray = gray[offset_y:, :]
    elif alignment_region == "center":
        margin_x = width // 6
        margin_y = height // 6
        offset_x = margin_x
        offset_y = margin_y
        gray = gray[margin_y : height - margin_y, margin_x : width - margin_x]

    scale = min(max_dimension / max(width, height), 1.0)
    if scale < 1.0:
        gray = cv2.resize(
            gray,
            (int(width * scale), int(height * scale)),
            interpolation=cv2.INTER_AREA,
        )

    return gray, scale, offset_x, offset_y


def warp_frame(frame: Any, matrix: Any, cv2: Any, inverse_map: bool = False) -> Any:
    height, width = frame.shape[:2]
    flags = cv2.INTER_LANCZOS4
    if inverse_map:
        flags |= cv2.WARP_INVERSE_MAP

    return cv2.warpAffine(
        frame,
        matrix,
        (width, height),
        flags=flags,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(0, 0, 0),
    )


def valid_mask_for_frame(frame: Any, np: Any) -> Any:
    return np.full(frame.shape[:2], 255, dtype=np.uint8)


def warp_valid_mask(mask: Any, matrix: Any, cv2: Any, inverse_map: bool = False) -> Any:
    height, width = mask.shape[:2]
    flags = cv2.INTER_NEAREST
    if inverse_map:
        flags |= cv2.WARP_INVERSE_MAP

    return cv2.warpAffine(
        mask,
        matrix,
        (width, height),
        flags=flags,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0,
    )


def erode_valid_mask(mask: Any, cv2: Any, np: Any, pixels: int = 32) -> Any:
    if pixels <= 0:
        return mask > 0

    kernel = np.ones((pixels * 2 + 1, pixels * 2 + 1), dtype=np.uint8)
    return cv2.erode(mask, kernel, iterations=1) > 0


def compensate_phase_translation(
    frame: Any, reference_gray: Any, cv2: Any, np: Any
) -> Any:
    gray = grayscale_float(frame, cv2, np)
    shift, _response = cv2.phaseCorrelate(reference_gray, gray)
    dx, dy = shift
    matrix = np.array([[1.0, 0.0, -dx], [0.0, 1.0, -dy]], dtype=np.float32)

    return warp_frame(frame, matrix, cv2)


def compensate_ecc(
    frame: Any, reference_gray: Any, motion_compensation: str, cv2: Any, np: Any
) -> Any:
    gray = grayscale_float(frame, cv2, np)
    if motion_compensation == "affine":
        motion_model = cv2.MOTION_AFFINE
        matrix = np.eye(2, 3, dtype=np.float32)
    else:
        motion_model = cv2.MOTION_TRANSLATION
        matrix = np.eye(2, 3, dtype=np.float32)

    criteria = (
        cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT,
        80,
        1e-6,
    )

    try:
        _correlation, matrix = cv2.findTransformECC(
            reference_gray,
            gray,
            matrix,
            motion_model,
            criteria,
            None,
            3,
        )
    except cv2.error:
        return compensate_phase_translation(frame, reference_gray, cv2, np)

    return warp_frame(frame, matrix, cv2, inverse_map=True)


def compensate_movement(
    frame: Any,
    reference_gray: Any | None,
    motion_compensation: str,
    cv2: Any,
    np: Any,
) -> tuple[Any, Any]:
    if reference_gray is None:
        return frame, grayscale_float(frame, cv2, np)

    if motion_compensation == "phase":
        return (
            compensate_phase_translation(frame, reference_gray, cv2, np),
            reference_gray,
        )

    if motion_compensation in {"ecc", "affine"}:
        return compensate_ecc(
            frame, reference_gray, motion_compensation, cv2, np
        ), reference_gray

    return frame, reference_gray


def preprocess_frame(
    frame: Any,
    output_size: Size,
    reference_gray: Any,
    motion_compensation: str,
    cv2: Any,
    np: Any,
) -> Any:
    frame = ensure_size(frame, output_size, cv2)
    frame, _reference_gray = compensate_movement(
        frame, reference_gray, motion_compensation, cv2, np
    )
    return frame


def estimate_frame_translation(
    previous_gray: Any,
    current_gray: Any,
    scale: float,
    max_shift: float,
    cv2: Any,
    np: Any,
) -> tuple[float, float, int]:
    points = cv2.goodFeaturesToTrack(
        previous_gray,
        maxCorners=1000,
        qualityLevel=0.01,
        minDistance=12,
        blockSize=7,
    )
    if points is None or len(points) < 8:
        dx, dy, inliers = estimate_phase_shift(
            previous_gray, current_gray, scale, cv2, np
        )
        if abs(dx) > max_shift or abs(dy) > max_shift:
            return 0.0, 0.0, inliers
        return dx, dy, inliers

    next_points, status, _error = cv2.calcOpticalFlowPyrLK(
        previous_gray,
        current_gray,
        points,
        None,
        winSize=(31, 31),
        maxLevel=4,
        criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 40, 0.01),
    )
    if next_points is None or status is None:
        dx, dy, inliers = estimate_phase_shift(
            previous_gray, current_gray, scale, cv2, np
        )
        if abs(dx) > max_shift or abs(dy) > max_shift:
            return 0.0, 0.0, inliers
        return dx, dy, inliers

    good_previous = points[status.ravel() == 1].reshape(-1, 2)
    good_current = next_points[status.ravel() == 1].reshape(-1, 2)
    if len(good_previous) < 8:
        dx, dy, inliers = estimate_phase_shift(
            previous_gray, current_gray, scale, cv2, np
        )
        if abs(dx) > max_shift or abs(dy) > max_shift:
            return 0.0, 0.0, inliers
        return dx, dy, inliers

    transform, inliers = cv2.estimateAffinePartial2D(
        good_previous,
        good_current,
        method=cv2.RANSAC,
        ransacReprojThreshold=3.0,
        maxIters=2000,
        confidence=0.99,
    )
    if transform is not None and inliers is not None and int(inliers.sum()) >= 8:
        dx = float(transform[0, 2]) / scale
        dy = float(transform[1, 2]) / scale
        if abs(dx) <= max_shift and abs(dy) <= max_shift:
            return dx, dy, int(inliers.sum())
        return 0.0, 0.0, int(inliers.sum())

    deltas = good_current - good_previous
    median_delta = np.median(deltas, axis=0)
    dx = float(median_delta[0]) / scale
    dy = float(median_delta[1]) / scale
    if abs(dx) > max_shift or abs(dy) > max_shift:
        return 0.0, 0.0, len(good_previous)

    return dx, dy, len(good_previous)


def estimate_phase_shift(
    previous_gray: Any,
    current_gray: Any,
    scale: float,
    cv2: Any,
    np: Any,
) -> tuple[float, float, int]:
    shift, _response = cv2.phaseCorrelate(
        previous_gray.astype(np.float32), current_gray.astype(np.float32)
    )
    dx, dy = shift
    return float(dx) / scale, float(dy) / scale, 0


def read_batch(video: Any, chunk_size: int, remaining: int | None = None) -> list[Any]:
    frames = []
    limit = chunk_size if remaining is None else min(chunk_size, remaining)
    for _index in range(limit):
        ok, frame = video.read()
        if not ok:
            break
        frames.append(frame)

    return frames


def iter_tracked_translation_frames(
    video: Any,
    np: Any,
    cv2: Any,
    size: Size | None,
    progress: ProgressBar,
    alignment_region: str,
    max_shift: float,
    frame_limit: int | None,
) -> Any:
    if frame_limit == 0:
        raise ValueError("No frames were read from the selected trim range.")

    ok, first_frame = video.read()
    if not ok:
        raise ValueError("No frames were read from the video.")

    output_size = size or frame_size(first_frame)
    first_frame = ensure_size(first_frame, output_size, cv2)
    previous_gray, scale, _offset_x, _offset_y = tracking_gray(
        first_frame, cv2, np, alignment_region
    )
    total_dx = 0.0
    total_dy = 0.0
    count = 1
    progress.update(count)
    yield first_frame, output_size

    while frame_limit is None or count < frame_limit:
        ok, frame = video.read()
        if not ok:
            break

        frame = ensure_size(frame, output_size, cv2)
        current_gray, current_scale, _offset_x, _offset_y = tracking_gray(
            frame, cv2, np, alignment_region
        )
        if current_scale != scale:
            scale = current_scale

        dx, dy, _inliers = estimate_frame_translation(
            previous_gray, current_gray, scale, max_shift, cv2, np
        )
        total_dx += dx
        total_dy += dy
        matrix = np.array(
            [[1.0, 0.0, -total_dx], [0.0, 1.0, -total_dy]], dtype=np.float32
        )
        frame = warp_frame(frame, matrix, cv2)

        previous_gray = current_gray
        count += 1
        progress.update(count)
        yield frame, output_size


def iter_tracked_translation_frames_with_masks(
    video: Any,
    np: Any,
    cv2: Any,
    size: Size | None,
    progress: ProgressBar,
    alignment_region: str,
    max_shift: float,
    frame_limit: int | None,
) -> Any:
    if frame_limit == 0:
        raise ValueError("No frames were read from the selected trim range.")

    ok, first_frame = video.read()
    if not ok:
        raise ValueError("No frames were read from the video.")

    output_size = size or frame_size(first_frame)
    first_frame = ensure_size(first_frame, output_size, cv2)
    previous_gray, scale, _offset_x, _offset_y = tracking_gray(
        first_frame, cv2, np, alignment_region
    )
    total_dx = 0.0
    total_dy = 0.0
    count = 1
    progress.update(count)
    yield (
        first_frame,
        erode_valid_mask(valid_mask_for_frame(first_frame, np), cv2, np),
        output_size,
    )

    while frame_limit is None or count < frame_limit:
        ok, frame = video.read()
        if not ok:
            break

        frame = ensure_size(frame, output_size, cv2)
        current_gray, current_scale, _offset_x, _offset_y = tracking_gray(
            frame, cv2, np, alignment_region
        )
        if current_scale != scale:
            scale = current_scale

        dx, dy, _inliers = estimate_frame_translation(
            previous_gray, current_gray, scale, max_shift, cv2, np
        )
        total_dx += dx
        total_dy += dy
        matrix = np.array(
            [[1.0, 0.0, -total_dx], [0.0, 1.0, -total_dy]], dtype=np.float32
        )
        frame = warp_frame(frame, matrix, cv2)
        mask = warp_valid_mask(valid_mask_for_frame(frame, np), matrix, cv2)
        mask = erode_valid_mask(mask, cv2, np)

        previous_gray = current_gray
        count += 1
        progress.update(count)
        yield frame, mask, output_size


def iter_processed_frames_with_masks(
    video: Any,
    np: Any,
    cv2: Any,
    size: Size | None,
    motion_compensation: str,
    progress: ProgressBar,
    workers: int,
    chunk_size: int,
    alignment_region: str,
    max_shift: float,
    frame_limit: int | None,
) -> Any:
    if motion_compensation == "translation":
        yield from iter_tracked_translation_frames_with_masks(
            video, np, cv2, size, progress, alignment_region, max_shift, frame_limit
        )
        return

    for frame, output_size in iter_processed_frames(
        video,
        np,
        cv2,
        size,
        motion_compensation,
        progress,
        workers,
        chunk_size,
        alignment_region,
        max_shift,
        frame_limit,
    ):
        yield (
            frame,
            erode_valid_mask(valid_mask_for_frame(frame, np), cv2, np),
            output_size,
        )


def iter_processed_frames(
    video: Any,
    np: Any,
    cv2: Any,
    size: Size | None,
    motion_compensation: str,
    progress: ProgressBar,
    workers: int,
    chunk_size: int,
    alignment_region: str,
    max_shift: float,
    frame_limit: int | None,
) -> Any:
    """Yield preprocessed frames, parallelizing each bounded batch."""
    if motion_compensation == "translation":
        yield from iter_tracked_translation_frames(
            video, np, cv2, size, progress, alignment_region, max_shift, frame_limit
        )
        return

    if frame_limit == 0:
        raise ValueError("No frames were read from the selected trim range.")

    ok, first_frame = video.read()
    if not ok:
        raise ValueError("No frames were read from the video.")

    output_size = size or frame_size(first_frame)
    first_frame = ensure_size(first_frame, output_size, cv2)
    reference_gray = grayscale_float(first_frame, cv2, np)
    count = 1
    progress.update(1)
    yield first_frame, output_size

    def process_batch(batch: list[Any]) -> list[Any]:
        if workers == 1:
            return [
                preprocess_frame(
                    frame, output_size, reference_gray, motion_compensation, cv2, np
                )
                for frame in batch
            ]

        with ThreadPoolExecutor(max_workers=workers) as executor:
            return list(
                executor.map(
                    lambda frame: preprocess_frame(
                        frame,
                        output_size,
                        reference_gray,
                        motion_compensation,
                        cv2,
                        np,
                    ),
                    batch,
                )
            )

    while frame_limit is None or count < frame_limit:
        remaining = None if frame_limit is None else frame_limit - count
        batch = read_batch(video, chunk_size, remaining)
        if not batch:
            break

        for frame in process_batch(batch):
            count += 1
            progress.update(count)
            yield frame, output_size


def stack_streaming(
    video: Any,
    method: str,
    np: Any,
    cv2: Any,
    size: Size | None,
    motion_compensation: str,
    progress: ProgressBar,
    workers: int,
    chunk_size: int,
    alignment_region: str,
    max_shift: float,
    frame_limit: int | None,
) -> tuple[Any, int, Size]:
    """Stack frames without keeping the whole video in memory."""
    count = 0
    accumulator: np.ndarray | None = None

    output_size = size
    for frame, output_size in iter_processed_frames(
        video,
        np,
        cv2,
        size,
        motion_compensation,
        progress,
        workers,
        chunk_size,
        alignment_region,
        max_shift,
        frame_limit,
    ):
        frame_float = frame.astype(np.float64)
        if accumulator is None:
            accumulator = frame_float
        elif method in {"mean", "average"}:
            accumulator += frame_float
        elif method == "max":
            np.maximum(accumulator, frame_float, out=accumulator)
        elif method == "min":
            np.minimum(accumulator, frame_float, out=accumulator)

        count += 1

    if method in {"mean", "average"}:
        accumulator /= count

    return np.clip(accumulator, 0, 255).astype(np.uint8), count, output_size


def stack_all_frames(
    video: Any,
    method: str,
    np: Any,
    cv2: Any,
    size: Size | None,
    motion_compensation: str,
    progress: ProgressBar,
    workers: int,
    chunk_size: int,
    alignment_region: str,
    max_shift: float,
    frame_limit: int | None,
) -> tuple[Any, int, Size]:
    """Stack frames using methods that require the whole video in memory."""
    frames = []
    output_size = size
    for frame, output_size in iter_processed_frames(
        video,
        np,
        cv2,
        size,
        motion_compensation,
        progress,
        workers,
        chunk_size,
        alignment_region,
        max_shift,
        frame_limit,
    ):
        frames.append(frame)

    stack = np.stack(frames, axis=0)

    if method == "median":
        output = np.median(stack, axis=0)
    else:
        raise ValueError(f"Unsupported all-frame method: {method}")

    return np.clip(output, 0, 255).astype(np.uint8), len(frames), output_size


def luminance(frame: Any) -> Any:
    return 0.114 * frame[:, :, 0] + 0.587 * frame[:, :, 1] + 0.299 * frame[:, :, 2]


def lightning_detail(frame: Any, cv2: Any, np: Any) -> tuple[Any, Any]:
    frame_luma = luminance(frame.astype(np.float32))
    local_background = cv2.GaussianBlur(frame_luma, (0, 0), sigmaX=9.0)
    detail = frame_luma - local_background
    threshold = max(float(np.percentile(detail, 99.82)), 20.0)
    mask = (detail > threshold) & (frame_luma > 70.0)

    kernel = np.ones((2, 2), dtype=np.uint8)
    mask = cv2.dilate(mask.astype(np.uint8), kernel, iterations=1).astype(bool)
    return mask, detail


def lightning_base_score(frame: Any, np: Any) -> float:
    luma = luminance(frame.astype(np.float32))
    return float(np.percentile(luma, 99.2) + 0.25 * np.mean(luma))


def open_video(input_path: Path, cv2: Any) -> Any:
    video = cv2.VideoCapture(str(input_path))
    if not video.isOpened():
        raise ValueError(f"Could not open video: {input_path}")
    return video


def stack_lightning_video(
    input_path: Path,
    cv2: Any,
    np: Any,
    size: Size | None,
    total_frames: int | None,
    motion_compensation: str,
    workers: int,
    chunk_size: int,
    alignment_region: str,
    max_shift: float,
    start_frame: int,
    frame_limit: int | None,
    max_background_samples: int = 96,
) -> tuple[Any, int, Size]:
    sample_total = frame_limit or total_frames or max_background_samples
    sample_stride = max(sample_total // max_background_samples, 1)
    samples = []
    best_base = None
    best_base_score = -1.0

    print("Pass 1/2: selecting lightning base frame", file=sys.stderr)
    video = open_video(input_path, cv2)
    source_size(video, cv2)
    seek_to_frame(video, cv2, start_frame)
    progress = ProgressBar(total_frames)
    try:
        count = 0
        output_size = size
        for frame, valid_mask, output_size in iter_processed_frames_with_masks(
            video,
            np,
            cv2,
            size,
            motion_compensation,
            progress,
            workers,
            chunk_size,
            alignment_region,
            max_shift,
            frame_limit,
        ):
            if count % sample_stride == 0:
                samples.append(frame)
            score = lightning_base_score(frame, np) if valid_mask.any() else -1.0
            if score > best_base_score:
                best_base = frame.copy()
                best_base_score = score
            count += 1
        progress.finish(count)
    finally:
        video.release()

    if best_base is None or not samples:
        raise ValueError("No frames were read from the video.")

    quiet_background = np.percentile(np.stack(samples, axis=0), 65, axis=0).astype(
        np.float32
    )
    output = np.maximum(quiet_background, best_base.astype(np.float32) * 0.88)
    best_detail = np.zeros(output.shape[:2], dtype=np.float32)

    print("Pass 2/2: extracting lightning detail", file=sys.stderr)
    video = open_video(input_path, cv2)
    source_size(video, cv2)
    seek_to_frame(video, cv2, start_frame)
    progress = ProgressBar(total_frames)
    try:
        count = 0
        for frame, valid_mask, output_size in iter_processed_frames_with_masks(
            video,
            np,
            cv2,
            size,
            motion_compensation,
            progress,
            workers,
            chunk_size,
            alignment_region,
            max_shift,
            frame_limit,
        ):
            frame_float = frame.astype(np.float32)
            mask, detail = lightning_detail(frame_float, cv2, np)
            mask &= valid_mask
            better = mask & (detail > best_detail)
            output[better] = frame_float[better]
            best_detail[better] = detail[better]
            count += 1
        progress.finish(count)
    finally:
        video.release()

    return np.clip(output, 0, 255).astype(np.uint8), count, output_size


def stack_video(
    input_path: Path,
    output_path: Path,
    method: str,
    motion_compensation: str,
    workers: int,
    chunk_size: int,
    alignment_region: str,
    max_shift: float,
    start_seconds: float,
    stop_seconds: float | None,
) -> tuple[int, Size]:
    """Read a video, stack its frames, and write the resulting image."""
    cv2, np = load_dependencies()
    video = cv2.VideoCapture(str(input_path))
    if not video.isOpened():
        raise ValueError(f"Could not open video: {input_path}")

    size = source_size(video, cv2)
    start_frame, frame_limit = trim_frame_range(
        video, cv2, start_seconds, stop_seconds
    )
    if frame_limit == 0:
        raise ValueError("Selected trim range contains no frames.")
    progress = ProgressBar(frame_limit)
    seek_to_frame(video, cv2, start_frame)

    try:
        if method == "lightning":
            video.release()
            image, frame_count, output_size = stack_lightning_video(
                input_path,
                cv2,
                np,
                size,
                frame_limit,
                motion_compensation,
                workers,
                chunk_size,
                alignment_region,
                max_shift,
                start_frame,
                frame_limit,
            )
        elif method in STREAMING_METHODS:
            image, frame_count, output_size = stack_streaming(
                video,
                method,
                np,
                cv2,
                size,
                motion_compensation,
                progress,
                workers,
                chunk_size,
                alignment_region,
                max_shift,
                frame_limit,
            )
        else:
            image, frame_count, output_size = stack_all_frames(
                video,
                method,
                np,
                cv2,
                size,
                motion_compensation,
                progress,
                workers,
                chunk_size,
                alignment_region,
                max_shift,
                frame_limit,
            )
        if method != "lightning":
            progress.finish(frame_count)
    finally:
        video.release()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(output_path), image):
        raise ValueError(f"Could not write output image: {output_path}")

    return frame_count, output_size


def default_output_path(input_path: Path) -> Path:
    return input_path.with_name(f"{input_path.stem}_stacked.png")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Stack every frame from a MOV/video file into one image."
    )
    parser.add_argument(
        "--version",
        action="version",
        version=f"%(prog)s {__version__}",
    )
    parser.add_argument(
        "input", type=Path, help="Path to the input video, e.g. clip.mov"
    )
    parser.add_argument(
        "output",
        type=Path,
        nargs="?",
        help=(
            "Path for the output image, e.g. stacked.png or stacked.jpg. "
            "Default: <input name>_stacked.png"
        ),
    )
    parser.add_argument(
        "-m",
        "--method",
        choices=METHODS,
        default="mean",
        help="Frame stacking method. Default: mean",
    )
    parser.add_argument(
        "--movement-compensation",
        "--motion-compensation",
        choices=sorted(MOTION_COMPENSATION_METHODS),
        default="none",
        dest="motion_compensation",
        help=(
            "Align frames before stacking. Use 'translation' for tracked camera "
            "shift, 'affine' for ECC shift/rotation/scale, 'ecc' for direct "
            "ECC translation, or 'phase' for fast translation. Default: none"
        ),
    )
    parser.add_argument(
        "-j",
        "--workers",
        type=int,
        default=max((os.cpu_count() or 1) - 1, 1),
        help="Worker threads for resizing/alignment. Default: CPU count minus one",
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=32,
        help="Frames to read before parallel preprocessing. Default: 32",
    )
    parser.add_argument(
        "--alignment-region",
        choices=sorted(ALIGNMENT_REGIONS),
        default="bottom",
        help=(
            "Region used by tracked translation alignment. "
            "Use 'bottom' for lightning/night footage with stable foreground. "
            "Default: bottom"
        ),
    )
    parser.add_argument(
        "--max-shift",
        type=float,
        default=20.0,
        help=(
            "Maximum accepted frame-to-frame translation in pixels for tracked "
            "translation. Larger estimates are ignored. Default: 20"
        ),
    )
    parser.add_argument(
        "--start",
        type=float,
        default=0.0,
        help="Start time in seconds. Default: 0",
    )
    parser.add_argument(
        "--stop",
        type=float,
        default=None,
        help="Stop time in seconds. Default: end of video",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.workers < 1:
        raise SystemExit("--workers must be at least 1")
    if args.chunk_size < 1:
        raise SystemExit("--chunk-size must be at least 1")
    if args.max_shift <= 0:
        raise SystemExit("--max-shift must be greater than 0")
    if args.start < 0:
        raise SystemExit("--start must be at least 0")
    if args.stop is not None and args.stop <= args.start:
        raise SystemExit("--stop must be greater than --start")
    print(
        f"Stacking frames from {args.input} using method '{args.method}' "
        f"with motion compensation '{args.motion_compensation}' "
        f"using {args.workers} workers and chunk size {args.chunk_size}."
    )

    output_path = args.output or default_output_path(args.input)
    frame_count, output_size = stack_video(
        args.input,
        output_path,
        args.method,
        args.motion_compensation,
        args.workers,
        args.chunk_size,
        args.alignment_region,
        args.max_shift,
        args.start,
        args.stop,
    )
    width, height = output_size
    print(
        f"Stacked {frame_count} frames at {width}x{height} "
        f"with method '{args.method}' and motion compensation "
        f"'{args.motion_compensation}' into {output_path}"
    )


if __name__ == "__main__":
    main()
