from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Optional, Sequence, Tuple

import numpy as np
from PIL import Image, ImageFilter


IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".webp", ".bmp"}
VIDEO_SUFFIXES = {".gif", ".mp4", ".mov", ".webm", ".mkv"}


def _get_imageio():
    try:
        import imageio
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "imageio is required for GIF/video IO. Install dependencies from requirements.txt."
        ) from exc
    return imageio


@dataclass
class PlacementConfig:
    anchor_x: float = 0.5
    ground_y: float = 0.95
    height_fraction: float = 0.6
    end_anchor_x: Optional[float] = None
    end_ground_y: Optional[float] = None
    end_height_fraction: Optional[float] = None


@dataclass
class MaskConfig:
    mode: str = "alpha"
    key_color: Tuple[int, int, int] = (0, 255, 0)
    threshold: float = 60.0
    blur_radius: float = 2.0
    invert: bool = False


@dataclass
class ShadowConfig:
    enabled: bool = True
    opacity: float = 0.3
    blur_radius: float = 14.0
    offset_x: int = 8
    offset_y: int = 12
    scale: float = 1.03


def slugify(value: str, fallback: str = "scene") -> str:
    cleaned = "".join(ch.lower() if ch.isalnum() else "-" for ch in str(value or ""))
    parts = [part for part in cleaned.split("-") if part]
    return "-".join(parts)[:120] or fallback


def lerp(start: float, end: float, t: float) -> float:
    return start + (end - start) * t


def ensure_pil(image: Image.Image) -> Image.Image:
    if isinstance(image, Image.Image):
        return image
    raise TypeError(f"Expected PIL.Image.Image, got {type(image)}")


def load_frames(source: str | Path) -> List[Image.Image]:
    source_path = Path(source).expanduser()
    if not source_path.exists():
        raise FileNotFoundError(f"Frame source does not exist: {source_path}")

    if source_path.is_dir():
        frame_paths = sorted(
            path
            for path in source_path.iterdir()
            if path.suffix.lower() in IMAGE_SUFFIXES
        )
        if not frame_paths:
            raise ValueError(f"No image frames found in directory: {source_path}")
        return [Image.open(path).copy() for path in frame_paths]

    suffix = source_path.suffix.lower()
    if suffix in IMAGE_SUFFIXES:
        return [Image.open(source_path).copy()]

    if suffix in VIDEO_SUFFIXES:
        imageio = _get_imageio()
        frames = imageio.mimread(str(source_path))
        if not frames:
            raise ValueError(f"No frames decoded from media file: {source_path}")
        return [Image.fromarray(np.asarray(frame)) for frame in frames]

    raise ValueError(f"Unsupported frame source type: {source_path}")


def save_frames(frames: Sequence[Image.Image], path: str | Path, fps: int = 8) -> None:
    imageio = _get_imageio()
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    frame_arrays = [np.asarray(frame.convert("RGB"), dtype=np.uint8) for frame in frames]
    imageio.mimsave(str(output_path), frame_arrays, fps=fps)


def save_frame_sequence(frames: Sequence[Image.Image], directory: str | Path, prefix: str = "frame") -> None:
    out_dir = Path(directory)
    out_dir.mkdir(parents=True, exist_ok=True)
    for index, frame in enumerate(frames):
        frame.save(out_dir / f"{prefix}_{index:04d}.png")


def build_mask(
    image: Image.Image,
    mask_config: MaskConfig,
    explicit_mask: Optional[Image.Image] = None,
) -> Image.Image:
    if explicit_mask is not None:
        mask = ensure_pil(explicit_mask).convert("L")
    elif mask_config.mode == "alpha" and "A" in image.getbands():
        mask = image.getchannel("A").convert("L")
    else:
        rgb = np.asarray(image.convert("RGB"), dtype=np.float32)
        if mask_config.mode == "chroma_key":
            key = np.asarray(mask_config.key_color, dtype=np.float32)
            dist = np.sqrt(np.sum((rgb - key) ** 2, axis=-1))
            alpha = np.where(dist >= mask_config.threshold, 255, 0).astype(np.uint8)
        elif mask_config.mode == "white":
            dist = np.sqrt(np.sum((rgb - 255.0) ** 2, axis=-1))
            alpha = np.where(dist >= mask_config.threshold, 255, 0).astype(np.uint8)
        elif mask_config.mode == "black":
            dist = np.sqrt(np.sum(rgb**2, axis=-1))
            alpha = np.where(dist >= mask_config.threshold, 255, 0).astype(np.uint8)
        elif mask_config.mode == "luma":
            luma = (0.299 * rgb[..., 0]) + (0.587 * rgb[..., 1]) + (0.114 * rgb[..., 2])
            alpha = np.where(luma <= mask_config.threshold, 255, 0).astype(np.uint8)
        else:
            raise ValueError(f"Unsupported mask mode: {mask_config.mode}")
        mask = Image.fromarray(alpha, mode="L")

    if mask_config.invert:
        mask = Image.fromarray(255 - np.asarray(mask, dtype=np.uint8), mode="L")

    if mask_config.blur_radius > 0:
        mask = mask.filter(ImageFilter.GaussianBlur(radius=float(mask_config.blur_radius)))

    return mask


def _paste_with_box(base: Image.Image, overlay: Image.Image, position: Tuple[int, int], mask: Image.Image) -> None:
    box = (int(position[0]), int(position[1]))
    base.paste(overlay, box, mask)


def _shadow_from_mask(mask: Image.Image, shadow_config: ShadowConfig) -> Image.Image:
    alpha = np.asarray(mask, dtype=np.float32) * float(max(0.0, min(1.0, shadow_config.opacity)))
    shadow = np.zeros((mask.height, mask.width, 4), dtype=np.uint8)
    shadow[..., 3] = np.clip(alpha, 0, 255).astype(np.uint8)
    shadow_image = Image.fromarray(shadow, mode="RGBA")
    if shadow_config.blur_radius > 0:
        shadow_image = shadow_image.filter(ImageFilter.GaussianBlur(radius=float(shadow_config.blur_radius)))
    return shadow_image


def composite_foreground(
    scene_image: Image.Image,
    foreground_image: Image.Image,
    mask_image: Image.Image,
    *,
    anchor_x: float,
    ground_y: float,
    height_fraction: float,
    shadow_config: Optional[ShadowConfig] = None,
) -> Image.Image:
    scene_rgba = scene_image.convert("RGBA")
    foreground_rgba = foreground_image.convert("RGBA")
    mask_l = mask_image.convert("L")

    if foreground_rgba.width == 0 or foreground_rgba.height == 0:
        raise ValueError("Foreground image has invalid dimensions")

    target_height = max(1, int(scene_rgba.height * float(height_fraction)))
    scale = target_height / float(foreground_rgba.height)
    target_width = max(1, int(round(foreground_rgba.width * scale)))

    foreground_rgba = foreground_rgba.resize((target_width, target_height), resample=Image.LANCZOS)
    mask_l = mask_l.resize((target_width, target_height), resample=Image.LANCZOS)

    x = int(round(scene_rgba.width * float(anchor_x) - (target_width / 2.0)))
    y = int(round(scene_rgba.height * float(ground_y) - target_height))

    output = scene_rgba.copy()

    if shadow_config is not None and shadow_config.enabled:
        shadow_w = max(1, int(round(target_width * float(shadow_config.scale))))
        shadow_h = max(1, int(round(target_height * float(shadow_config.scale))))
        shadow_mask = mask_l.resize((shadow_w, shadow_h), resample=Image.LANCZOS)
        shadow_overlay = _shadow_from_mask(shadow_mask, shadow_config)
        _paste_with_box(
            output,
            shadow_overlay,
            (x + int(shadow_config.offset_x), y + int(shadow_config.offset_y)),
            shadow_overlay.getchannel("A"),
        )

    foreground_rgba.putalpha(mask_l)
    _paste_with_box(output, foreground_rgba, (x, y), mask_l)
    return output.convert("RGB")
