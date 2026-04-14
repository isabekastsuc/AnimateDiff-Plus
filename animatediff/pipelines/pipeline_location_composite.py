from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Sequence

import numpy as np
from PIL import Image

from animatediff.utils.composite import (
    MaskConfig,
    PlacementConfig,
    ShadowConfig,
    build_mask,
    composite_foreground,
    lerp,
)


@dataclass
class LocationCompositePipelineOutput:
    frames: List[Image.Image]
    masks: List[Image.Image]
    scene: Image.Image


class LocationCompositePipeline:
    def __init__(
        self,
        placement_config: Optional[PlacementConfig] = None,
        mask_config: Optional[MaskConfig] = None,
        shadow_config: Optional[ShadowConfig] = None,
    ):
        self.placement_config = placement_config or PlacementConfig()
        self.mask_config = mask_config or MaskConfig()
        self.shadow_config = shadow_config or ShadowConfig()

    def __call__(
        self,
        *,
        scene_image: Image.Image,
        foreground_frames: Sequence[Image.Image],
        mask_frames: Optional[Sequence[Image.Image]] = None,
        loop_foreground: bool = True,
        total_frames: Optional[int] = None,
    ) -> LocationCompositePipelineOutput:
        if not foreground_frames:
            raise ValueError("foreground_frames is required")

        scene_rgb = scene_image.convert("RGB")
        frame_count = int(total_frames or len(foreground_frames))
        composed_frames: List[Image.Image] = []
        resolved_masks: List[Image.Image] = []

        for frame_index in range(frame_count):
            source_index = frame_index
            if loop_foreground:
                source_index = frame_index % len(foreground_frames)
            else:
                source_index = min(frame_index, len(foreground_frames) - 1)

            foreground = foreground_frames[source_index].convert("RGBA")
            explicit_mask = None
            if mask_frames:
                explicit_mask = mask_frames[source_index % len(mask_frames)] if loop_foreground else mask_frames[min(source_index, len(mask_frames) - 1)]
            mask = build_mask(foreground, self.mask_config, explicit_mask=explicit_mask)
            resolved_masks.append(mask)

            if frame_count == 1:
                progress = 0.0
            else:
                progress = frame_index / float(frame_count - 1)

            placement = self._interpolate_placement(progress)
            composed = composite_foreground(
                scene_rgb,
                foreground,
                mask,
                anchor_x=placement.anchor_x,
                ground_y=placement.ground_y,
                height_fraction=placement.height_fraction,
                shadow_config=self.shadow_config,
            )
            composed_frames.append(composed)

        return LocationCompositePipelineOutput(
            frames=composed_frames,
            masks=resolved_masks,
            scene=scene_rgb,
        )

    def _interpolate_placement(self, progress: float) -> PlacementConfig:
        base = self.placement_config
        return PlacementConfig(
            anchor_x=lerp(base.anchor_x, base.end_anchor_x if base.end_anchor_x is not None else base.anchor_x, progress),
            ground_y=lerp(base.ground_y, base.end_ground_y if base.end_ground_y is not None else base.ground_y, progress),
            height_fraction=lerp(
                base.height_fraction,
                base.end_height_fraction if base.end_height_fraction is not None else base.height_fraction,
                progress,
            ),
        )
