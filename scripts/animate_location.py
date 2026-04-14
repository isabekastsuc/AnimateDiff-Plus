import argparse
import datetime
import inspect
import os
from pathlib import Path

import numpy as np
import torch
import torchvision.transforms as transforms
from omegaconf import OmegaConf
from PIL import Image
from diffusers import AutoencoderKL, DDIMScheduler
from diffusers.utils.import_utils import is_xformers_available
from transformers import CLIPTextModel, CLIPTokenizer
from einops import rearrange

from animatediff.models.sparse_controlnet import SparseControlNetModel
from animatediff.models.unet import UNet3DConditionModel
from animatediff.pipelines.pipeline_animation import AnimationPipeline
from animatediff.pipelines.pipeline_location_composite import LocationCompositePipeline
from animatediff.utils.composite import (
    MaskConfig,
    PlacementConfig,
    ShadowConfig,
    load_frames,
    save_frame_sequence,
    save_frames,
    slugify,
)
from animatediff.utils.util import auto_download, load_weights
from animatediff.utils.util import safe_enable_xformers


def _to_pil_frames(video_tensor: torch.Tensor) -> list[Image.Image]:
    frames = []
    video = video_tensor[0].detach().cpu()
    for frame_idx in range(video.shape[1]):
        frame = video[:, frame_idx].permute(1, 2, 0).numpy()
        frame = np.clip(frame * 255.0, 0, 255).astype(np.uint8)
        frames.append(Image.fromarray(frame))
    return frames


def _center_crop_resize(image: Image.Image, width: int, height: int) -> Image.Image:
    src_w, src_h = image.size
    src_ratio = src_w / float(src_h)
    dst_ratio = width / float(height)

    if src_ratio > dst_ratio:
        new_w = int(src_h * dst_ratio)
        left = (src_w - new_w) // 2
        image = image.crop((left, 0, left + new_w, src_h))
    elif src_ratio < dst_ratio:
        new_h = int(src_w / dst_ratio)
        top = (src_h - new_h) // 2
        image = image.crop((0, top, src_w, top + new_h))

    return image.resize((width, height), resample=Image.LANCZOS)


def _try_save_frames(frames: list[Image.Image], path: Path, fps: int) -> None:
    try:
        save_frames(frames, path, fps=fps)
    except ModuleNotFoundError as exc:
        print(f"warning: could not write {path} because {exc}")


def _load_controlnet_images(config, height: int, width: int, vae, use_latent_condition: bool):
    image_paths = config.get("controlnet_images", [])
    if not image_paths:
        return None

    if isinstance(image_paths, str):
        image_paths = [image_paths]

    images = []
    for path in image_paths:
        image = Image.open(path).convert("RGB")
        image = _center_crop_resize(image, width, height)
        tensor = transforms.ToTensor()(image)
        if config.get("normalize_condition_images", False):
            tensor = tensor.mean(dim=0, keepdim=True).repeat(3, 1, 1)
            tensor -= tensor.min()
            tensor /= max(float(tensor.max()), 1e-6)
        images.append(tensor)

    controlnet_images = torch.stack(images).unsqueeze(0).cuda()
    controlnet_images = rearrange(controlnet_images, "b f c h w -> b c f h w")

    if use_latent_condition:
        num_controlnet_images = controlnet_images.shape[2]
        controlnet_images = rearrange(controlnet_images, "b c f h w -> (b f) c h w")
        controlnet_images = vae.encode(controlnet_images * 2.0 - 1.0).latent_dist.sample() * 0.18215
        controlnet_images = rearrange(controlnet_images, "(b f) c h w -> b c f h w", f=num_controlnet_images)

    return controlnet_images


def _build_pipeline(config):
    tokenizer = CLIPTokenizer.from_pretrained(config.pretrained_model_path, subfolder="tokenizer")
    text_encoder = CLIPTextModel.from_pretrained(config.pretrained_model_path, subfolder="text_encoder").cuda()
    vae = AutoencoderKL.from_pretrained(config.pretrained_model_path, subfolder="vae").cuda()

    inference_config = OmegaConf.load(config.get("inference_config", "configs/inference/inference-v3.yaml"))
    unet = UNet3DConditionModel.from_pretrained_2d(
        config.pretrained_model_path,
        subfolder="unet",
        unet_additional_kwargs=OmegaConf.to_container(inference_config.unet_additional_kwargs),
    ).cuda()

    controlnet = None
    controlnet_images = None
    if config.get("controlnet_path", ""):
        controlnet_config = OmegaConf.load(config.controlnet_config)
        unet.config.num_attention_heads = 8
        unet.config.projection_class_embeddings_input_dim = None

        controlnet = SparseControlNetModel.from_unet(
            unet,
            controlnet_additional_kwargs=controlnet_config.get("controlnet_additional_kwargs", {}),
        )
        auto_download(config.controlnet_path, is_dreambooth_lora=False)
        controlnet_state_dict = torch.load(config.controlnet_path, map_location="cpu")
        controlnet_state_dict = controlnet_state_dict["controlnet"] if "controlnet" in controlnet_state_dict else controlnet_state_dict
        controlnet_state_dict = {
            name: param for name, param in controlnet_state_dict.items() if "pos_encoder.pe" not in name
        }
        controlnet_state_dict.pop("animatediff_config", "")
        controlnet.load_state_dict(controlnet_state_dict)
        controlnet.cuda()

        use_latent_condition = bool(
            controlnet_config.get("controlnet_additional_kwargs", {}).get("use_simplified_condition_embedding", False)
        )
        controlnet_images = _load_controlnet_images(
            config,
            int(config.get("height", 512)),
            int(config.get("width", 512)),
            vae,
            use_latent_condition=use_latent_condition,
        )

    if is_xformers_available() and (not bool(config.get("without_xformers", False))):
        safe_enable_xformers(unet, "unet")
        if controlnet is not None:
            safe_enable_xformers(controlnet, "controlnet")

    pipeline = AnimationPipeline(
        vae=vae,
        text_encoder=text_encoder,
        tokenizer=tokenizer,
        unet=unet,
        controlnet=controlnet,
        scheduler=DDIMScheduler(**OmegaConf.to_container(inference_config.noise_scheduler_kwargs)),
    ).to("cuda")

    pipeline = load_weights(
        pipeline,
        motion_module_path=config.get("motion_module", ""),
        motion_module_lora_configs=config.get("motion_module_lora_configs", []),
        adapter_lora_path=config.get("adapter_lora_path", ""),
        adapter_lora_scale=config.get("adapter_lora_scale", 1.0),
        dreambooth_model_path=config.get("dreambooth_path", ""),
        lora_model_path=config.get("lora_model_path", ""),
        lora_alpha=config.get("lora_alpha", 0.8),
    ).to("cuda")

    return pipeline, controlnet_images


def _generate_frames(
    pipeline: AnimationPipeline,
    section,
    *,
    width: int,
    height: int,
    length: int,
    controlnet_images=None,
) -> list[Image.Image]:
    prompt = str(section.prompt)
    negative_prompt = str(section.get("negative_prompt", ""))
    seed = int(section.get("seed", -1))
    if seed != -1:
        torch.manual_seed(seed)
    else:
        torch.seed()

    sample = pipeline(
        prompt,
        negative_prompt=negative_prompt,
        num_inference_steps=int(section.get("steps", 25)),
        guidance_scale=float(section.get("guidance_scale", 7.5)),
        width=width,
        height=height,
        video_length=length,
        controlnet_images=controlnet_images,
        controlnet_image_index=section.get("controlnet_image_indexs", [0]),
    ).videos
    return _to_pil_frames(sample)


def _with_isolation_prompt(prompt: str, section) -> str:
    if not bool(section.get("auto_isolate_subject", True)):
        return prompt
    background_color = str(section.get("background_color", "green")).strip().lower() or "green"
    suffix = (
        f" isolated full body character, centered subject, plain {background_color} studio background, "
        "no environment, no scenery, no props"
    )
    return f"{prompt}{suffix}"


def _resolve_scene(scene_cfg, output_dir: Path, pipeline, default_width: int, default_height: int):
    cache_dir = Path(scene_cfg.get("cache_dir", output_dir / "scene_cache")).expanduser()
    cache_dir.mkdir(parents=True, exist_ok=True)

    image_path = str(scene_cfg.get("image_path", "")).strip()
    scene_key = str(scene_cfg.get("key", "")).strip()
    if not scene_key:
        scene_key = slugify(image_path or scene_cfg.get("prompt", "scene"))

    cached_scene_path = cache_dir / f"{scene_key}.png"
    if not bool(scene_cfg.get("regenerate", False)) and cached_scene_path.exists():
        return Image.open(cached_scene_path).convert("RGB"), cached_scene_path

    if image_path:
        scene_image = Image.open(Path(image_path).expanduser()).convert("RGB")
    else:
        width = int(scene_cfg.get("width", default_width))
        height = int(scene_cfg.get("height", default_height))
        scene_prompt = str(scene_cfg.prompt)
        no_character_suffix = (
            " empty location, environment only, no characters, no people, no creatures, no animals, no humanoids"
        )
        merged_prompt = f"{scene_prompt}{no_character_suffix}"
        temp_section = OmegaConf.create(
            {
                "prompt": merged_prompt,
                "negative_prompt": scene_cfg.get("negative_prompt", ""),
                "seed": scene_cfg.get("seed", -1),
                "steps": scene_cfg.get("steps", 25),
                "guidance_scale": scene_cfg.get("guidance_scale", 7.5),
                "controlnet_image_indexs": [0],
            }
        )
        scene_frames = _generate_frames(
            pipeline,
            temp_section,
            width=width,
            height=height,
            length=1,
            controlnet_images=None,
        )
        scene_image = scene_frames[0]

    scene_image.save(cached_scene_path)
    return scene_image, cached_scene_path


def main(args):
    *_, func_args = inspect.getargvalues(inspect.currentframe())
    _ = dict(func_args)

    config = OmegaConf.load(args.config)
    models_cfg = config.models
    scene_cfg = config.scene
    foreground_cfg = config.foreground
    composite_cfg = config.get("composite", {})
    output_cfg = config.get("outputs", {})

    time_str = datetime.datetime.now().strftime("%Y-%m-%dT%H-%M-%S")
    output_dir = Path(output_cfg.get("dir", f"samples/{Path(args.config).stem}-{time_str}")).expanduser()
    output_dir.mkdir(parents=True, exist_ok=True)

    models_cfg.width = int(foreground_cfg.get("width", scene_cfg.get("width", args.W)))
    models_cfg.height = int(foreground_cfg.get("height", scene_cfg.get("height", args.H)))

    pipeline, controlnet_images = _build_pipeline(models_cfg)

    scene_image, cached_scene_path = _resolve_scene(
        scene_cfg,
        output_dir,
        pipeline,
        default_width=models_cfg.width,
        default_height=models_cfg.height,
    )

    foreground_source = str(foreground_cfg.get("frames_path", "")).strip()
    mask_source = str(foreground_cfg.get("masks_path", "")).strip()

    if foreground_source:
        foreground_frames = load_frames(foreground_source)
    else:
        foreground_section = OmegaConf.create(
            {
                "prompt": _with_isolation_prompt(str(foreground_cfg.prompt), foreground_cfg),
                "negative_prompt": foreground_cfg.get(
                    "negative_prompt",
                    "environment, scenery, landscape, duplicate person, extra limbs, cropped subject",
                ),
                "seed": foreground_cfg.get("seed", -1),
                "steps": foreground_cfg.get("steps", 25),
                "guidance_scale": foreground_cfg.get("guidance_scale", 7.5),
                "controlnet_image_indexs": foreground_cfg.get("controlnet_image_indexs", [0]),
            }
        )
        foreground_frames = _generate_frames(
            pipeline,
            foreground_section,
            width=int(foreground_cfg.get("width", scene_image.width)),
            height=int(foreground_cfg.get("height", scene_image.height)),
            length=int(foreground_cfg.get("length", args.L)),
            controlnet_images=controlnet_images,
        )
        _try_save_frames(foreground_frames, output_dir / "foreground_raw.gif", fps=int(composite_cfg.get("fps", 8)))
        save_frame_sequence(foreground_frames, output_dir / "foreground_frames")

    mask_frames = load_frames(mask_source) if mask_source else None

    placement_cfg = PlacementConfig(**OmegaConf.to_container(composite_cfg.get("placement", {}), resolve=True))
    mask_cfg = MaskConfig(**OmegaConf.to_container(composite_cfg.get("mask", {}), resolve=True))
    shadow_cfg = ShadowConfig(**OmegaConf.to_container(composite_cfg.get("shadow", {}), resolve=True))

    compositor = LocationCompositePipeline(
        placement_config=placement_cfg,
        mask_config=mask_cfg,
        shadow_config=shadow_cfg,
    )
    composed = compositor(
        scene_image=scene_image,
        foreground_frames=foreground_frames,
        mask_frames=mask_frames,
        loop_foreground=bool(composite_cfg.get("loop_foreground", True)),
        total_frames=int(composite_cfg.get("total_frames", len(foreground_frames))),
    )

    final_fps = int(composite_cfg.get("fps", 8))
    final_path = output_dir / str(output_cfg.get("video_filename", "composite.mp4"))
    gif_path = output_dir / str(output_cfg.get("gif_filename", "composite.gif"))
    scene_copy_path = output_dir / "scene_plate.png"

    composed.scene.save(scene_copy_path)
    _try_save_frames(composed.frames, final_path, fps=final_fps)
    _try_save_frames(composed.frames, gif_path, fps=final_fps)
    save_frame_sequence(composed.frames, output_dir / "composited_frames")
    save_frame_sequence(composed.masks, output_dir / "mask_frames", prefix="mask")

    OmegaConf.save(config, output_dir / "config.resolved.yaml")
    print(f"scene plate: {cached_scene_path}")
    print(f"final video: {final_path}")
    print(f"final gif: {gif_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--L", type=int, default=16)
    parser.add_argument("--W", type=int, default=512)
    parser.add_argument("--H", type=int, default=512)
    args = parser.parse_args()
    main(args)
