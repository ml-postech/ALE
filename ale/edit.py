import torch

from .config import DEVICE, DTYPE, MODEL_ID
from .attention_control import AttentionRefine, BackgroundBlend
from .ore import get_object_restricted_embeds, get_substituted_base_embeds
from .sam_utils import load_attention_masks_using_SAM
from . import rgb_cam
from diffusers import LCMScheduler
from .pipeline_ead import EditPipeline


def load_edit_pipeline(model_id=MODEL_ID):
    scheduler = LCMScheduler.from_pretrained(model_id, subfolder="scheduler")
    pipe_edit = EditPipeline.from_pretrained(
        model_id, scheduler=scheduler, torch_dtype=DTYPE
    )

    pipe_edit = pipe_edit.to(DEVICE)
    return pipe_edit


def ale_edit(
    img,
    source_prompt,
    target_prompt,
    source_prompt2,
    target_prompt2,
    source_prompt3,
    target_prompt3,
    negative_prompt="",
    guidance_s=1,
    guidance_t=2,
    num_inference_steps=15,
    width=768,
    height=768,
    seed=42,
    self_replace_steps=0.5,
    dilation_percent=0.04,
    box_threshold=0.45,
    text_threshold=0.4,
    pipe=None,
):
    if pipe is None:
        raise TypeError("pipe must be provided.")

    prompt_list = [
        (source, target)
        for (source, target) in zip(
            [s.strip() for s in [source_prompt, source_prompt2, source_prompt3]],
            [t.strip() for t in [target_prompt, target_prompt2, target_prompt3]],
        )
        if source != "" and target != ""
    ]
    source_list = [s for s, _ in prompt_list]
    target_list = [t for _, t in prompt_list]

    base_source_prompt = " and ".join([s for s, _ in prompt_list])
    base_target_prompt = " and ".join([t for _, t in prompt_list])

    torch.manual_seed(seed)

    img = img.resize((height, width))
    print("Image size: ", img.height, img.width)

    assert len(prompt_list) != 0, "no editing prompts"

    # masks for RGB-CAM and BB
    mask_list, raw_masks = load_attention_masks_using_SAM(
        prompt_list,
        pipe.tokenizer,
        img,
        dilation_percent,
        box_threshold=box_threshold,
        text_threshold=text_threshold,
    )

    # setup BB
    background_blend = BackgroundBlend(
        save_inter=False,
    )
    background_blend.background_mask = raw_masks[-1]
    controller = AttentionRefine(
        num_inference_steps,
        start_steps=0,
        self_replace_steps=self_replace_steps,
        background_blend=background_blend,
        num_heads=8,
    )

    prompt_embeds_source = get_substituted_base_embeds(
        base_source_prompt, source_list, pipe=pipe
    )

    # get OREs
    prompt_embeds = get_object_restricted_embeds(base_target_prompt, target_list, pipe)

    # register RGB-CAM
    rgb_cam.register_attention_control_RGB(
        pipe,
        controller,
        prompt_embeds,
        raw_masks=raw_masks,
        mask_list=mask_list,
    )

    results = pipe(
        prompt=base_target_prompt,
        source_prompt=base_source_prompt,
        negative_prompt=negative_prompt,
        image=img,
        num_inference_steps=num_inference_steps,
        guidance_scale=guidance_t,
        source_guidance_scale=guidance_s,
        callback=controller.step_callback,
        prompt_embeds_source=prompt_embeds_source,
        denoise_model=False,
    )

    return results.images[0]
