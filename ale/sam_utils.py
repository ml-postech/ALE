import torch
import numpy as np
from scipy.ndimage import binary_dilation
import os
import cv2
from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor
from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection


from .ore import find_sublist_indices
from .config import DEVICE, SAM2_MODEL_CONFIG, SAM2_CHECKPOINT_PATH, GROUNDING_MODEL_ID


sam_model = build_sam2(SAM2_MODEL_CONFIG, SAM2_CHECKPOINT_PATH, device=DEVICE)
sam_predictor = SAM2ImagePredictor(sam_model)
sam_processor = AutoProcessor.from_pretrained(GROUNDING_MODEL_ID)
grounding_model = AutoModelForZeroShotObjectDetection.from_pretrained(
    GROUNDING_MODEL_ID
).to(DEVICE)


def load_attention_masks_using_SAM(
    prompt_list,
    tokenizer,
    img,
    dilation_percent,
    box_threshold=0.45,
    text_threshold=0.4,
    device="cuda",
):
    source_mask = torch.ones((77, img.height, img.width), device=device).bool()
    object_list = [chunk for chunk, _ in prompt_list]
    sam_prompt = ". ".join(object_list) + "."
    inputs = sam_processor(images=img, text=sam_prompt, return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = grounding_model(**inputs)
    # what if results are empty?
    results = sam_processor.post_process_grounded_object_detection(
        outputs,
        inputs.input_ids,
        box_threshold=box_threshold,
        text_threshold=text_threshold,
        target_sizes=[img.size[::-1]],
    )
    # get the box prompt for SAM 2
    input_boxes = results[0]["boxes"].cpu().numpy()
    sam_predictor.set_image(np.array(img))
    raw_masks, _, _ = sam_predictor.predict(
        point_coords=None,
        point_labels=None,
        box=input_boxes,
        multimask_output=False,
    )
    # convert the shape to (n, H, W)
    if raw_masks.ndim == 4:
        raw_masks = raw_masks.squeeze(1)

    class_names = results[0]["labels"]

    print(f"SAM prompt: {sam_prompt}")
    print(f"Detected objects: {class_names}")

    raw_masks = np.array(
        [m for m, detected in zip(raw_masks, class_names) if detected in object_list]
    )
    class_names = [detected for detected in class_names if detected in object_list]

    print(f"Revised detected objects: {class_names}")

    # dilation
    kernel_size = int(raw_masks[0].shape[0] * dilation_percent)
    print(f"Extending masks by {kernel_size} pixels....")
    y, x = np.ogrid[-kernel_size : kernel_size + 1, -kernel_size : kernel_size + 1]
    circular_structure = x**2 + y**2 <= kernel_size**2
    # kernel = np.ones((kernel_size, kernel_size), dtype=bool)

    masks = np.array(
        [binary_dilation(mask, structure=circular_structure) for mask in raw_masks]
    )

    background_mask = np.ones_like(masks[0]) - masks.sum(axis=0)
    background_mask[background_mask < 1] = 0.0

    # background_mask = binary_erosion(background_mask, structure=kernel)

    if len(class_names) == len(object_list) and sorted(class_names) != sorted(
        object_list
    ):
        # missing detections...
        print("Grounded-SAM changed class names")

    background_mask = np.expand_dims(background_mask, axis=0)
    background_class_name = "###background###"
    class_names.append(background_class_name)
    masks = np.concatenate((masks, background_mask), axis=0)

    # merge same class masks
    merged_masks = {}
    for class_name, mask in zip(class_names, masks):
        if class_name not in merged_masks.keys():
            merged_masks[class_name] = mask
        else:
            merged_masks[class_name] += mask

    prompt_dict = dict(prompt_list)

    object_index_dict = dict(zip(object_list, range(len(object_list))))
    # masks + background
    raw_masks = torch.zeros(
        len(merged_masks), masks[0].shape[0], masks[0].shape[1], device=device
    ).bool()

    sam_dir = "_".join(object_list)
    for class_name, mask in merged_masks.items():
        # save mask
        os.makedirs(f"outputs/sam/{sam_dir}", exist_ok=True)
        cv2.imwrite(
            f"outputs/sam/{sam_dir}/{class_name}.png",
            mask.astype(np.uint8) * 255,
        )
        # apply mask
        mask = torch.tensor(mask, device=device).bool()
        raw_masks[object_index_dict.get(class_name, -1)] = mask
        if class_name == background_class_name:
            continue
        target_name = prompt_dict[class_name]
        target_ids = tokenizer(
            " and ".join([chunk for _, chunk in prompt_list])
        ).input_ids
        class_ids = tokenizer(target_name).input_ids[1:-1]  # remove BOS, EOS
        indices = find_sublist_indices(target_ids, class_ids)
        source_mask[indices] = mask

    return (
        source_mask,
        raw_masks,
    )
