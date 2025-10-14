import json
import argparse
import os
import numpy as np
from PIL import Image, ImageChops
import csv
from metrics_calculator import MetricsCalculator, calculate_metric
import re
import pandas as pd

# Mapping from target_count to directories containing ALE-edited result images
edit_result_dirs = {
    1: "./results/1",
    2: "./results/2",
    3: "./results/3",
}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--metrics",
        nargs="+",
        type=str,
        default=[
            "TELS",
            "TILS",
            "structure_distance",
            "editing_performance",
            "psnr_unedit_part",
            "lpips_unedit_part",
            "mse_unedit_part",
            "ssim_unedit_part",
        ],
    )
    parser.add_argument("--src_image_folder", type=str, default="ale_bench/images")
    parser.add_argument(
        "--tgt_folders",
        nargs="+",
        type=int,
        default=[1, 2, 3],
        help="Target counts to evaluate (keys of edit_result_dirs).",
    )
    parser.add_argument(
        "--csv_name",
        type=str,
        default="evaluation_results.csv",
        help="Path to save the evaluation CSV file.",
    )
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument(
        "--edit_category_list",
        nargs="+",
        type=str,
        default=["color", "object", "material", "color+object", "object+material"],
        help="Edit categories to evaluate (should match subfolder names).",
    )
    parser.add_argument(
        "--stats_dir",
        type=str,
        default="./stats",
        help="Directory to save Excel stats files.",
    )

    args = parser.parse_args()

    metrics = args.metrics
    src_image_folder = args.src_image_folder
    tgt_folders = args.tgt_folders
    edit_category_list = args.edit_category_list
    stats_dir = args.stats_dir

    masks_dir = "ale_bench/masks"

    # Resolve target_count -> image result folder
    tgt_image_folders = {key: edit_result_dirs[key] for key in tgt_folders}

    csv_path = os.path.join(stats_dir, args.csv_name)
    metrics_calculator = MetricsCalculator(args.device)

    # 1) Write CSV header
    os.makedirs(stats_dir, exist_ok=True)
    with open(csv_path, "w", newline="") as f:
        csv_write = csv.writer(f)
        csv_head = [metric for metric in metrics]
        data_row = ["file_id", "target_count", "edit_type"] + csv_head
        csv_write.writerow(data_row)

    # 2) Iterate result images and compute metrics → append to CSV
    for target_count, target_image_folder in tgt_image_folders.items():
        for edit_type in edit_category_list:
            result_base_dir = os.path.join(target_image_folder, edit_type)
            if not os.path.isdir(result_base_dir):
                continue
            for image_name in os.listdir(result_base_dir):
                src_image_file = os.path.join(src_image_folder, f"{image_name}.jpg")
                if not os.path.isfile(src_image_file):
                    continue
                src_image = Image.open(src_image_file)

                edit_result_path = os.path.join(result_base_dir, image_name)
                masks_path = os.path.join(masks_dir, image_name)
                if not os.path.isdir(masks_path):
                    continue

                mask_dict = {
                    mask.split("_")[0]: Image.open(
                        os.path.join(masks_path, mask)
                    ).convert("L")
                    for mask in os.listdir(masks_path)
                    if ".png" in mask
                }

                prompt_file = os.path.join(edit_result_path, "prompt.json")
                if not os.path.isfile(prompt_file):
                    continue
                with open(prompt_file, "r") as f:
                    data = json.load(f)
                    prompt_list = data["prompts"]

                # For every generated edited image (*.png) in the case folder
                for edit_result_image in [
                    img for img in os.listdir(edit_result_path) if ".png" in img
                ]:
                    target_image_path = os.path.join(
                        edit_result_path, edit_result_image
                    )
                    target_image = Image.open(target_image_path).resize(src_image.size)
                    edit_index = int(re.search(r"\d+", edit_result_image).group())
                    print(
                        f".......calculating {image_name} object {target_count} {edit_type} edit {edit_index}......."
                    )

                    edits_dict = dict(prompt_list[edit_index]["edits"])
                    source_masks = [mask_dict[src] for src, _ in edits_dict.items()]
                    merged_source_mask = source_masks[0]
                    for mask in source_masks[1:]:
                        merged_source_mask = ImageChops.lighter(
                            merged_source_mask, mask
                        )

                    merged_source_mask = (
                        np.array(merged_source_mask)[:, :, np.newaxis].repeat(
                            [3], axis=2
                        )
                        / 255.0
                    )

                    source_prompt = " and ".join([src for src, _ in edits_dict.items()])
                    editing_prompt = " and ".join(
                        [target for _, target in edits_dict.items()]
                    )

                    # Compute all metrics for this sample
                    evaluation_result = [
                        f"{image_name}_{edit_result_image}",  # file_id
                        target_count,
                        edit_type,
                    ]
                    for metric in metrics:
                        print(f"evaluating metric: {metric}")
                        evaluation_result.append(
                            calculate_metric(
                                metrics_calculator,
                                metric,
                                src_image,
                                target_image,
                                merged_source_mask,
                                merged_source_mask,
                                source_prompt,
                                editing_prompt,
                                edits_dict,
                                mask_dict,
                            )
                        )
                    with open(csv_path, "a+", newline="") as f:
                        csv.writer(f).writerow(evaluation_result)

    # 3) Compute stats and save as Excel files
    df = pd.read_csv(csv_path)

    mean_per_target_count = df.groupby("target_count").mean(numeric_only=True)
    mean_per_edit_type = df.groupby("edit_type").mean(numeric_only=True)
    mean_per_target_count_edit_type = df.groupby(["target_count", "edit_type"]).mean(
        numeric_only=True
    )
    mean_all = df.mean(numeric_only=True)

    print("Mean per target_count:")
    print(mean_per_target_count)
    print("\nMean per edit_type:")
    print(mean_per_edit_type)
    print("\nMean per target_count and edit_type:")
    print(mean_per_target_count_edit_type)
    print("\nMean for all data:")
    print(mean_all)

    # Save Excel summaries
    mean_per_target_count.to_excel(
        os.path.join(stats_dir, "ale_mean_per_target_count.xlsx"), header=True
    )
    mean_per_edit_type.to_excel(
        os.path.join(stats_dir, "ale_mean_per_edit_type.xlsx"), header=True
    )
    mean_per_target_count_edit_type.to_excel(
        os.path.join(stats_dir, "ale_mean_per_target_count_edit_type.xlsx"),
        header=True,
    )
    mean_all.to_frame().to_excel(
        os.path.join(stats_dir, "ale_mean_all.xlsx"), header=True
    )
