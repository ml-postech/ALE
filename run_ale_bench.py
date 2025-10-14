import os
from PIL import Image
import json
import argparse

from ale.edit import ale_edit, load_edit_pipeline


def get_hyps(editing_types):
    if editing_types == ["color"]:
        return 1.0
    elif editing_types == ["object"]:
        return 0.5
    elif editing_types == ["material"]:
        return 0.6
    elif editing_types == ["color", "object"]:
        return 0.5
    elif editing_types == ["object", "material"]:
        return 0.5
    print("editing types are weird...")
    return 1.0


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--prompts_base_dir",
        type=str,
        default="./ale_bench/prompts",
        help="Directory containing prompt json files",
    )
    parser.add_argument(
        "--images_dir",
        type=str,
        default="./ale_bench/images",
        help="Directory containing source images",
    )
    parser.add_argument(
        "--results_dir",
        type=str,
        default="./results",
        help="Directory to save edited results",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility",
    )
    parser.add_argument(
        "--object_counts",
        nargs="+",
        type=int,
        default=[1, 2, 3],
        help="List of number of objects to edit",
    )
    parser.add_argument(
        "--types",
        nargs="+",
        type=str,
        default=["color", "object", "material", "color+object", "object+material"],
        help="Types of edits to perform",
    )

    return parser.parse_args()


def main():
    args = parse_args()
    total_edit = 0
    pipe = load_edit_pipeline()

    for object_count in args.object_counts:
        for edit_type in args.types:
            prompt_dir = os.path.join(
                args.prompts_base_dir, f"{object_count}", edit_type
            )
            for filename in os.listdir(prompt_dir):
                print(prompt_dir, filename)
                prompt_file = os.path.join(prompt_dir, filename)
                with open(prompt_file, "r") as f:
                    data = json.load(f)

                image_filename = data["image"] + ".jpg"
                image_path = os.path.join(args.images_dir, image_filename)
                image_source = Image.open(image_path).convert("RGB")

                prompt_list = data["prompts"]

                out_dir = os.path.join(
                    args.results_dir, f"{object_count}", edit_type, data["image"]
                )
                os.makedirs(out_dir, exist_ok=True)

                prompt_copy = f"{out_dir}/prompt.json"
                with open(prompt_copy, "w") as f:
                    json.dump(data, f, indent=4)

                for i, prompt in enumerate(prompt_list):
                    editing_type = prompt["editing_types"]
                    self_replace_steps = get_hyps(editing_type)
                    edits = prompt["edits"]
                    edits.extend([["", ""]] * (3 - object_count))
                    [
                        [source_prompt, target_prompt],
                        [source_prompt2, target_prompt2],
                        [source_prompt3, target_prompt3],
                    ] = edits

                    image_out = ale_edit(
                        img=image_source,
                        source_prompt=source_prompt,
                        target_prompt=target_prompt,
                        source_prompt2=source_prompt2,
                        target_prompt2=target_prompt2,
                        source_prompt3=source_prompt3,
                        target_prompt3=target_prompt3,
                        seed=args.seed,
                        self_replace_steps=self_replace_steps,
                        pipe=pipe,
                    )

                    out_file = os.path.join(out_dir, f"edit_{i}.png")
                    image_out.save(out_file)
                    total_edit += 1

    print(f"{total_edit} editing done...")


if __name__ == "__main__":
    main()
