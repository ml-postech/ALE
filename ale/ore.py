import torch

from .config import DEVICE


def get_single_prompt_embeds(prompt, pipe):
    embeds_max_length = pipe.tokenizer.model_max_length
    text_inputs_id = pipe.tokenizer(
        prompt,
        padding="max_length",
        max_length=embeds_max_length,
    ).input_ids

    prompt_embeds = pipe.text_encoder(
        torch.tensor(text_inputs_id).to(DEVICE).unsqueeze(dim=0)
    )[0]

    return prompt_embeds


def find_sublist_indices(A, B):
    # Get the length of A and B
    len_a, len_b = len(A), len(B)
    indices = []

    # Iterate over A and check for a matching sublist
    for i in range(len_a - len_b + 1):
        if A[i : i + len_b] == B:
            indices = list(range(i, i + len_b))
            break

    return indices


def get_substituted_base_embeds(base_prompt, target_noun_list, pipe):
    base_prompt_embeds = get_single_prompt_embeds(base_prompt, pipe=pipe)
    base_ids = pipe.tokenizer(base_prompt).input_ids
    for noun_chunk in target_noun_list:
        target_ids = pipe.tokenizer(noun_chunk).input_ids[1:-1]
        indices = find_sublist_indices(base_ids, target_ids)
        base_prompt_embeds[0, indices, :] = get_single_prompt_embeds(
            noun_chunk, pipe=pipe
        )[0, indices, :]
    return base_prompt_embeds


def get_object_restricted_embeds(base_prompt, target_noun_list, pipe):
    EOS_modified_embeds = []
    prompt_embeds = get_substituted_base_embeds(
        base_prompt, target_noun_list, pipe=pipe
    )
    base_ids = pipe.tokenizer(base_prompt).input_ids
    base_EOS_start = len(base_ids) - 1
    for noun_chunk in target_noun_list:
        prompt_embed = prompt_embeds.clone()
        target_ids = pipe.tokenizer(noun_chunk).input_ids[1:-1]
        indices = find_sublist_indices(base_ids, target_ids)

        target_EOS_start = len(target_ids) + 1
        target_embed = get_single_prompt_embeds(noun_chunk, pipe=pipe)

        # remove other objects
        prompt_embed[0, 1:base_EOS_start] = 0

        # substitute target object again?
        prompt_embed[0, indices] = target_embed[0, 1:target_EOS_start]
        # substitute EOS
        prompt_embed[0, base_EOS_start:] = target_embed[
            0, target_EOS_start : 77 - base_EOS_start + target_EOS_start
        ]

        EOS_modified_embeds.append(prompt_embed)

    prompt_embeds = EOS_modified_embeds + [prompt_embeds]
    return prompt_embeds
