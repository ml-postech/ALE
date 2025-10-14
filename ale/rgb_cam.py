# Copyright 2022 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# based on ptp_utils.py
# https://github.com/google/prompt-to-prompt/blob/9c472e44aa1b607da59fea94820f7be9480ec545/ptp_utils.py

import torch
import torchvision


class AttnProcessor:
    """
    A custom attention processor that hijacks the forward pass of an attention layer.
    This is the core class that implements the RGB-CAM logic.
    """

    def __init__(
        self,
        controller,
        place_in_unet,
        layer_num,
        prompt_embeds,
        mask_list,
        raw_masks,
    ):
        """
        Args:
            place_in_unet (str): The location of the current attention layer in the U-Net ('down', 'mid', 'up').
            layer_num (int): The layer number.
        """
        self.controller = controller
        self.place_in_unet = place_in_unet
        self.replace_value = None
        self.prompt_embeds = prompt_embeds
        self.attention_mask_list = mask_list
        self.raw_masks = raw_masks
        self.current_step = 0
        self.layer_num = layer_num

    def __call__(
        self,
        attn,
        hidden_states,
        encoder_hidden_states=None,
        attention_mask=None,
        temb=None,
        scale=1.0,
    ):
        # The 'Attention' class in diffusers will execute this __call__ method.
        residual = hidden_states
        if attn.spatial_norm is not None:
            hidden_states = attn.spatial_norm(hidden_states, temb)

        input_ndim = hidden_states.ndim

        if input_ndim == 4:
            batch_size, channel, height, width = hidden_states.shape
            hidden_states = hidden_states.view(
                batch_size, channel, height * width
            ).transpose(1, 2)

        is_cross = encoder_hidden_states is not None
        if encoder_hidden_states is None:
            encoder_hidden_states = hidden_states
        elif attn.norm_cross:
            encoder_hidden_states = attn.norm_encoder_hidden_states(
                encoder_hidden_states
            )

        # sequence_length is 77 for cross-attention, (latent_dim)^2 for self-attention.
        batch_size, sequence_length, _ = (
            hidden_states.shape
            if encoder_hidden_states is None
            else encoder_hidden_states.shape
        )

        # initialization for RGB-CAM
        if is_cross and (self.attention_mask_list is not None):
            # To prevent redundant calculations, prepare masks only at the first layer of each step.
            if self.current_step == 0:
                # Resize the mask to match the current layer's latent size.
                size = int(hidden_states.shape[1] ** 0.5)

                resize = torchvision.transforms.Resize(
                    size=(size, size),
                    interpolation=torchvision.transforms.InterpolationMode.NEAREST,
                    antialias=False,
                )

                masks = resize(self.attention_mask_list.clone())
                masks = masks.reshape(masks.shape[0], -1)
                masks = torch.transpose(masks, 0, 1)

                # Convert mask values to be added to the attention scores.
                # Masked positions get a large negative value (-100), others get 0.
                ATTENTION_MASKED_VALUE = -100
                ATTENTION_UNMASKED_VALUE = 0
                masks = torch.where(
                    masks < 0.5, ATTENTION_MASKED_VALUE, ATTENTION_UNMASKED_VALUE
                )
                masks = masks.to(hidden_states.dtype)

                # The pipeline processes a batch of 4:
                # (source_uncond, target_uncond, source_cond, target_cond).
                # We only want to apply the mask to the 4th item, 'target_cond'.
                # Therefore, we create a stack of masks in the format [0, 0, 0, mask] for the batch of 4.
                attention_masks = torch.stack(
                    [torch.zeros_like(masks)] * 3 + [masks] * 1,
                    dim=0,
                )

                # Also resize the raw masks, which will be used in the blending step later.
                # objects + background
                raw_masks = resize(self.raw_masks)

                # If masks overlap, prioritize the background mask (the last mask) to avoid conflicts.
                temp = raw_masks[:-1].to(hidden_states.dtype)
                temp = temp.sum(dim=0, keepdim=True)
                raw_masks[:-1] = torch.where(temp == 1, raw_masks[:-1], False)
                raw_masks[-1] = torch.where(temp[0] > 1, True, raw_masks[-1])

                # Reshape the raw masks for the final attention value blending.
                prepared_raw_masks = raw_masks.reshape(
                    raw_masks.shape[0],
                    1,
                    raw_masks.shape[1] * raw_masks.shape[2],
                    1,
                )

                # Prepare the cross-attention masks.
                prepared_attention_masks = attn.prepare_attention_mask(
                    attention_masks,
                    sequence_length - attention_masks.shape[2],
                    batch_size,
                )

                # Cache the prepared masks in the current instance for reuse in subsequent layers.
                self.rgb_masks = prepared_raw_masks
                self.cross_attention_masks = prepared_attention_masks

        else:  # For self-attention or if no mask is provided, use the default mask (None).
            self.cross_attention_masks = attn.prepare_attention_mask(
                None,
                sequence_length,
                batch_size,
            )

        # Calculate Query, Key, Value.
        q = attn.to_q(hidden_states)
        k = attn.to_k(encoder_hidden_states)
        v = attn.to_v(encoder_hidden_states)

        q = attn.head_to_batch_dim(q)
        k = attn.head_to_batch_dim(k)
        v = attn.head_to_batch_dim(v)

        # self-attention injection
        if not is_cross:
            q, k, v = self.controller.self_attn_forward(q, k, v, attn.heads)

        # Calculate attention scores (prepared_attention_masks is applied here).
        attention_probs = attn.get_attention_scores(q, k, self.cross_attention_masks)

        # RGB-CAM
        if is_cross:
            attention_probs = self.controller(
                attention_probs, is_cross, self.place_in_unet
            )
            # Standard attention calculation.
            hidden_states = torch.bmm(attention_probs, v)

            # If masks are provided, perform "Region-guided Blending".
            # Recalculate hidden_states using OREs (prompt_embeds)
            # and raw spatial masks (`prepared_raw_masks`).
            # This guides each spatial region to be influenced
            # only by its corresponding text embedding.
            if self.rgb_masks is not None:
                # Isolate the part of the batch corresponding to the target_cond branch.
                blended_hidden_states = torch.zeros_like(hidden_states[24:32])
                # Extract the attention probabilities for the target_cond branch.
                target_cond_attention_probs = attention_probs[24:32]

                # Iterate over each object (prompt embedding) and its corresponding mask.
                for prompt_embed, mask in zip(self.prompt_embeds, self.rgb_masks):
                    # Recalculate the Value vector 'v' using the object-specific embedding.
                    v = attn.to_v(prompt_embed)
                    v = attn.head_to_batch_dim(v)

                    # Calculate attention output for this specific object.
                    object_hidden_states = torch.bmm(target_cond_attention_probs, v)

                    # Apply the object's spatial mask and accumulate (blend) the results.
                    blended_hidden_states += object_hidden_states * mask

                # Replace the target_cond part of the original hidden_states with the newly calculated values.
                hidden_states[24:32] = blended_hidden_states

            hidden_states = attn.batch_to_head_dim(hidden_states)
        else:  # self-attention
            hidden_states = torch.bmm(attention_probs, v)
            hidden_states = attn.batch_to_head_dim(hidden_states)

        # Linear projection and dropout.
        hidden_states = attn.to_out[0](hidden_states)
        hidden_states = attn.to_out[1](hidden_states)

        if input_ndim == 4:
            hidden_states = hidden_states.transpose(-1, -2).reshape(
                batch_size, channel, height, width
            )

        # Residual connection.
        if attn.residual_connection:
            hidden_states = hidden_states + residual

        hidden_states = hidden_states / attn.rescale_output_factor

        self.current_step += 1
        return hidden_states


def register_attention_control_RGB(
    model,
    controller,
    prompt_embeds,
    raw_masks=None,
    mask_list=None,
):
    """
    Registers a custom attention processor (`AttnProcessor`) into all attention layers of the U-Net.
    This enables the control of attention maps during the diffusion process to implement
    Region-guided blending for cross-attention masking (RGB-CAM).

    Args:
        model: The diffusion pipeline containing the U-Net model.
        controller: An attention control object (e.g., AttentionRefine) that holds the main logic.
        prompt_embeds: Text embeddings that are separated for each object to be edited (ORE).
        raw_masks: The original binary masks generated by SAM.
        mask_list: The attention masks prepared for the RGB-CAM mechanism.
    """

    # A recursive helper function to traverse U-Net sub-networks and register the AttnProcessor.
    def register_recr(net_, count, place_in_unet):
        for idx, m in enumerate(net_.modules()):
            if m.__class__.__name__ == "Attention":
                m.processor = AttnProcessor(
                    controller,
                    place_in_unet,
                    count,
                    prompt_embeds,
                    mask_list,
                    raw_masks,
                )
                count += 1
        return count

    cross_att_count = 0
    sub_nets = model.unet.named_children()
    for net in sub_nets:
        if "down" in net[0]:
            cross_att_count += register_recr(net[1], 0, "down")
        elif "up" in net[0]:
            cross_att_count += register_recr(net[1], 0, "up")
        elif "mid" in net[0]:
            cross_att_count += register_recr(net[1], 0, "mid")

    # Inform the controller about the total number of attention layers.
    controller.num_att_layers = cross_att_count
