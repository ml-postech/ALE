import torch
import torchvision
import abc
from typing import Optional, Union, Tuple


device = "cuda"
torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32

# Background Blending and self-attention injection


class BackgroundBlend:
    def __init__(
        self,
        save_inter=True,
    ):
        self.save_inter = save_inter
        self.background_mask = None

    def __call__(
        self,
        x_s,
        x_t,
    ):
        if self.background_mask is not None:
            size = x_t.shape[-1]
            resize = torchvision.transforms.Resize(
                size=(size, size),
                interpolation=torchvision.transforms.InterpolationMode.NEAREST,
                antialias=False,
            )

            foreground_mask = (
                resize(~self.background_mask[None, None, :, :]).to(device).to(x_t.dtype)
            )
            for mult in [2, 4, 8]:
                _size = size // mult
                resize_temp = torchvision.transforms.Resize(
                    size=(_size, _size),
                    interpolation=torchvision.transforms.InterpolationMode.NEAREST,
                    antialias=False,
                )

                mask_temp = (
                    resize_temp(~self.background_mask[None, None, :, :])
                    .to(device)
                    .to(x_t.dtype)
                )
                mask_temp = resize(mask_temp)

                foreground_mask += mask_temp
            foreground_mask = foreground_mask > 0
            background_mask = ~foreground_mask
            # hard masking
            x_t_out = torch.where(background_mask, x_s, x_t)
            return x_t_out
        return x_t


class AttentionControl(abc.ABC):
    def step_callback(self, x_t):
        return x_t

    @property
    def num_uncond_att_layers(self):
        return 0

    @abc.abstractmethod
    def forward(self, attn, is_cross: bool, place_in_unet: str):
        raise NotImplementedError

    def __call__(self, attn, is_cross: bool, place_in_unet: str):
        if self.cur_att_layer >= self.num_uncond_att_layers:
            attn[2 * self.num_heads : 4 * self.num_heads] = self.forward(
                attn[2 * self.num_heads : 4 * self.num_heads],
                is_cross,
                place_in_unet,
            )
        self.cur_att_layer += 1
        if self.cur_att_layer == self.num_att_layers // 2 + self.num_uncond_att_layers:
            self.cur_att_layer = 0
            self.cur_step += 1
        return attn

    def reset(self):
        self.cur_step = 0
        self.cur_att_layer = 0

    def __init__(self, num_heads):
        self.cur_step = 0
        self.num_att_layers = -1
        self.cur_att_layer = 0
        self.num_heads = num_heads


class AttentionControlEdit(AttentionControl, abc.ABC):
    def step_callback(self, i, t, x_s, x_t, alpha_prod):
        if (self.background_blend is not None) and (i > 0):
            x_t = self.background_blend(x_s, x_t)
        return x_t

    def replace_self_attention(self, attn_base, att_replace):
        if att_replace.shape[2] <= 16**2:
            return attn_base.unsqueeze(0).expand(att_replace.shape[0], *attn_base.shape)
        else:
            return att_replace

    def forward(self, attn, is_cross: bool, place_in_unet: str):
        return attn

    def self_attn_forward(self, q, k, v, num_heads):
        (qu, qc) = (
            q[0 * num_heads : 2 * num_heads],
            q[2 * num_heads : 4 * num_heads],
        )
        (ku, kc) = (
            k[0 * num_heads : 2 * num_heads],
            k[2 * num_heads : 4 * num_heads],
        )
        (vu, vc) = (
            v[0 * num_heads : 2 * num_heads],
            v[2 * num_heads : 4 * num_heads],
        )

        if self.self_replace_steps <= (
            (self.cur_step + self.start_steps + 1) * 1.0 / self.num_steps
        ):
            # no injection
            qu = torch.cat([qu[: num_heads * 2]])
            qc = torch.cat([qc[: num_heads * 2]])
            ku = torch.cat([ku[: num_heads * 2]])
            kc = torch.cat([kc[: num_heads * 2]])
            vu = torch.cat([vu[: num_heads * 2]])
            vc = torch.cat([vc[: num_heads * 2]])

        else:
            # inject Q, K
            qu = torch.cat([qu[:num_heads], qu[:num_heads]])
            qc = torch.cat([qc[:num_heads], qc[:num_heads]])
            ku = torch.cat([ku[:num_heads], ku[:num_heads]])
            kc = torch.cat([kc[:num_heads], kc[:num_heads]])
            vu = torch.cat([vu[: num_heads * 2]])
            vc = torch.cat([vc[: num_heads * 2]])

        return (
            torch.cat([qu, qc], dim=0),
            torch.cat([ku, kc], dim=0),
            torch.cat([vu, vc], dim=0),
        )

    def __init__(
        self,
        num_steps: int,
        start_steps: int,
        self_replace_steps: Union[float, Tuple[float, float]],
        background_blend: Optional[BackgroundBlend],
        num_heads: int,
    ):
        super(AttentionControlEdit, self).__init__(num_heads)
        self.self_replace_steps = self_replace_steps
        self.num_steps = num_steps
        self.start_steps = start_steps
        self.background_blend = background_blend


class AttentionRefine(AttentionControlEdit):
    def self_attn_forward(self, q, k, v, num_heads):
        return super(AttentionRefine, self).self_attn_forward(q, k, v, num_heads)

    def __init__(
        self,
        num_steps: int,
        start_steps: int,
        self_replace_steps: float,
        background_blend: Optional[BackgroundBlend] = None,
        num_heads=8,
    ):
        super(AttentionRefine, self).__init__(
            num_steps,
            start_steps,
            self_replace_steps,
            background_blend,
            num_heads,
        )
