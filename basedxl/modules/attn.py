import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
from diffusers.models.attention_processor import Attention
from diffusers.utils.constants import USE_PEFT_BACKEND

from basedxl.modules.base_module import BaseModule
from basedxl.utils import BasedXLConfig


class BasedXLAttentionPP(BaseModule):
    def __init__(self, module: Attention, basedxl_config: BasedXLConfig):
        super(BasedXLAttentionPP, self).__init__(module, basedxl_config)

        to_k = module.to_k
        to_v = module.to_v
        assert isinstance(to_k, nn.Linear)
        assert isinstance(to_v, nn.Linear)
        assert (to_k.bias is None) == (to_v.bias is None)
        assert to_k.weight.shape == to_v.weight.shape

        in_size, out_size = to_k.in_features, to_k.out_features
        to_kv = nn.Linear(
            in_size,
            out_size * 2,
            bias=to_k.bias is not None,
            device=to_k.weight.device,
            dtype=to_k.weight.dtype,
        )
        to_kv.weight.data[:out_size].copy_(to_k.weight.data)
        to_kv.weight.data[out_size:].copy_(to_v.weight.data)

        if to_k.bias is not None:
            assert to_v.bias is not None
            to_kv.bias.data[:out_size].copy_(to_k.bias.data)
            to_kv.bias.data[out_size:].copy_(to_v.bias.data)

        self.to_kv = to_kv


class BasedXLCrossAttentionPP(BasedXLAttentionPP):
    def __init__(self, module: Attention, basedxl_config: BasedXLConfig):
        super(BasedXLCrossAttentionPP, self).__init__(module, basedxl_config)
        self.kv_cache = None

    def forward(  # type: ignore
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor | None = None,
        scale: float = 1.0,
        *args,
        **kwargs,
    ):
        torch.cuda.nvtx.range_push("cross_attn::forward")
        assert encoder_hidden_states is not None
        recompute_kv = self.counter == 0

        attn = self.module
        assert isinstance(attn, Attention)

        residual = hidden_states

        batch_size, sequence_length, _ = (
            hidden_states.shape if encoder_hidden_states is None else encoder_hidden_states.shape
        )

        args = () if USE_PEFT_BACKEND else (scale,)
        query = attn.to_q(hidden_states, *args)

        if encoder_hidden_states is None:
            encoder_hidden_states = hidden_states

        if recompute_kv or self.kv_cache is None:
            kv = self.to_kv(encoder_hidden_states)
            self.kv_cache = kv
        else:
            kv = self.kv_cache
        key, value = torch.split(kv, kv.shape[-1] // 2, dim=-1)

        inner_dim = key.shape[-1]
        head_dim = inner_dim // attn.heads

        query = query.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        key = key.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        value = value.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)

        hidden_states = F.scaled_dot_product_attention(query, key, value, dropout_p=0.0, is_causal=False)

        hidden_states = hidden_states.transpose(1, 2).reshape(batch_size, -1, attn.heads * head_dim)
        hidden_states = hidden_states.to(query.dtype)

        # linear proj
        hidden_states = attn.to_out[0](hidden_states, *args)
        # dropout
        hidden_states = attn.to_out[1](hidden_states)

        if attn.residual_connection:
            hidden_states = hidden_states + residual

        hidden_states = hidden_states / attn.rescale_output_factor

        self.counter += 1
        torch.cuda.nvtx.range_pop()
        return hidden_states


class BasedXLSelfAttentionPP(BasedXLAttentionPP):
    def __init__(self, module: Attention, basedxl_config: BasedXLConfig):
        super(BasedXLSelfAttentionPP, self).__init__(module, basedxl_config)

    def _forward(self, hidden_states: torch.Tensor, scale: float = 1.0):
        attn = self.module
        basedxl_config = self.basedxl_config
        assert isinstance(attn, Attention)
        assert self.comm_manager is not None

        residual = hidden_states

        batch_size, sequence_length, _ = hidden_states.shape

        args = () if USE_PEFT_BACKEND else (scale,)
        query = attn.to_q(hidden_states, *args)

        encoder_hidden_states = hidden_states

        kv = self.to_kv(encoder_hidden_states)

        if basedxl_config.n_device_per_batch == 1:
            full_kv = kv
        else:
            if self.buffer_list is None:  # buffer not created
                full_kv = torch.cat([kv for _ in range(basedxl_config.n_device_per_batch)], dim=1)
            elif self.counter <= basedxl_config.warmup_steps:
                dist.all_gather(self.buffer_list, kv, group=basedxl_config.batch_group, async_op=False)
                full_kv = torch.cat(self.buffer_list, dim=1)
            else:
                new_buffer_list = [buffer for buffer in self.buffer_list]
                new_buffer_list[basedxl_config.split_idx()] = kv
                full_kv = torch.cat(new_buffer_list, dim=1)
                self.comm_manager.enqueue(self.idx, kv)

        key, value = torch.split(full_kv, full_kv.shape[-1] // 2, dim=-1)

        inner_dim = key.shape[-1]
        head_dim = inner_dim // attn.heads

        query = query.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        key = key.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        value = value.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)

        # the output of sdp = (batch, num_heads, seq_len, head_dim)
        hidden_states = F.scaled_dot_product_attention(query, key, value, dropout_p=0.0, is_causal=False)

        hidden_states = hidden_states.transpose(1, 2).reshape(batch_size, -1, attn.heads * head_dim)
        hidden_states = hidden_states.to(query.dtype)

        # linear proj
        hidden_states = attn.to_out[0](hidden_states, *args)
        # dropout
        hidden_states = attn.to_out[1](hidden_states)

        if attn.residual_connection:
            hidden_states = hidden_states + residual

        hidden_states = hidden_states / attn.rescale_output_factor

        return hidden_states

    def forward(  # type: ignore
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor | None = None,
        scale: float = 1.0,
        *args,
        **kwargs,
    ) -> torch.Tensor:
        torch.cuda.nvtx.range_push("self_attn::forward")
        basedxl_config = self.basedxl_config
        if self.comm_manager is not None and self.comm_manager.handles is not None and self.idx is not None:
            if self.comm_manager.handles[self.idx] is not None:
                self.comm_manager.handles[self.idx].wait()  # type: ignore
                self.comm_manager.handles[self.idx] = None

        b, l, c = hidden_states.shape
        if basedxl_config.n_device_per_batch > 1 and self.buffer_list is None and self.comm_manager is not None:
            if self.comm_manager.buffer_list is None:
                self.idx = self.comm_manager.register_tensor(
                    shape=(b, l, self.to_kv.out_features), torch_dtype=hidden_states.dtype, layer_type="attn"
                )
            else:
                self.buffer_list = self.comm_manager.get_buffer_list(self.idx)
        output = self._forward(hidden_states, scale=scale)

        self.counter += 1
        torch.cuda.nvtx.range_pop()
        return output
