import torch
import torch.nn as nn
import torch.distributed as dist

from diffusers.models.unets.unet_2d_condition import UNet2DConditionModel
from diffusers.models.modeling_utils import ModelMixin
from diffusers.configuration_utils import ConfigMixin
from diffusers.models.attention_processor import Attention


from basedxl.modules.attn import BasedXLCrossAttentionPP, BasedXLSelfAttentionPP
from basedxl.modules.base_module import BaseModule
from basedxl.modules.conv2d import BasedConv2dPP
from basedxl.modules.groupnorm import BasedXLGroupNorm
from basedxl.utils import BasedXLConfig, PatchParallelismCommManager


class BaseModel(ModelMixin, ConfigMixin):
    def __init__(self, model: nn.Module, basedxl_config: BasedXLConfig):
        super(BaseModel, self).__init__()
        self.model = model
        self.basedxl_config = basedxl_config
        self.comm_manager = None

        self.buffer_list = None
        self.output_buffer = None
        self.counter = 0

        # for cuda graph
        self.static_inputs = None
        self.static_outputs = None
        self.cuda_graphs = None

    def forward(self, *args, **kwargs):
        raise NotImplementedError

    def set_counter(self, counter: int = 0):
        self.counter = counter
        for module in self.model.modules():
            if isinstance(module, BaseModule):
                module.set_counter(counter)

    def set_comm_manager(self, comm_manager: PatchParallelismCommManager):
        self.comm_manager = comm_manager
        for module in self.model.modules():
            if isinstance(module, BaseModule):
                module.set_comm_manager(comm_manager)

    def setup_cuda_graph(self, static_outputs, cuda_graphs):
        self.static_outputs = static_outputs
        self.cuda_graphs = cuda_graphs

    @property
    def config(self):
        return self.model.config

    def synchronize(self):
        if self.comm_manager is not None and self.comm_manager.handles is not None:
            for i in range(len(self.comm_manager.handles)):
                if self.comm_manager.handles[i] is not None:
                    self.comm_manager.handles[i].wait()  # type: ignore
                    self.comm_manager.handles[i] = None


class BasedXLUnet(BaseModel):
    def __init__(self, model: UNet2DConditionModel, basedxl_config: BasedXLConfig):
        assert isinstance(model, UNet2DConditionModel)

        self._apply_patch_parallel_layers(model, basedxl_config)

        if basedxl_config.compile_unet:
            model.to(memory_format=torch.channels_last)  # type: ignore
            model.compile(mode="reduce-overhead", fullgraph=True)

        super(BasedXLUnet, self).__init__(model, basedxl_config)

    def _apply_patch_parallel_layers(self, model: UNet2DConditionModel, basedxl_config: BasedXLConfig):
        if basedxl_config.world_size > 1 and basedxl_config.n_device_per_batch > 1:
            for name, module in model.named_modules():
                if isinstance(module, BaseModule):
                    continue
                for subname, submodule in module.named_children():
                    if isinstance(submodule, nn.Conv2d):
                        kernel_size = submodule.kernel_size
                        if kernel_size == (1, 1) or kernel_size == 1:
                            continue
                        wrapped_submodule = BasedConv2dPP(
                            submodule, basedxl_config, is_first_layer=subname == "conv_in"
                        )
                        setattr(module, subname, wrapped_submodule)
                    elif isinstance(submodule, Attention):
                        if subname == "attn1":  # self attention
                            wrapped_submodule = BasedXLSelfAttentionPP(submodule, basedxl_config)
                        else:  # cross attention
                            assert subname == "attn2"
                            wrapped_submodule = BasedXLCrossAttentionPP(submodule, basedxl_config)
                        setattr(module, subname, wrapped_submodule)
                    elif isinstance(submodule, nn.GroupNorm):
                        wrapped_submodule = BasedXLGroupNorm(submodule, basedxl_config)
                        setattr(module, subname, wrapped_submodule)

    @property
    def add_embedding(self):
        return self.model.add_embedding

    def forward(  # type: ignore
        self,
        sample: torch.Tensor,
        timestep: torch.Tensor | float | int,
        encoder_hidden_states: torch.Tensor,
        added_cond_kwargs: dict[str, torch.Tensor] = {},
        **kwargs,
    ) -> tuple[torch.Tensor]:
        torch.cuda.nvtx.range_push("unet::forward")
        config = self.basedxl_config
        b, c, h, w = sample.shape

        if config.world_size == 1:
            output = self.model(
                sample,
                timestep,
                encoder_hidden_states,
                added_cond_kwargs=added_cond_kwargs,
                return_dict=False,
            )[0]
        elif config.use_cfg:
            torch.cuda.nvtx.range_push("unet cfg")
            assert b == 2
            batch_idx = config.batch_idx()
            sample = sample[batch_idx : batch_idx + 1]
            timestep = (
                timestep[batch_idx : batch_idx + 1]
                if isinstance(timestep, torch.Tensor) and timestep.ndim > 0
                else timestep
            )
            encoder_hidden_states = encoder_hidden_states[batch_idx : batch_idx + 1]
            new_added_cond_kwargs = {}

            for k in added_cond_kwargs:
                new_added_cond_kwargs[k] = added_cond_kwargs[k][batch_idx : batch_idx + 1]

            added_cond_kwargs = new_added_cond_kwargs

            output = self.model(
                sample,
                timestep,
                encoder_hidden_states,
                added_cond_kwargs=added_cond_kwargs,
                return_dict=False,
            )[0]

            if self.output_buffer is None:
                self.output_buffer = torch.empty((b, c, h, w), device=output.device, dtype=output.dtype)

            if self.buffer_list is None:
                self.buffer_list = [torch.empty_like(output) for _ in range(config.world_size)]

            dist.all_gather(self.buffer_list, output.contiguous(), async_op=False)
            torch.cat(self.buffer_list[: config.n_device_per_batch], dim=2, out=self.output_buffer[0:1])
            torch.cat(self.buffer_list[config.n_device_per_batch :], dim=2, out=self.output_buffer[1:2])
            output = self.output_buffer
            torch.cuda.nvtx.range_pop()
        else:
            output = self.model(
                sample,
                timestep,
                encoder_hidden_states,
                added_cond_kwargs=added_cond_kwargs,
                return_dict=False,
            )[0]

            if self.output_buffer is None:
                self.output_buffer = torch.empty((b, c, h, w), device=output.device, dtype=output.dtype)

            if self.buffer_list is None:
                self.buffer_list = [torch.empty_like(output) for _ in range(config.world_size)]

            output = output.contiguous()
            dist.all_gather(self.buffer_list, output, async_op=False)
            torch.cat(self.buffer_list, dim=2, out=self.output_buffer)
            output = self.output_buffer

        self.counter += 1
        torch.cuda.nvtx.range_pop()
        return (output,)
