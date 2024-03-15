import torch
from torch import distributed as dist
from torch import nn

from basedxl.modules.base_module import BaseModule
from basedxl.utils import BasedXLConfig


class BasedXLGroupNorm(BaseModule):
    def __init__(self, module: nn.GroupNorm, basedxl_config: BasedXLConfig):
        assert isinstance(module, nn.GroupNorm)
        super(BasedXLGroupNorm, self).__init__(module, basedxl_config)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # type: ignore
        torch.cuda.nvtx.range_push("group_norm::forward")

        module = self.module
        assert isinstance(module, nn.GroupNorm)
        basedxl_config = self.basedxl_config

        if self.comm_manager is not None and self.comm_manager.handles is not None and self.idx is not None:
            if self.comm_manager.handles[self.idx] is not None:
                self.comm_manager.handles[self.idx].wait()  # type: ignore
                self.comm_manager.handles[self.idx] = None

        assert x.ndim == 4
        n, c, h, w = x.shape
        num_groups = module.num_groups
        group_size = c // num_groups

        if self.buffer_list is None and self.comm_manager is not None:
            if self.comm_manager.buffer_list is None:
                n, c, h, w = x.shape
                self.idx = self.comm_manager.register_tensor(
                    shape=[2, n, num_groups, 1, 1, 1], torch_dtype=x.dtype, layer_type="gn"
                )
            else:
                self.buffer_list = self.comm_manager.get_buffer_list(self.idx)
        x = x.view([n, num_groups, group_size, h, w])
        x_mean = x.mean(dim=[2, 3, 4], keepdim=True)  # [1, num_groups, 1, 1, 1]
        x2_mean = (x**2).mean(dim=[2, 3, 4], keepdim=True)  # [1, num_groups, 1, 1, 1]
        slice_mean = torch.stack([x_mean, x2_mean], dim=0)

        if self.buffer_list is None:
            full_mean = slice_mean
        elif self.counter <= basedxl_config.warmup_steps:
            dist.all_gather(self.buffer_list, slice_mean, group=basedxl_config.batch_group, async_op=False)
            full_mean = sum(self.buffer_list) / basedxl_config.n_device_per_batch
        else:
            correction = slice_mean - self.buffer_list[basedxl_config.split_idx()]
            full_mean = sum(self.buffer_list) / basedxl_config.n_device_per_batch + correction
            assert full_mean.shape == slice_mean.shape
            assert self.comm_manager is not None
            self.comm_manager.enqueue(self.idx, slice_mean)

        full_x_mean, full_x2_mean = full_mean[0], full_mean[1]  # type: ignore
        var = full_x2_mean - full_x_mean**2
        slice_x_mean, slice_x2_mean = slice_mean[0], slice_mean[1]
        slice_var = slice_x2_mean - slice_x_mean**2
        var = torch.where(var < 0, slice_var, var)  # Correct negative variance

        num_elements = group_size * h * w
        var = var * (num_elements / (num_elements - 1))
        std = (var + module.eps).sqrt()
        output = (x - full_x_mean) / std
        output = output.view([n, c, h, w])
        if module.affine:
            output = output * module.weight.view([1, -1, 1, 1])
            output = output + module.bias.view([1, -1, 1, 1])
        self.counter += 1
        torch.cuda.nvtx.range_pop()
        return output
