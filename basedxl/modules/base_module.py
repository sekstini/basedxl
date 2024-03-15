from torch import nn

from basedxl.utils import BasedXLConfig, PatchParallelismCommManager


class BaseModule(nn.Module):
    def __init__(
        self,
        module: nn.Module,
        basedxl_config: BasedXLConfig,
    ):
        super(BaseModule, self).__init__()
        self.module = module
        self.basedxl_config = basedxl_config
        self.comm_manager: PatchParallelismCommManager | None = None

        self.counter = 0

        self.buffer_list = None
        self.idx = None

    def forward(self, *args, **kwargs):
        raise NotImplementedError

    def set_counter(self, counter: int = 0):
        self.counter = counter

    def set_comm_manager(self, comm_manager: PatchParallelismCommManager):
        self.comm_manager = comm_manager
