from dataclasses import dataclass

import torch
from torch import distributed as dist
from torch._C._distributed_c10d import Work, ProcessGroup  # type: ignore


@dataclass
class BasedXLConfig:
    pretrained_model_name_or_path: str = "stabilityai/stable-diffusion-xl-base-1.0"
    dtype: torch.dtype = torch.float16
    compile_unet: bool = False
    width: int = 1024
    height: int = 1024
    use_cfg: bool = True
    warmup_steps: int = 4
    comm_checkpoint: int = 60
    verbose: bool = False

    def __post_init__(self):
        try:
            dist.init_process_group(backend="nccl")
            self.rank = dist.get_rank()
            self.world_size = dist.get_world_size()
            self.device = f"cuda:{self.rank}"
        except Exception:
            print("Failed to initialize distributed process group. Running in single-device mode.")
            self.rank = 0
            self.world_size = 1
            self.device = "cuda"

        if self.use_cfg:
            n_device_per_batch = self.world_size // 2
            if n_device_per_batch == 0:
                n_device_per_batch = 1
        else:
            n_device_per_batch = self.world_size

        self.n_device_per_batch = n_device_per_batch

        batch_group: ProcessGroup | None = None
        split_group: ProcessGroup | None = None

        if self.use_cfg and self.world_size >= 2:
            batch_groups: list[ProcessGroup] = []
            for i in range(2):
                start, end = i * (self.world_size // 2), (i + 1) * (self.world_size // 2)
                batch_groups.append(dist.new_group(list(range(start, end))))  # type: ignore
            batch_group = batch_groups[self.batch_idx()]

            split_groups: list[ProcessGroup] = []
            for i in range(self.world_size // 2):
                split_groups.append(dist.new_group([i, i + self.world_size // 2]))  # type: ignore
            split_group = split_groups[self.split_idx()]

        self.batch_group = batch_group
        self.split_group = split_group

    def batch_idx(self, rank: int | None = None) -> int:
        if rank is None:
            rank = self.rank
        if self.use_cfg:
            return 1 - int(rank < (self.world_size // 2))
        else:
            return 0  # raise NotImplementedError

    def split_idx(self, rank: int | None = None) -> int:
        if rank is None:
            rank = self.rank
        return rank % self.n_device_per_batch


class PatchParallelismCommManager:
    def __init__(self, basedxl_config: BasedXLConfig):
        self.basedxl_config = basedxl_config

        self.torch_dtype = None
        self.numel = 0
        self.numel_dict = {}

        self.buffer_list: list[torch.Tensor] | None = None

        self.starts = []
        self.ends = []
        self.shapes = []

        self.idx_queue = []

        self.handles: list[Work | None] | None = None

    def register_tensor(
        self, shape: tuple[int, ...] | list[int], torch_dtype: torch.dtype, layer_type: str | None = None
    ) -> int:
        if self.torch_dtype is None:
            self.torch_dtype = torch_dtype
        else:
            assert self.torch_dtype == torch_dtype
        self.starts.append(self.numel)
        numel = 1
        for dim in shape:
            numel *= dim
        self.numel += numel
        if layer_type is not None:
            if layer_type not in self.numel_dict:
                self.numel_dict[layer_type] = 0
            self.numel_dict[layer_type] += numel

        self.ends.append(self.numel)
        self.shapes.append(shape)
        return len(self.starts) - 1

    def create_buffer(self):
        basedxl_config = self.basedxl_config
        if basedxl_config.rank == 0 and basedxl_config.verbose:
            print(
                f"Create buffer with {self.numel / 1e6:.3f}M parameters for {len(self.starts)} tensors on each device."
            )
            for layer_type, numel in self.numel_dict.items():
                print(f"  {layer_type}: {numel / 1e6:.3f}M parameters")

        self.buffer_list = [
            torch.empty(self.numel, dtype=self.torch_dtype, device=self.basedxl_config.device)
            for _ in range(self.basedxl_config.n_device_per_batch)
        ]
        self.handles = [None for _ in range(len(self.starts))]

    def get_buffer_list(self, idx: int) -> list[torch.Tensor]:
        assert self.buffer_list is not None
        buffer_list = [t[self.starts[idx] : self.ends[idx]].view(self.shapes[idx]) for t in self.buffer_list]
        return buffer_list

    def communicate(self):
        assert self.buffer_list is not None
        assert self.handles is not None

        config = self.basedxl_config
        start = self.starts[self.idx_queue[0]]
        end = self.ends[self.idx_queue[-1]]
        tensor = self.buffer_list[config.split_idx()][start:end]
        buffer_list = [t[start:end] for t in self.buffer_list]
        handle = dist.all_gather(buffer_list, tensor, group=config.batch_group, async_op=True)
        for i in self.idx_queue:
            self.handles[i] = handle
        self.idx_queue = []

    def enqueue(self, idx: int, tensor: torch.Tensor):
        assert self.buffer_list is not None
        config = self.basedxl_config
        if idx == 0 and len(self.idx_queue) > 0:
            self.communicate()
        assert len(self.idx_queue) == 0 or self.idx_queue[-1] == idx - 1
        self.idx_queue.append(idx)
        self.buffer_list[config.split_idx()][self.starts[idx] : self.ends[idx]].copy_(tensor.flatten())

        if len(self.idx_queue) == config.comm_checkpoint:
            self.communicate()

    def clear(self):
        if len(self.idx_queue) > 0:
            self.communicate()

        if self.handles is not None:
            for i in range(len(self.handles)):
                if self.handles[i] is not None:
                    self.handles[i].wait()  # type: ignore
                    self.handles[i] = None
