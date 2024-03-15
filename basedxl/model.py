import torch

from diffusers.pipelines.stable_diffusion_xl.pipeline_stable_diffusion_xl import StableDiffusionXLPipeline
from diffusers.models.unets.unet_2d_condition import UNet2DConditionModel

from basedxl.unet import BasedXLUnet
from basedxl.utils import BasedXLConfig


class BasedXLPipeline:
    def __init__(self, pipeline: StableDiffusionXLPipeline, basedxl_config: BasedXLConfig):
        self.pipeline = pipeline
        self.basedxl_config = basedxl_config

    @staticmethod
    def from_pretrained(basedxl_config: BasedXLConfig, **kwargs):
        unet = UNet2DConditionModel.from_pretrained(
            basedxl_config.pretrained_model_name_or_path,
            torch_dtype=basedxl_config.dtype,
            subfolder="unet",
            device_map={"": basedxl_config.device},
        )
        unet = BasedXLUnet(unet, basedxl_config)  # type: ignore

        pipeline = StableDiffusionXLPipeline.from_pretrained(
            basedxl_config.pretrained_model_name_or_path,
            torch_dtype=basedxl_config.dtype,
            unet=unet,
            device_map={"": basedxl_config.device},
            **kwargs,
        )
        pipeline.set_progress_bar_config(disable=basedxl_config.rank != 0)

        return BasedXLPipeline(pipeline, basedxl_config)  # type: ignore

    def set_progress_bar_config(self, **kwargs):
        self.pipeline.set_progress_bar_config(**kwargs)

    @torch.no_grad()
    def __call__(self, *args, **kwargs):
        assert "width" not in kwargs or self.basedxl_config.width == kwargs["width"]
        assert "height" not in kwargs or self.basedxl_config.height == kwargs["height"]
        config = self.basedxl_config
        self.pipeline.unet.set_counter(0)
        return self.pipeline(height=config.height, width=config.width, *args, **kwargs)
