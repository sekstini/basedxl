import torch
import torch._dynamo.config
import torch._inductor.config

import fire
from diffusers.schedulers.scheduling_euler_ancestral_discrete import EulerAncestralDiscreteScheduler

from basedxl import BasedXLPipeline
from basedxl.utils import BasedXLConfig


torch._inductor.config.coordinate_descent_tuning = True
torch._inductor.config.triton.unique_kernel_names = False
torch._inductor.config.fx_graph_cache = True
torch._inductor.config.conv_1x1_as_mm = True
torch._inductor.config.freezing = True
torch._inductor.config.freezing_discard_parameters = True


def main(
    prompt: str = "Photo of a lychee-inspired spherical chair, with a bumpy white exterior and plush interior, set against a tropical wallpaper.",
    negative_prompt: str = "",
    seed: int = 42,
    guidance_scale: float = 5.0,
    num_inference_steps: int = 50,
    width: int = 1024,
    height: int = 1024,
    compile_unet: bool = False,
    fp16_acc_matmul: bool = False,
    patch_parallelism: bool = True,
    distrifusion_warmup_steps: int = 4,
    out_path: str = "sample.png",
    verbose: bool = False,
):
    assert not (compile_unet and fp16_acc_matmul), "torch.compile currently breaks on user defined triton kernels."

    basedxl_config = BasedXLConfig(
        width=width,
        height=height,
        compile_unet=compile_unet,
        fp16_acc_matmul=fp16_acc_matmul,
        patch_parallelism=patch_parallelism,
        warmup_steps=distrifusion_warmup_steps,
        verbose=verbose,
    )

    scheduler = EulerAncestralDiscreteScheduler.from_pretrained(
        basedxl_config.pretrained_model_name_or_path,
        subfolder="scheduler",
    )

    pipeline = BasedXLPipeline.from_pretrained(
        basedxl_config,
        variant="fp16",
        use_safetensors=True,
        scheduler=scheduler,
    )

    # Warmup
    pipeline(
        prompt=prompt,
        negative_prompt=negative_prompt,
        guidance_scale=guidance_scale,
        num_inference_steps=10,
    )

    # Generate
    torch.cuda.cudart().cudaProfilerStart()  # type: ignore
    output = pipeline(
        prompt=prompt,
        negative_prompt=negative_prompt,
        generator=torch.Generator(device="cuda").manual_seed(seed),
        num_inference_steps=num_inference_steps,
        guidance_scale=guidance_scale,
    )
    torch.cuda.cudart().cudaProfilerStop()  # type: ignore

    image = output.images[0]  # type: ignore
    image.save(out_path)


if __name__ == "__main__":
    fire.Fire(main)
