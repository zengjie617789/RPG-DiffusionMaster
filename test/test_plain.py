from diffusers.schedulers import KarrasDiffusionSchedulers,DPMSolverMultistepScheduler
from diffusers import StableDiffusionXLPipeline
from mllm import local_llm,GPT4
import torch
from utils.common import save_path
from pathlib import  Path


ckpt_path = "/data/zzzj/models/stable-diffusion-xl-base-1.0"
pipe = StableDiffusionXLPipeline.from_pretrained(ckpt_path,torch_dtype=torch.float16, use_safetensors=True, variant="fp16").to("cuda")
pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config,use_karras_sigmas=True)

negative_prompt = "" # negative_prompt, 


# split_ratio = '0.5,0.5,0.5; 0.5,0.5,0.5'
# prompt = "This comic is a black and white illustration that "
prompt = "empty"

lora_path = "/data/zzzj/models/kohya_ss_output/sdxl_lora_prodigy_comics_wjs_upscale/sdxl_lora_prodigy_comics_wjs_upscale.safetensors"


# pipe.unet = torch.compile(pipe.unet)

pipe.load_lora_weights(Path(lora_path).parent, weight_name=Path(lora_path).name)
pipe.fuse_lora()

num = 4
prompt = [prompt] * num
for i in range(num):
    images = pipe(
    prompt=prompt,
    num_inference_steps=20, # sampling step
    height = 1024, 
    negative_prompt=negative_prompt, # negative prompt
    width = 1024, 
    seed = 1234567,# random seed
    guidance_scale = 7.0
    ).images
    save_path(images)
# images.save("test.png")