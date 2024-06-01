from RegionalDiffusion_base import RegionalDiffusionPipeline
from RegionalDiffusion_xl import RegionalDiffusionXLPipeline
from diffusers.schedulers import KarrasDiffusionSchedulers,DPMSolverMultistepScheduler
from mllm import local_llm,GPT4
import torch
from utils.common import save_path
from pathlib import  Path



ckpt_path = "***/models/stable-diffusion-xl-base-1.0"
pipe = RegionalDiffusionXLPipeline.from_pretrained(ckpt_path,torch_dtype=torch.float16, use_safetensors=True, variant="fp16").to("cuda")
pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config,use_karras_sigmas=True)

negative_prompt = "" # negative_prompt, 


prompt = "This comic is a black and white illustration that "


split_ratio = " 1,1,2;2,1,2"
prompt = "monochrome, comics"



regional_prompt = """
 A massive, detailed China dragon, its scales shimmering and claws extended, winding through the air with a sense of power and grace. BREAK
 a white sky BREAK 
 a white sky BREAK
 Some disciplined  soldiers, each holding a keyboard, marching in perfect unison, their faces determined and focused.
"""
lora_path = "***/models/kohya_ss_output/sdxl_lora_prodigy_comics_wjs_upscale/sdxl_lora_prodigy_comics_wjs_upscale.safetensors"



pipe.load_lora_weights(Path(lora_path).parent, weight_name=Path(lora_path).name)
pipe.fuse_lora()

num = 4
num =1; batch_size = 4
for i in range(num):
    images = pipe(
    prompts=regional_prompt,
    split_ratios=split_ratio, # The ratio of the regional prompt, the number of prompts is the same as the number of regions
    batch_size = batch_size, #batch size
    base_ratios = 0.5, # The ratio of the base prompt    
    base_prompts= prompt,       
    num_inference_steps=20, # sampling step
    height = 1024, 
    negative_prompt=negative_prompt, # negative prompt
    width = 1024, 
    seed = 1234567,# random seed
    guidance_scale = 7.0
    ).images
    save_path(images)
    