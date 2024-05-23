from RegionalDiffusion_base import RegionalDiffusionPipeline
from RegionalDiffusion_xl import RegionalDiffusionXLPipeline
from diffusers.schedulers import KarrasDiffusionSchedulers,DPMSolverMultistepScheduler
from mllm import local_llm,GPT4
import torch
from utils.common import save_path
from pathlib import  Path
# If you want to load ckpt, initialize with ".from_single_file".

# MEMORY_CAL = GPUMemory()

ckpt_path = "/data/zzzj/models/stable-diffusion-xl-base-1.0/sd_xl_base_1.0.safetensors"
# pipe = RegionalDiffusionXLPipeline.from_single_file(ckpt_path,torch_dtype=torch.float16, use_safetensors=True, variant="fp16")
# If you want to use diffusers, initialize with ".from_pretrained".

ckpt_path = "/data/zzzj/models/stable-diffusion-xl-base-1.0"
pipe = RegionalDiffusionXLPipeline.from_pretrained(ckpt_path,torch_dtype=torch.float16, use_safetensors=True, variant="fp16").to("cuda")
pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config,use_karras_sigmas=True)
## User input
# prompt= 'A girl with white ponytail and black dress are chatting with a blonde curly hair girl in a white dress in a cafe.'
# prompt = "monochrome, a huge china dragon in the air and a crowd of soldier hold a keyboard march on"
# para_dict = GPT4(prompt,key='...Put your api-key here...')
# ## MLLM based split generation results
# split_ratio = para_dict['Final split ratio']
# regional_prompt = para_dict['Regional Prompt']
negative_prompt = "" # negative_prompt, 


# split_ratio = '0.5,0.5,0.5; 0.5,0.5,0.5'
prompt = "This comic is a black and white illustration that "
# regional_prompt = " Captures the woman in the black dress within the top half of the image. BREAK \
# Contains the woman in the white blouse within the top half. BREAK \
# Shows the lower half of the woman in the black dress. BREAK \
# Displays the lower half of the woman in the white blouse."

# split_ratio = "0.5,1;0.5,1"
# split_ratio = " 0.5,0.5,0.5;0.5,0.5,0.5"
# prompt = "monochrome, comics"
# regional_prompt = """
# A massive, detailed China dragon, its scales shimmering and claws extended, winding through the air with a sense of power and grace. BREAK 
# BREAK
# BREAK
# Some disciplined  soldiers, each holding a keyboard, marching in perfect unison, their faces determined and focused.
# """


split_ratio = " 1,1,2;2,1,2"
prompt = "monochrome, comics"
# regional_prompt = """
# white sky BREAK
# A massive, detailed China dragon, its scales shimmering and claws extended, winding through the air with a sense of power and grace. BREAK 
# Some disciplined  soldiers, each holding a keyboard, marching in perfect unison, their faces determined and focused.
# blank BREAK
# """


regional_prompt = """ A massive, detailed China dragon, its scales shimmering and claws extended, winding through the air with a sense of power and grace. BREAK
 a white sky BREAK 
 a white sky BREAK
 Some disciplined  soldiers, each holding a keyboard, marching in perfect unison, their faces determined and focused.
"""
# lora_path = "/data/zzzj/models/kohya_ss_output/sdxl_lora_prodigy_dim128_128/sdxl_lora_prodigy_dim128_128.safetensors"
lora_path = "/data/zzzj/models/kohya_ss_output/sdxl_lora_prodigy_comics_wjs_upscale/sdxl_lora_prodigy_comics_wjs_upscale.safetensors"


# pipe.unet = torch.compile(pipe.unet)

pipe.load_lora_weights(Path(lora_path).parent, weight_name=Path(lora_path).name)
pipe.fuse_lora()

num = 4
num =1; batch_size = 1
for i in range(num):
    images = pipe(
    prompt=regional_prompt,
    split_ratio=split_ratio, # The ratio of the regional prompt, the number of prompts is the same as the number of regions
    batch_size = batch_size, #batch size
    base_ratios = 0.5, # The ratio of the base prompt    
    base_prompt= prompt,       
    num_inference_steps=20, # sampling step
    height = 1024, 
    negative_prompt=negative_prompt, # negative prompt
    width = 1024, 
    seed = 1234567,# random seed
    guidance_scale = 7.0
    ).images
    save_path(images)
# images.save("test.png")

# MEMORY_CAL.after_memory()
# MEMORY_CAL.show()