from pathlib import Path
import time
from uuid import uuid1
from PIL import Image, ImageDraw
import time
import datetime



def get_time():
    time.strftime("%Y_%m_%d_%H_%M_%S", time.localtime())
    localtime = time.strftime('%Y-%m-%d_%H-%M-%S', time.localtime(time.time()))
    uuid_value = uuid1()
    uuid_str = uuid_value.hex
    random_str = localtime+"_"+uuid_str
    return random_str


output_dir = Path("./results")
output_dir.mkdir(exist_ok=True)

def save_path(images, output_dir=output_dir, prefix="test"):
    timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    current_day = timestamp.split("-")[0]
    for i, img in enumerate(images):
        save_dir = output_dir.joinpath(current_day)
        save_dir.mkdir(exist_ok=True)
        save_name = f"{prefix}_{timestamp}_{i:03d}.png" if prefix else f"image_{timestamp}_{i:03d}.png"
        save_path = str(save_dir.joinpath(save_name).absolute())
        img.save(save_path)