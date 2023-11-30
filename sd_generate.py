import time

import cv2
import numpy as np
import matplotlib.pyplot as plt
import torch
from PIL import Image
from diffusers import ControlNetModel, StableDiffusionControlNetPipeline, UniPCMultistepScheduler
from diffusers.utils import load_image

# Constants
CONTROLNET_MODEL = "lllyasviel/sd-controlnet-canny"
DIFFUSER_MODEL = "runwayml/stable-diffusion-v1-5"
LORA_WEIGHTS = "400shijing.safetensors"
SAVE_PREFIX = "output/test/test_14_{}.png"


def image_canny(image):
    image = np.array(image)
    low_threshold = 100
    high_threshold = 200
    image = cv2.Canny(image, low_threshold, high_threshold)
    image = image[:, :, None]
    image = np.concatenate([image, image, image], axis=2)
    return Image.fromarray(image)


def generate_controlnet_i2i(image, prompt, controlnet_model=CONTROLNET_MODEL, diffuser_model=DIFFUSER_MODEL,
                            lora_weights=LORA_WEIGHTS):
    image = image_canny(image)
    controlnet = ControlNetModel.from_pretrained(controlnet_model,
                                                 torch_dtype=torch.float16,
                                                 use_safetensors=True)
    pipe = StableDiffusionControlNetPipeline.from_pretrained(diffuser_model,
                                                             controlnet=controlnet,
                                                             torch_dtype=torch.float16,
                                                             use_safetensors=True).to("cuda")
    pipe.load_lora_weights('./models', weight_name=lora_weights)
    pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config)
    pipe.enable_model_cpu_offload()
    output = pipe(prompt, image=image).images[0]
    return output


def display_image(image):
    plt.imshow(image)
    plt.axis('off')  # No axes for this image
    plt.show()


def save_image(image, prefix=SAVE_PREFIX):
    image.save(prefix.format(int(time.time())))


def generate_controlnet_t2i(prompt, controlnet_model=CONTROLNET_MODEL, diffuser_model=DIFFUSER_MODEL,
                            lora_weights=LORA_WEIGHTS):

    return
