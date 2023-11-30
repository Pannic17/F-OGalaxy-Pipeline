import time

import PIL
import cv2
import numpy as np
import requests
import torch
from io import BytesIO

from PIL import Image
from diffusers import StableDiffusionInpaintPipeline
from matplotlib import pyplot as plt


def download_image(url):
    response = requests.get(url)
    return PIL.Image.open(BytesIO(response.content)).convert("RGB")

from cv_vignette import create_target_image, add_vignette, create_vignette_mask, extend_image_horizontal, \
    extend_image_all, create_extend_canvas, place_enlarge_blur, place_center_image

# init_image = load_image("https://huggingface.co/datasets/diffusers/test-arrays/resolve/main
# /stable_diffusion_inpaint/boy.png") mask_image = load_image(
# "https://huggingface.co/datasets/diffusers/test-arrays/resolve/main/stable_diffusion_inpaint/boy_mask.png")
width, height = 1024, 576

original_image = cv2.imread("output/universe/sd_nasa_1700211555.png")
# canvas = create_target_image(original_image, width, height)
# invert = create_vignette_image(width, height)
mask = create_vignette_mask(width, height, center_fraction=0.7)
# center = invert
# extend = extend_image_horizontal(original_image, width, height)
canvas = create_extend_canvas(width, height)
canvas = place_enlarge_blur(canvas, original_image)


# extend = extend_image_all(original_image, width, height)
# init_image = canvas * invert
extend = place_center_image(canvas, original_image)
init_image = cv2.cvtColor(extend.astype(np.uint8), cv2.COLOR_BGR2RGB)
init_image = Image.fromarray(init_image)
init_image = init_image.resize((width, height))
mask_image = cv2.cvtColor(mask.astype(np.uint8), cv2.COLOR_BGR2RGB)
mask_image = Image.fromarray(mask_image)
mask_image = mask_image.resize((width, height))

# img_url = "https://raw.githubusercontent.com/CompVis/latent-diffusion/main/data/inpainting_examples/overture-creations-5sI6fQgYIuo.png"
# mask_url = "https://raw.githubusercontent.com/CompVis/latent-diffusion/main/data/inpainting_examples/overture-creations-5sI6fQgYIuo_mask.png"

# init_image = download_image(img_url).resize((512, 512))
# mask_image = download_image(mask_url).resize((512, 512))

pipe = StableDiffusionInpaintPipeline.from_pretrained(
    "runwayml/stable-diffusion-inpainting", torch_dtype=torch.float16
)
pipe = pipe.to("cuda")

prompt = "universe, hdr, nebula, vignette, black on the edge, dark, space, galaxy, stars"
image = pipe(prompt=prompt, image=init_image, mask_image=mask_image, width=width, height=height).images[0]

plt.figure(figsize=(12, 6))

plt.subplot(1, 3, 1)
plt.title("init")
plt.imshow(init_image)
plt.axis('off')

plt.subplot(1, 3, 2)
plt.title("mask")
plt.imshow(mask_image)
plt.axis('off')

plt.subplot(1, 3, 3)
plt.title("generated")
plt.imshow(image)
plt.axis('off')
id = str(int(time.time()))
image.save("output/vignette/sd_inpaint_{}.png".format(id))

plt.show()

