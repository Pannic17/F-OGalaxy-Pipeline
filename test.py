import time

import cv2

from sd_pipeline import StableDiffusionGenerate

print("START")
pipeline = StableDiffusionGenerate.from_default()
print(pipeline)
print("INIT")

print(type(pipeline))
print(type(pipeline.upscale))
print(dir(pipeline))


# TEST
image = cv2.imread("test/ngc7635.jpg")
# image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
print("READ IMAGE")
pipeline.photo = image
w, h, c = image.shape
points = [(0, 0), (w-1, 0), (w-1, h-1), (0, h-1)]
print("SET POINTS")

pipeline.cv_process_photo(points)
pipeline.set_status(901, "process photo<>calculate centers")
print(pipeline.message)

centers = pipeline.cv_calculate_centers(7)
pipeline.set_status(902, "calculate centers<>generate")
print(pipeline.message)

generated_path = pipeline.generate()
pipeline.set_status(903, "generate<>inpaint")
print(pipeline.message)

inpainted_path = pipeline.inpaint()
pipeline.set_status(904, "inpaint<>upscale")
print(pipeline.message)

upsacle_path = pipeline.upscale()
pipeline.set_status(905, "upscale<>save hdr")
print(pipeline.message)

hdri_path, hdra_path = pipeline.save_hdr()
pipeline.set_status(906, "save hdr<>calculate colors")
print(pipeline.message)

colors = pipeline.cv_cluster_colors(5)
pipeline.set_status(800, ">finish")
print(pipeline.message)

# pipeline.sd_controlnet_i2i(image, pipeline.prompt_generate)
time.sleep(10)
print("任务完成")