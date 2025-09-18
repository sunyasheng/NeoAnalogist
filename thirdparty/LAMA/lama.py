from simple_lama_inpainting import SimpleLama
from PIL import Image

simple_lama = SimpleLama()

img_path = "/home/suny0a/Proj/ImageBrush/NeoAnalogist/workspace/imgs/cat.png"
mask_path = "/home/suny0a/Proj/ImageBrush/NeoAnalogist/workspace/masks/cat_mask.png"

image = Image.open(img_path)
mask = Image.open(mask_path).convert('L')

result = simple_lama(image, mask)
result.save("inpainted.png")