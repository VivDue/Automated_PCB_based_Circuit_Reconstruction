from simple_lama_inpainting import SimpleLama
from PIL import Image

simple_lama = SimpleLama()

img_path = "dcc_encoder_top_black_back.png"
mask_path = "mask.png"

image = Image.open(img_path)
mask = Image.open(mask_path).convert('L')

result = simple_lama(image, mask)
result.save("inpainted.png")