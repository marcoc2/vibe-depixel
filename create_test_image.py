from PIL import Image
import numpy as np

# Create a small 10x10 pixel art image
# A simple red cross on a white background
data = np.ones((10, 10, 3), dtype=np.uint8) * 255
data[5, :] = [255, 0, 0] # Horizontal line
data[:, 5] = [255, 0, 0] # Vertical line

img = Image.fromarray(data)
img.save('test_pixel.png')
print("Imagem de teste 'test_pixel.png' gerada.")
