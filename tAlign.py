import sys
import numpy as np
from PIL import Image

binarized_text = sys.argv[1] if len(sys.argv) == 2 else 'text.png'

data = np.array(Image.open(binarized_text))
fft = np.fft.fft2(data)

max_peak = np.max(np.abs(fft))
fft[fft < (max_peak * 0.25)] = 0
abs_data = 1 + np.abs(fft)
c = 255.0 / np.log(1 + max_peak)
log_data = c * np.log(abs_data)

max_scaled_peak = np.max(log_data)
rows, cols = np.where(log_data > (max_scaled_peak * 0.90))
min_col, max_col = np.min(cols), np.max(cols)
min_row, max_row = np.min(rows), np.max(rows)
dy, dx = max_col - min_col, max_row - min_row
theta = np.arctan(dy / float(dx))
cos_theta = np.cos(theta)
sin_theta = np.sin(theta)

width, height = data.shape
cx, cy = width / 2, height / 2
new_image = np.zeros(data.shape)
for x, row in enumerate(data):
    for y, value in enumerate(row):
        xp = cx + (x - cx) * cos_theta - (y - cy) * sin_theta
        yp = cy + (x - cx) * sin_theta + (y - cy) * cos_theta
        if xp < 0 or yp < 0 or xp > width or yp > height:
            continue
        new_image[xp, yp] = data[x, y]

Image.fromarray(np.uint8(new_image) * 255).show()




# In[ ]:



