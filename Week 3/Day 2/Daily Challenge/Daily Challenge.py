'''

Hands-On Image Transformation...

Data Augmentation
Data visualization using MatPlotLib or PIL libraries.

Load and Visualize Images using the Flower Color Images dataset


'''

print ("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")


# pip install tensorflow  # I can't run it on my machine because you need python 64-bit (not 32-bit)
# pip install keras
# pip install pillow


# ~~~~~~~~~~~~~~~~~~~~~


# To load an image and display it :

# Load an image using PIL
from PIL import Image

image_path = 'flowers/flowers/19_010.png'
original_image = Image.open(image_path)

# Display the original image using matplotlib
import matplotlib.pyplot as plt

plt.imshow(original_image)


# ~~~~~~~~~~~~~~~~~~~~~

# Rotate an image by 90 degrees:

from scipy.ndimage import rotate

def rotate_image_90_degrees(image):
    return rotate(image , 90, reshape=False, mode='nearest')

rotated_image = rotate_image_90_degrees(original_image)

plt.imshow(rotated_image)

# ~~~~~~~~~~~~~~~~~~~~~

# Flip an image horizontally and then vertically using mirror and flip functions from ImageOps

from PIL import Image, ImageOps

# Flip the image horizontally
flipped_horizontally = ImageOps.mirror(original_image)
# plt.imshow(flipped_horizontally)

# And then flip the image vertically
flipped_vertically = ImageOps.flip(flipped_horizontally)

plt.imshow(flipped_vertically)


# ~~~~~~~~~~~~~~~~~~~~~

# Zoom in on an image (scale by 1.2x) using .resize(...)

from PIL import Image

# Get the original size of the image
original_width, original_height = original_image.size

# Calculate the new size (zoom by 1.2x)
new_width = int(original_width * 1.2)
new_height = int(original_height * 1.2)

# Resize the image to the new size
zoomed_image = original_image.resize((new_width, new_height))

plt.imshow(zoomed_image)

