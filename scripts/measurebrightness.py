import sys
from PIL import Image


def calculate_brightness(image):
    greyscale_image = image.convert('L')
    histogram = greyscale_image.histogram()
    pixels = sum(histogram)
    brightness = scale = len(histogram)

    for index in range(0, scale):
        ratio = histogram[index] / pixels
        brightness += ratio * (-scale + index)

    return 1 if brightness == 255 else brightness / scale


image = Image.open(r"C:\DeepLearningFolder\Pose Estimation\speed\preprocessed_images_600\train\img002085.jpg")#img002085.jpg img002087
print(calculate_brightness(image))
        