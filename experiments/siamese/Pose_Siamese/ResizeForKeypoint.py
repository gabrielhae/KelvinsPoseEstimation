from PIL import Image
import os, sys

path = r"C:\DeepLearningFolder\Pose Estimation\speed\preprocessed_images_1200_cropped_resized_bbox\BlackBackground"
dirs = os.listdir( path )

#def resize():
for item in dirs:
    im = Image.open(path+'/'+item)
    f, e = os.path.splitext(path+'/'+item)
    print(path+'/'+item)
    imResize = im.resize((300,300), Image.ANTIALIAS)
    imResize.save(f+'_resized.jpg', 'JPEG', quality=90)

#resize()