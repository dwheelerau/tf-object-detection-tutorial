#!/usr/bin/env python
import os
from PIL import Image
import sys

# resize images in folder into a sqr
# NOTE: this will not preserve the aspect ration. padding methods
# are required for that, but that comes at a cost of trainging time.
# Given my datasets are small that pronbably won't work.  

# usage: 640 640 is x=640 by y=640
# python resize_images_in_folder.py <path> 640 640

path = sys.argv[1]
x = int(sys.argv[2])
y = int(sys.argv[3])

IMAGE_PATHS = []
for subdir, dirs, files in os.walk(path):
    for file in files:
        #print os.path.join(subdir, file)
        filepath = subdir + os.sep + file

        if filepath.endswith(".jpg"):
            IMAGE_PATHS.append(filepath)

for img in IMAGE_PATHS:
    print('resizing %s to %sx%s' % (img, x, y))
    image = Image.open(img)
    image = image.resize((x, y))
    fname = img.split(os.sep)[-1]
    new_fname = fname[:-4] + '_%sx%sresized.jpg' % (x, y)
    new_image_path = img.replace(fname, new_fname)
    print('resized version saved to %s' % new_image_path)
    image.save(new_image_path)
