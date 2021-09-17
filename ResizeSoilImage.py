import os
import shutil
import glob
from PIL import Image

size_width = 28
size_height = 28

dst_dir = "data/"
type_soil = ["clay", "gravel", "loam", "sand"]


for i in range(len(type_soil)):
    type = type_soil[i]

    # dir setting
    shutil.rmtree(dst_dir + type)
    os.mkdir(dst_dir + type)

    # file setting
    # files = glob.glob('./soil_data/' + type +  '/*.png') # all excavations
    files = glob.glob('./soil_data/' + type +  '/*2.png') # only 3rd excavation

    # loop in soil type
    for f in files:
        # file open
        img = Image.open(f)
        # gray scale
        img = img.convert('L')
        # crop image
        w = 30
        img_crop = img.crop((0+w, 0, img.size[0]-w, img.size[1]))

        # normalization?

        # edge

        # reseize image
        img_resize = img.resize((size_width, size_height))
        # save path
        root, ext = os.path.splitext(f)
        basename = os.path.basename(root)
        path = dst_dir + type + "/" + basename + ext
        # save
        img_resize.save(path)