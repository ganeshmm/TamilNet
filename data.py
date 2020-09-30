import os
import shutil
from PIL import Image, ImageFilter, ImageOps
import numpy as np
from scipy import ndimage

def process(img):
    conv = img.convert("L")
    inv = ImageOps.invert(conv)
    pad = ImageOps.expand(inv, 2)
    thick = pad.filter(ImageFilter.MaxFilter(5))
    ratio = 40 / max(thick.size)
    new_size = tuple([int(round(x*ratio)) for x in thick.size])
    res = thick.resize(new_size, Image.LANCZOS)

    arr = np.asarray(res)
    com = ndimage.measurements.center_of_mass(arr)
    result = Image.new("L", (64, 64))
    box = (int(round(32 - com[1])), int(round(32 - com[0])))
    result.paste(res, box)
    return result

train_root = 'data/unprocessed/train/'
test_root = 'data/unprocessed/test/'
train_target = 'data/processed/train/'
test_target = 'data/processed/test/'

for i in range(156):
    os.mkdir(os.path.join(train_target, str(i)))
    os.mkdir(os.path.join(test_target, str(i)))

train_count = 0
for root, dirs, files in os.walk(train_root):
    for dir_name in dirs:
        for dir_root, subdirs, dir_files in os.walk(os.path.join(root, dir_name)):
            for file_name in dir_files:
                if (file_name == "Thumbs.db"):
                    continue
                # img = Image.open(os.path.join(dir_root, file_name))
                # result = process(img)
                file_path = os.path.join(dir_root, file_name)
                label = file_name[:3]
                new_name = label + 'u' + dir_name[4:] + file_name[3:]
                location = os.path.join(train_target, str(int(label)), new_name)
                shutil.copy(file_path, location)
                # img.save(location)
                train_count += 1
print(str(train_count) + " training examples")

test_count = 0
f = open('data/ground_truth.txt', 'r')
for i in range(26926):
    # img = Image.open(test_root + str(i).zfill(5) + ".tiff")
    # result = process(img)
    file_path = test_root + str(i).zfill(5) + ".tiff"
    line = f.readline()
    label = line[6:-1]
    new_name = str(i).zfill(5) + '.tiff'
    location = os.path.join(test_target, label, new_name)
    shutil.copy(file_path, location)
    # img.save(location)
    test_count += 1
print(str(test_count) + " test examples")
