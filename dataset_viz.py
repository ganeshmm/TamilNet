import random
import numpy as np
from PIL import Image

train_input = np.load("data/processed/train.npy")
train_labels = np.load("data/processed/train_labels.npy")
test_input = np.load("data/processed/test.npy")
test_labels = np.load("data/processed/test_labels.npy")

rand = random.randrange(len(train_labels))
train = train_input[rand]
train_label = train_labels[rand]
train_img = Image.fromarray(train)
train_img.show()
print(train_label)

rand = random.randrange(len(test_labels))
test = test_input[rand]
test_label = test_labels[rand]
test_img = Image.fromarray(test)
test_img.show()
print(test_label)