import pandas as pd
import os
import random
import shutil


df = pd.read_csv('/ToT/train.csv', header=None)
categories = df[0].tolist()
source_folder = '/imagenet2012/train'
target_folder = '/ToT/data'


if not os.path.exists(target_folder):
    os.makedirs(target_folder)


for category in categories:

    category_folder = os.path.join(source_folder, category)
    images = os.listdir(category_folder)
    selected_images = random.sample(images, 500)


    target_category_folder = os.path.join(target_folder, category)
    if not os.path.exists(target_category_folder):
        os.makedirs(target_category_folder)


    for image in selected_images:
        source_path = os.path.join(category_folder, image)
        target_path = os.path.join(target_category_folder, image)
        shutil.copyfile(source_path, target_path)
