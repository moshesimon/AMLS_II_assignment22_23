import os
import multiprocessing
import torchvision
from PIL import Image
import pandas as pd
from config import *

def transform_and_save_image(img_file, input_folder, output_folder, transform):
    img_path = os.path.join(input_folder, img_file)
    img = Image.open(img_path)
    transformed_img = transform(img)
    save_path = os.path.join(output_folder, img_file)
    transformed_img.save(save_path)
    

def transform_and_save_images(input_folder, output_folder, transform):
    os.makedirs(output_folder, exist_ok=True)
    image_files = [f for f in os.listdir(input_folder) if os.path.isfile(os.path.join(input_folder, f))]
    
    # Prepare arguments for the worker function
    args = [(img_file, input_folder, output_folder, transform) for img_file in image_files]

    # Use a multiprocessing pool to parallelize the image transformations
    with multiprocessing.Pool(processes=20) as pool:
        pool.starmap(transform_and_save_image, args)

def add_more_possitive_data():
    
    labels = pd.read_csv(labels_dir)
    cancer = labels[labels['cancer'] == 1]
    for i in cancer.index:
        img_name = os.path.join(dataset_dir,f"{labels.loc[i, 'patient_id']}_{labels.loc[i, 'image_id']}.png")
        image = Image.open(img_name)
        for j in range(45):
            row = labels.loc[i]
            row['image_id'] = f"{row['image_id']}_{j}"
            labels = labels.append(row, ignore_index=True)
            image.save(os.path.join(dataset_dir,f"{row['patient_id']}_{row['image_id']}.png"))
    labels.to_csv(os.path.join(dataset_dir, "train_augmented.csv"), index=False)


def process_data():
    
    # Add more positive data
    add_more_possitive_data()
    # Transform and save images
    transform_and_save_images(images_dir, images_augmented_dir, transform_pipeline)
    