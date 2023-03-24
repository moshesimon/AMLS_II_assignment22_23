import os
import multiprocessing
from PIL import Image
import pandas as pd
from config import *

def transform_and_save_image(img_file, input_folder, output_folder, transform):
    """
    Transforms an image using the provided transformation function and saves it to the output folder.
    
    Args:
        img_file (str): The image file name.
        input_folder (str): The path to the input folder containing the original image.
        output_folder (str): The path to the output folder where the transformed image will be saved.
        transform (function): The transformation function to apply to the image.
    
    Returns:
        None

    """
    img_path = os.path.join(input_folder, img_file)
    img = Image.open(img_path)
    transformed_img = transform(img)
    save_path = os.path.join(output_folder, img_file)
    transformed_img.save(save_path)
    

def transform_and_save_images(input_folder, output_folder, transform):
    """
    Transforms all images in the input folder using the provided transformation function and saves them to the output folder.
    
    Args:
        input_folder (str): The path to the input folder containing the original images.
        output_folder (str): The path to the output folder where the transformed images will be saved.
        transform (function): The transformation function to apply to the images.

    Returns:
        None
    """
    os.makedirs(output_folder, exist_ok=True)
    image_files = [f for f in os.listdir(input_folder) if os.path.isfile(os.path.join(input_folder, f))]
    
    # Prepare arguments for the worker function
    args = [(img_file, input_folder, output_folder, transform) for img_file in image_files]

    # Use a multiprocessing pool to parallelize the image transformations
    with multiprocessing.Pool(processes=20) as pool:
        pool.starmap(transform_and_save_image, args)

def add_more_possitive_data():
    """
    Adds more positive (cancer) samples to the dataset by duplicating the existing positive samples.
    """
    labels = pd.read_csv(labels_dir)
    cancer = labels[labels['cancer'] == 1]
    for i in cancer.index:
        img_name = os.path.join(images_dir,f"{labels.loc[i, 'patient_id']}_{labels.loc[i, 'image_id']}.png")
        image = Image.open(img_name)
        for j in range(45):
            row = labels.loc[i]
            row['image_id'] = f"{row['image_id']}_{j}"
            labels = labels.append(row, ignore_index=True)
            image.save(os.path.join(images_dir,f"{row['patient_id']}_{row['image_id']}.png"))
    
    labels['age'].fillna(61.0, inplace=True)
    labels.to_csv(labels_augmented_dir, index=False)


def process_data():
    """
    Processes the dataset by adding more positive samples and transforming the images.
    """
    # Add more positive data
    add_more_possitive_data()
    # Transform and save images
    transform_and_save_images(images_dir, images_augmented_dir, transform_pipeline)

