"""
BiDA Lab - Universidad Autonoma de Madrid
Author: Sergio Romero-Tapiador
Creation Date: 20/07/2022
Last Modification: 03/11/2023
-----------------------------------------------------
This code provides the implementation of the Food Recognition Systems using Xception and EfficientNetV2 architectures.
All three models have been trained on the AI4Food-NutritionDB food image database. To run it successfully,
please follow the provided instructions posted in https://github.com/BiDAlab/AI4Food-NutritionDB
"""

# Import some libraries
import argparse
import glob
import os
import requests
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
from tensorflow.keras import models
from tensorflow.keras.preprocessing import image


# Parse the arguments
def parser_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--arch', default='efficientnetv2', help='Architecture to use: xception or efficientnetv2')
    parser.add_argument('--model', default='category', help='Model to use: category, subcategory, product')
    parser.add_argument('--img', default='single', help='Run a  single or multiple images')
    parser.add_argument('--show', default='false', help='true if show the images (false otherwise)')

    opt = parser.parse_args()

    assert opt.arch == "xception" or opt.arch == "efficientnetv2", "Architecture must be: xception or efficientnetv2"
    assert opt.model == "category" or opt.model == "subcategory" or opt.model == "product", "Model must be: category, subcategory or product"
    assert opt.img == "single" or opt.img == "multiple", "Img parameter must be single or multiple"
    assert opt.show == "true" or opt.show == "false", "Parameter to show (true) or not (false) the images"
    print(opt)
    return opt


# Show the image with the model prediction of the food class and the categorization level used
def show_img(img_path, final_class):
    img = image.load_img(img_path)
    plt.imshow(img)
    plt.title("Categorization level: " + opt.model)
    plt.xlabel("Prediction: " + final_class)
    plt.show()


# Load the image in numpy format
def load_img(img_path):
    img = image.load_img(img_path, target_size=(224, 224))
    x = image.img_to_array(img)
    x = x / 255
    x = np.expand_dims(x, axis=0)

    return x, img


# Predict the food class from the loaded model
def predict_img(model, img):
    preds = model.predict(img)
    result = np.argmax(preds, axis=1)
    return result


# Load the classes of the corresponding category selected
def load_classes(model, path):
    lst_classes = []
    file = open(os.path.join(path, model + "_classes.txt"), "r")

    classes = file.readlines()
    for current_class in classes:
        lst_classes.append(current_class.split("\n")[0])

    return lst_classes


# Download the model from the URL
def download_model(path, architecture, model):
    model_name = architecture + "_" + model + "_model" + ".hdf5"
    url = "http://atvs.ii.uam.es/atvs/AI4Food-NutritionDB/" + model_name

    response = requests.get(url, stream=True)

    # Check if the request was successful (status code 200)
    if response.status_code == 200:
        total_size = int(response.headers.get('content-length', 0))
        block_size = 1024  # 1 KB
        progress_bar = tqdm(total=total_size, unit='B', unit_scale=True)

        if os.path.isdir(path) is False:
            os.makedirs(path)

        with open(os.path.join(path, model_name), 'wb') as file:
            for data in response.iter_content(block_size):
                progress_bar.update(len(data))
                file.write(data)
        progress_bar.close()

        return True
    else:
        return False


# Main function
if __name__ == "__main__":
    # First, the arguments are parsed
    opt = parser_arguments()

    # Get the current path
    os.getcwd()
    os.chdir("..")
    path = os.getcwd()

    # Download models from URL if they are not in the models folder
    model_folder_dir = os.path.join(path, "models")
    model_dir = os.path.join(model_folder_dir, opt.arch + "_" + opt.model + "_model" + ".hdf5")
    print(model_dir)
    if not os.path.exists(model_dir):
        print("\n\n\nDownloading the model...")
        flag_download = download_model(model_folder_dir, opt.arch, opt.model)

        if flag_download:
            print("\n\n\nModel downloaded properly!")
        else:
            print("\n\n\nModel not downloaded properly!")
            exit()

    # Get the full path of the test directory
    test_dir = os.path.join(path, "media", "sample")

    # Load the food recognition system
    print("\n\n\nLoading the model...")
    model = models.load_model(model_dir)
    lst_classes = load_classes(opt.model, os.path.join(path, "src"))
    print("\n\n\nModel loaded properly!")

    # For category efficientnetv2 model, replace Dairy products with Beans and vice versa
    if opt.arch == "efficientnetv2" and opt.model == "category":
        lst_classes[7], lst_classes[8] = lst_classes[8], lst_classes[7]

    # Load the test image and predict its class
    print("\n\n\nTesting food images...")
    lst_imgs = os.listdir(test_dir)
    if opt.img == "single":
        lst_imgs = [lst_imgs[0]]

    for current_img in lst_imgs:
        if ".jpg" in current_img:
            # Load the current image
            img_path = glob.glob(os.path.join(test_dir, current_img), recursive=True)[0]
            x, img = load_img(img_path)

            # Predict the food class
            result = predict_img(model, x)
            final_class = lst_classes[result[0]]
            print(current_img + " food image predicted as " + final_class + "!")

            # Finally the image is shown
            if opt.show == "true":
                show_img(img_path, final_class)

    print("\n\n\nExiting...")
