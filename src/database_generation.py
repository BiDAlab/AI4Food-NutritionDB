"""
BiDA Lab - Universidad Autonoma de Madrid
Author: Sergio Romero-Tapiador
Creation Date: 20/07/2022
Last Modification: 14/09/2023
-----------------------------------------------------
This code implements the generation of the AI4Food-NutritionDB food image database. To generate it successfully, please ensure 
that all databases and the AI4Food-NutritionDB.txt file are located in the same directory as this Python program file. 
Then, execute this program and wait until the AI4Food-NutritionDB database is fully generated.
"""

# Import some libraries
import os
import shutil
from tqdm import tqdm

# Databases path for each one
UECFood256_path = None
Food101_path = None
Food11_path = None
FruitVeg81_path = None
MAFood121_path = None
ISIAFood500_path = None
VIPERFoodNet_path = None

# Get current path
os.getcwd()
os.chdir("..")
current_path = os.getcwd()


# Check if all the database paths are set properly
def check_ddbb_paths():
    flag = True

    if UECFood256_path is None:
        flag = False

    if Food101_path is None:
        flag = False

    if Food11_path is None:
        flag = False

    if FruitVeg81_path is None:
        flag = False

    if MAFood121_path is None:
        flag = False

    if ISIAFood500_path is None:
        flag = False

    if VIPERFoodNet_path is None:
        flag = False

    return flag


# Set each corresponding database path to the corresponding global variable
def search_current_path(path):
    global UECFood256_path
    global Food101_path
    global Food11_path
    global FruitVeg81_path
    global MAFood121_path
    global ISIAFood500_path
    global VIPERFoodNet_path

    if os.path.isdir(path):
        len_current_path = len(os.listdir(path))
        if len_current_path == 256:  # UECFood256
            UECFood256_path = path
        elif len_current_path == 101:  # Food101
            Food101_path = path
        elif len_current_path == 11:  # Food11
            if path[-4:] == "test":
                path = path[:-4]
            if path[-5:] == "train":
                path = path[:-5]
            if path[-3:] == "val":
                path = path[:-3]
            Food11_path = path
        elif len_current_path == 53:  # FruitVeg81
            FruitVeg81_path = path
        elif len_current_path == 121:  # MAFood121
            MAFood121_path = path
        elif len_current_path == 500:  # ISIAFood500
            ISIAFood500_path = path
        elif len_current_path == 82:  # VIPERFoodNet
            VIPERFoodNet_path = path
        else:
            return False

        return True
    else:
        return False


# Search automatically for each ddbb path, considering 3 different directory levels
def search_ddbb_path():
    try:
        list_folders = os.listdir(current_path)

        for folder in list_folders:
            current_path_folder = os.path.join(current_path, folder)
            found_flag = search_current_path(current_path_folder)

            if found_flag is False:
                if os.path.isdir(current_path_folder):
                    list_folders_level1 = os.listdir(current_path_folder)
                    if len(list_folders_level1) < 550:

                        for folder_level1 in list_folders_level1:
                            current_path_folder_level2 = os.path.join(current_path_folder, folder_level1)
                            found_flag_level1 = search_current_path(current_path_folder_level2)
                            if found_flag_level1 is False:
                                if os.path.isdir(current_path_folder_level2):
                                    list_folders_level2 = os.listdir(current_path_folder_level2)
                                    if len(list_folders_level2) < 550:

                                        for folder_level2 in list_folders_level2:
                                            current_path_folder_level3 = os.path.join(current_path_folder_level2,
                                                                                      folder_level2)
                                            search_current_path(current_path_folder_level3)

        return check_ddbb_paths()
    except:
        print("Please download all food image databases and placed them in the same current path as the python "
              "program file!")


# Link the database name with each corresponding database path
def link_ddbb_path(ddbb):
    if ddbb == "UECFood-256":
        return UECFood256_path
    elif ddbb == "Food-101":
        return Food101_path
    elif ddbb == "Food-11":
        return Food11_path
    elif ddbb == "FruitVeg-81":
        return FruitVeg81_path
    elif ddbb == "MAFood-121":
        return MAFood121_path
    elif ddbb == "ISIA Food-500":
        return ISIAFood500_path
    elif ddbb == "VIPER-FoodNet":
        return VIPERFoodNet_path

    return None


# Generate the AI4Food-NutritionDB food image database
def generate_ddbb():
    correspondence_file = None
    try:
        # First, the correspondence file is loaded
        correspondence_file = open(os.path.join(current_path, "AI4Food-NutritionDB.txt"), "r", encoding='utf-8')
    except:
        print("AI4Food-NutritionDB.txt must be placed in the same path as the python program file!")

    if correspondence_file is not None:
        # Each line correspond to an unique image split in 3 different objects by a | character:
        #       - Corresponding database
        #       - Source image path (from the corresponding database)
        #       - Destination image path
        imgs_correspondences = correspondence_file.readlines()
        for img_correspondence in tqdm(imgs_correspondences):
            split_line = img_correspondence.split("|")

            ddbb = split_line[0]  # Corresponding database
            img_src = split_line[1]  # Source image path (from the corresponding database)
            img_dst = split_line[2].split("\n")[0]  # Destination image path

            # Check if the directory exists
            final_folders_path = os.path.join(current_path, "/".join(img_dst.split("/")[0:-1]))
            if os.path.isdir(final_folders_path) is False:
                os.makedirs(final_folders_path)

            # Get the database path
            ddbb_path = link_ddbb_path(ddbb)
            if ddbb_path is not None:
                src = os.path.join(ddbb_path, img_src)
                dst = os.path.join(current_path, img_dst)

                try:
                    # Finally, make a copy of the source image in the final destination
                    shutil.copyfile(src, dst)
                except:
                    print("Error trying to copy the image " + src + " to " + dst)


def main():
    # Check if all databases are downloaded correctly
    print("Checking all the food image databases paths...")
    check_path = search_ddbb_path()
    if check_path is True:
        print("All the database paths have been set!\nGenerating AI4Food-NutritionDB food image database. Be patient, "
              "it may take a while...")
        generate_ddbb()
    else:
        print("Please download all food image databases and placed them in the same current path as the python "
              "program file!")


if __name__ == '__main__':
    main()
