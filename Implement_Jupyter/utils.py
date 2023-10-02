from glob import glob
from keras_unet_collection import utils
from shapely.affinity import affine_transform
from shapely.geometry import MultiPolygon, Polygon
from tensorflow import keras
from sklearn.preprocessing import MinMaxScaler

import cv2
import geopandas as gpd
import json
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import rasterio
import tifffile as tiff
import warnings

# Update the base path to the correct data directory later
# Used in makedir() , you should check the path Line 311~ 321
base_path = "/home/u2018144071/SIG" 

def merge_tiff(input_tiffs, output_path):
    """
    Merge multiple TIFF images into a single TIFF file while preserving metadata.

    Parameters
    ----------
    input_tiffs : list of str
        List of file paths to the input TIFF images.
    output_path : str
        File path to save the merged TIFF image.

    Returns
    -------
    None
        The function saves the merged TIFF image to the specified output path.

    Notes
    -----
    This function reads multiple TIFF images specified in `input_tiffs` and merges them into a single TIFF image.
    The metadata (nodata value, coordinate reference system, and transformation) of the first input TIFF image is preserved in the merged image.

    """
    # Get the number of input TIFFs (channels)
    channels = len(input_tiffs)
    
    # Loop over each TIFF file
    for i in range(channels):
        filepath = input_tiffs[i]
        
        # Read the TIFF image as a float32 array
        img = tiff.imread(filepath, dtype=np.float32)

        # If it's the first channel, get metadata from the first TIFF
        if i == 0:
            with rasterio.open(filepath) as src:
                nodata = src.nodata
                crs = src.crs
                transform = src.transform

            # Get the width and height of the image
            width, height = np.shape(img)
            # Initialize an array to hold the merged image
            merged_img = np.zeros((channels, width, height), dtype=np.float32)

        # Store the image data for this channel in the merged image array
        merged_img[i] = img

    # Write the merged image to the output path
    tiff.imwrite(output_path, merged_img)

    # Open the merged image file in 'r+' mode (read and write)
    with rasterio.open(output_path, 'r+') as dst:
        # Update the metadata of the target file
        dst.nodata = nodata
        dst.crs = crs
        dst.transform = transform

    # Print a message indicating that the merging is finished
    print("TIFF files merged successfully.")

def image_to_array(filenames, size, channel):
    '''
    Converting images to numpy arrays.
    
    Input
    ----------
    filenames : ndarray
        An iterable containing the paths of image files.
    size : int
        The output size (height == width) of the image.
    channel : int
        Number of image channels, e.g., channel=3 for RGB.
        
    Output
    ----------
    ndarray
        An array with shape (filenum, size, size, channel).
    '''

    # Get the number of files
    L = len(filenames)

    # Allocate an array to store the images
    out = np.empty((L, size, size, channel))

    # Load images into the array
    if channel == 1:
        # If the image is grayscale (1 channel)
        for i, name in enumerate(filenames):
            pix = tiff.imread(name)
            out[i, ..., 0] = np.array(pix)
    else:
        # If the image has multiple channels (e.g., RGB)
        for i, name in enumerate(filenames):
            pix = tiff.imread(name)
            out[i, ...] = np.array(pix)[..., :channel]

    # Reverse the order of the columns (mirror image)
    return out[:, ::-1, ...]

def split_samples_supraglacial(label_filenames, rate, supraglacial_id=0):
    '''
    Subsetting samples appropriate for training
    
    Parameters
    ----------
    label_filenames : ndarray of filenames
        An iterable containing the paths of label image files.
    rate : float
        The minimum proportion of pixels that should belong to the supraglacial class.
    supraglacial_id : int, optional
        The label value representing the supraglacial class (default is 0).

    Returns
    -------
    list of bool
        A list of booleans indicating whether each sample meets the pixel proportion threshold.

    Notes
    -----
    This function subsets samples from the input labels that have at least `rate` proportion of pixels
    belonging to the supraglacial class (identified by the specified label value, `supraglacial_id`).

    '''

    size = 256  # Input image size
    thres = int(size * size * rate)  # Pixel number threshold
    L = len(label_filenames)
    flag = []  # Return a list of booleans

    # Iterate through label files and check pixel proportion for the supraglacial class
    for i in range(L):
        sample_ = image_to_array([label_filenames[i]], size, channel=1)

        # Check if the supraglacial class proportion meets the threshold
        if np.sum(sample_[0, ..., 0] == supraglacial_id) > thres:
            flag.append(True)
        else:
            flag.append(False)

    return flag

def split_samples_edge(input_filenames):
    '''
    Subsetting samples appropriate for training
    
    Parameters
    ----------
    input_filenames : ndarray of filenames
        An iterable containing the paths of input image files.

    Returns
    -------
    list of bool
        A list of booleans indicating whether each sample contains edge pixels.

    Notes
    -----
    This function subsets samples from the input images, excluding those that contain edge pixels.

    '''
    
    size = 256  # Input image size
    L = len(input_filenames)
    flag = []  # Return a list of booleans

    # Iterate through input files and check for edge pixels
    for i in range(L):
        # Convert input image to array with 8 channels (assuming it's an 8-channel image)
        sample_ = image_to_array([input_filenames[i]], size, channel=8)

        # Check if any pixel in the first channel is less than 0, indicating edge pixels
        if np.any(sample_[0, ..., 0] < 0):
            flag.append(False)
        else:
            flag.append(True)

    return flag

def split_samples_cloud(input_filenames, under=0.45, upper=0.75, rate=0.05):
    '''
    Subsetting samples appropriate for training
    
    Parameters
    ----------
    input_filenames : ndarray of filenames
        An iterable containing the paths of input image files.

    under : threshold value of cloud in NDSI band

    over : threshold value of cloud in NDSI band

    rate : ratio of cloud in each image

    Returns
    -------
    list of bool
        A list of booleans indicating whether each sample contains edge pixels.

    Notes
    -----
    This function subsets samples from the input images, excluding those that contain many cloud pixels.

    '''

    size = 256  # Input image size
    thres = size * size * rate

    L = len(input_filenames)
    flag = [] # return a list of booleans
    for i in range(L):
        sample_ = image_to_array([input_filenames[i]], size, channel = 8)

        if np.sum((sample_[0,...,7] < upper) & (sample_[0,...,7] > under)) > thres:
            flag.append(False)
        else:
            flag.append(True)
    
    return flag

def split_samples_rock(input_filenames):
    '''
    Subsetting samples appropriate for training
    
    Parameters
    ----------
    input_filenames : ndarray of filenames
        An iterable containing the paths of input image files.

    Returns
    -------
    list of bool
        A list of booleans indicating whether each sample contains "rock" pixels.

    Notes
    -----
    This function subsets samples from the input images, excluding those that contain "rock" pixels.

    '''
    
    size = 256  # Input image size
    L = len(input_filenames)
    flag = []  # Return a list of booleans

    # Iterate through input files and check for "rock" pixels
    for i in range(L):
        # Convert input image to array with 8 channels (assuming it's an 8-channel image)
        sample_ = image_to_array([input_filenames[i]], size, channel=8)

        # Check if any pixel in the 6th channel (index 5, assuming 0-based indexing) is less than 0, indicating "rock" pixels
        if np.any(sample_[..., 5] < 0):
            flag.append(False)
        else:
            flag.append(True)

    return flag

def make_nameslist(data_dict, image=True):
    """
    Generates a list of file paths for .tif files based on the specified data and region in the input dictionary.

    Parameters
    ----------
    data_dict : dict
        A dictionary with dates as keys and corresponding region lists as items.
    image : bool, optional
        Specifies whether to get image paths (True) or label paths (False). Default is True.

    Returns
    -------
    list
        A list containing file paths for .tif files based on the specified data and region.

    Notes
    -----
    - For images, file paths are constructed as "./SIG/data/{date}/{date}_{region}".
    - For labels, file paths are constructed as "./SIG/data/{date}/{date}_{region}_mask".

    """
    
    # Initialize a list to store file paths
    names = []

    # Iterate over dates in the input dictionary
    for date in data_dict:
        regions = data_dict[date]  # Get the regions for the current date

        # Iterate over regions for the current date
        for region in regions:
            if image:
                # Construct the image file path
                filepath = f"{base_path}/data/{date}_{region}/"
                # Add all .tif files in the image folder to the names list
                names += glob(filepath + '*.tif')
            else:
                # Construct the label file path
                filepath_label = f"{base_path}/data/{date}_{region}_mask/"
                # Add all .tif files in the label folder to the names list
                names += glob(filepath_label + '*.tif')

    # Sort the file paths and convert to a numpy array
    names = np.array(sorted(names))

    return names

def data_subsetting(data_dict, case_name, rate):
    """
    Subsets data for training, validation, and testing.

    Parameters
    ----------
    data_dict : dict
        A dictionary specifying the desired regions to include in the data.
    case_name : str
        Name for the case or experiment.
    rate : float
        Rate to determine the minimum proportion of pixels for the supraglacial class.

    Returns
    -------
    str
        The path to the saved JSON file containing the data subsets.

    Notes
    -----
    This function processes the data based on the specified regions and creates subsets for training, validation, and testing.
    The subsets are saved in a JSON file for further use.

    """

    # Get input and label names for images
    input_names = make_nameslist(data_dict, image=True)
    label_names = make_nameslist(data_dict, image=False)

    # Subsetting based on supraglacial class proportion
    flag_supraglacial = split_samples_supraglacial(label_names, rate, supraglacial_id=0)
    input_names = input_names[flag_supraglacial]
    label_names = label_names[flag_supraglacial]

    # Subsetting based on edge pixels
    flag_edge = split_samples_edge(input_names)
    input_names = input_names[flag_edge]
    label_names = label_names[flag_edge]

    # Shuffle and split the data into training, validation, and testing sets
    L = len(input_names)
    ind_all = utils.shuffle_ind(L)  # Shuffle

    L_train = int(0.6 * L)
    L_test = int(0.2 * L)
    L_valid = L - L_train - L_test  # 6:2:2

    ind_train = ind_all[:L_train]
    ind_test = ind_all[L_train:L_train + L_test]
    ind_valid = ind_all[L_train + L_test:]

    train_input_names = input_names[ind_train]
    train_label_names = label_names[ind_train]
    valid_input_names = input_names[ind_valid]
    valid_label_names = label_names[ind_valid]
    test_input_names = input_names[ind_test]
    test_label_names = label_names[ind_test]

    # Create directories and save the data subsets to a JSON file
    os.makedirs(os.path.join(base_path, "result", case_name), exist_ok=True)
    json_save_path = os.path.join(base_path, "result", case_name, case_name) + ".json"

    data = {
        "input_names": input_names.tolist(),
        "label_names": label_names.tolist(),
        "train_input_names": train_input_names.tolist(),
        "train_label_names": train_label_names.tolist(),
        "valid_input_names": valid_input_names.tolist(),
        "valid_label_names": valid_label_names.tolist(),
        "test_input_names": test_input_names.tolist(),
        "test_label_names": test_label_names.tolist()
    }

    # Save the data dictionary to a JSON file
    with open(json_save_path, "w") as output_file:
        json.dump(data, output_file)

    # Print summary information
    print("Number of images with supraglacial lake : {0}".format(len(input_names)))
    print("Training:validation:testing = {}:{}:{}".format(L_train, L_valid, len(test_label_names)))

    return json_save_path


def input_data_process(input_array, delete_channel):
    '''
    Process input data by converting pixel values to the [0, 1] range using Z standard normalization.

    Parameters
    ----------
    input_array : ndarray
        Input data array.
    delete_channel : int
        Index of the channel to delete from the input array.

    Returns
    -------
    ndarray
        Processed input array.

    Notes
    -----
    This function performs Z standard normalization on the input data, which involves converting pixel values to the [0, 1] range.
    The specified channel is deleted from the input array before normalization.

    '''

    # Delete the specified channel from the input array
    input_array = np.delete(input_array, delete_channel, axis=-1)

    images, x_size, y_size, channels = np.shape(input_array)

    #----------------------------------Min-Max Normalization-----------------------------------------#
    # for img in range(images):
    #     for ch in range(channels):
    #         img_ = input_array[img,...,ch]
    #         scaler = MinMaxScaler()
    #         input_array[img,...,ch] = scaler.fit_transform(img_.reshape([-1,1])).reshape(img_.shape)
    #------------------------------------------------------------------------------------------------#

    # Perform Z standard normalization
    for img in range(images):
        for ch in range(channels):
            img_ = input_array[img, ..., ch]
            mean = np.mean(img_)
            std_dev = np.std(img_)
            input_array[img, ..., ch] = (img_ - mean) / std_dev

    return input_array

def target_data_process(target_array):
    """
    Process the target data for a binary classification problem.

    Parameters
    ----------
    target_array : ndarray
        Target data array.

    Returns
    -------
    ndarray
        Processed target data in one-hot encoded categorical format.

    Notes
    -----
    This function processes the target data by thresholding the values and converting them into a one-hot encoded format suitable for a binary classification problem.

    """
    
    # Threshold the target array to convert it to a binary format
    target_array[target_array > 0] = 1

    # Convert the binary target array to one-hot encoded categorical format
    # Assumes binary classification with 2 classes
    return keras.utils.to_categorical(target_array, num_classes=2)

# Function to decorate axes for visualization
def ax_decorate_box(ax):
    # Remove axis lines and ticks
    [j.set_linewidth(0) for j in ax.spines.values()]
    ax.tick_params(axis="both", which="both", bottom=False, top=False,
                   labelbottom=False, left=False, right=False, labelleft=False)
    return ax

# Function to visualize the results
def show_result(i_sample, n_images, threshold, test_input, test_label, y_pred):
    """
    Visualize the results of a segmentation model for a specific sample.

    Parameters
    ----------
    i_sample : int
        Index of the sample to visualize.
    n_images : int
        Number of images to display in the visualization.
    threshold : float
        Threshold for binarizing the predicted probabilities.
    test_input : ndarray
        Test input data array.
    test_label : ndarray
        Ground truth labels for the test data.
    y_pred : ndarray
        Predicted probabilities from the model.

    Returns
    -------
    None

    Notes
    -----
    This function visualizes the original image, predicted probabilities, thresholded result, and ground truth label for a specific sample.

    """

    # Binarize the predicted probabilities based on the threshold
    y_result = np.where(y_pred < threshold, 0, 1)

    # Create subplots for visualization
    fig, AX = plt.subplots(1, n_images, figsize=(13, (13-0.2)/n_images))
    plt.subplots_adjust(0, 0, 1, 1, hspace=0, wspace=0.1)
    for ax in AX:
        ax = ax_decorate_box(ax)

    # Display the original image
    AX[0].pcolormesh(np.mean(test_input[i_sample, ...,], axis=-1), cmap=plt.cm.gray)
    AX[0].set_title("Original", fontsize=14);

    # Display the predicted probabilities
    AX[1].pcolormesh(y_pred[i_sample, ..., 0], cmap=plt.cm.jet)
    AX[1].set_title("Red for high probabilities", fontsize=14);

    # Display the thresholded result
    AX[2].pcolormesh(y_result[i_sample, ..., 0], cmap=plt.cm.jet)
    AX[2].set_title("Threshold = {0}".format(threshold), fontsize=14);

    # Display the ground truth label
    AX[3].pcolormesh(test_label[i_sample, ..., 0], cmap=plt.cm.jet)
    AX[3].set_title("Labeled truth", fontsize=14);

    # Calculate Dice Coefficient
    a = y_result[i_sample, ..., 0]
    b = test_label[i_sample, ..., 0]
    intersect = np.sum(a * b)
    total_sum = np.sum(a) + np.sum(b)
    dice = np.mean(2 * intersect / total_sum)
    print("Dice Coefficient:", dice)

    plt.show()  # Show the visualization

# Example usage
# show_result(i_sample, n_images, threshold, test_input, test_label, y_pred)

def find_best_threshold(test_label, y_pred):
    """
    Find the best threshold that maximizes the F1 score for binarizing predicted probabilities.

    Parameters
    ----------
    test_label : ndarray
        Ground truth labels for the test data.
    y_pred : ndarray
        Predicted probabilities from the model.

    Returns
    -------
    float
        The best threshold.

    Notes
    -----
    This function iterates through a range of thresholds and selects the one that maximizes the F1 score.

    """

    threshold = 0.01  # Initial threshold
    best = 0  # Initialize the best F1 score

    while threshold <= 0.99:
        # Binarize the predicted probabilities based on the threshold
        y_result = np.where(y_pred > threshold, 1, 0)
        f1_score = 0

        # Calculate F1 score for each sample and accumulate the scores
        for i in range(len(y_result)):
            a = y_result[i, ..., 0]
            b = test_label[i, ..., 0]
            intersect = np.sum(a * b)
            total_sum = np.sum(a) + np.sum(b)
            dice = np.mean(2 * intersect / total_sum)
            f1_score += dice / len(y_pred)

        # Update the best F1 score and threshold if needed
        if f1_score > best:
            best = f1_score
            best_threshold = threshold

        threshold += 0.01  # Increase the threshold for the next iteration

    # Round the best threshold to two decimal places
    return round(best_threshold, 2)

def f1_score(test_label, y_pred, threshold=0.5):
    """
    Calculate the F1 score for a given threshold and identify the best and worst images.

    Parameters
    ----------
    test_label : ndarray
        Ground truth labels for the test data.
    y_pred : ndarray
        Predicted probabilities from the model.
    threshold : float, optional
        Threshold for binarizing the predicted probabilities. Default is 0.5.

    Returns
    -------
    float
        The F1 score.
    list
        Indices of the best images.
    list
        Indices of the worst images.

    Notes
    -----
    This function calculates the F1 score and identifies the best and worst images based on the F1 score.

    """

    # Binarize the predicted probabilities based on the threshold
    y_result = np.where(y_pred > threshold, 1, 0)
    n = len(y_result)
    f1_score = 0

    f1_min = 1
    f1_max = 0
    best_img = []
    worst_img = []

    # Calculate F1 score for each image and accumulate the scores
    for i in range(n):
        a = y_result[i, ..., 0]
        b = test_label[i, ..., 0]
        intersect = np.sum(a * b)
        total_sum = np.sum(a) + np.sum(b)
        dice = np.mean(2 * intersect / total_sum)

        # Identify best and worst images based on the F1 score
        if dice > 0.9:
            best_img.append(i)
        if dice < 0.1:
            worst_img.append(i)

        # Update the maximum and minimum F1 score and corresponding indices
        if dice >= f1_max:
            f1_max = dice
            n_max = i
        if dice <= f1_min:
            f1_min = dice
            n_min = i

        f1_score += dice / n

    # Print F1 score and information about the best and worst images
    print("F1 score : {0} // threshold : {1}".format(round(f1_score, 3), threshold))
    print("*{0}th image is the best : {1}\nbest image list : {2}".format(n_max, f1_max, best_img))
    print("*{0}th image is the worst : {1}\nworst image list : {2}".format(n_min, f1_min, worst_img))

    # Return the F1 score, best image indices, and worst image indices
    return round(f1_score, 3), best_img, worst_img  # Round up to 3 decimal places

def get_contours_hierarchy(img):
    """
    Find contours and hierarchy in a binary image using OpenCV.

    Parameters
    ----------
    img : ndarray
        Binary image for contour detection.

    Returns
    -------
    list
        List of contours found in the image.
    ndarray
        Contour hierarchy information.

    Notes
    -----
    This function finds contours and hierarchy in a binary image using OpenCV's `findContours` function.

    """

    # Apply a threshold to the input image to create a binary image
    _, img_th = cv2.threshold(img, 185, 255, cv2.THRESH_BINARY_INV)

    # Find contours and hierarchy in the binary image
    contours, hierarchy = cv2.findContours(img_th, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    hierarchy = np.squeeze(hierarchy)  # Remove extra dimensions from hierarchy

    return contours, hierarchy

def merge_polygons(polygon: MultiPolygon, idx: int, add: bool, contours, hierarchy) -> MultiPolygon:
    """
    Merge a new polygon with a main polygon based on a given contour and its hierarchy.

    Parameters
    ----------
    polygon : MultiPolygon
        Main polygon to which a new polygon is added or subtracted.
    idx : int
        Index of the contour.
    add : bool
        If True, the contour should be added to the main polygon. If False, subtracted.
    contours : list
        List of contours.
    hierarchy : ndarray
        Contour hierarchy information.

    Returns
    -------
    MultiPolygon
        The merged polygon.

    Notes
    -----
    This function merges a new polygon with a main polygon based on a given contour and its hierarchy.

    """

    # Get contour from the global list of contours
    contour = np.squeeze(contours[idx])

    # cv2.findContours() sometimes returns a single point -> skip this case
    if len(contour) > 2:
        # Convert contour to a Shapely polygon
        new_poly = Polygon(contour)

        # Not all polygons are Shapely-valid (self-intersection, etc.)
        if not new_poly.is_valid:
            # Convert an invalid polygon to a valid one
            new_poly = new_poly.buffer(0)

        # Merge or subtract the new polygon with/from the main one
        if add:
            polygon = polygon.union(new_poly)
        else:
            polygon = polygon.difference(new_poly)

    # Check if the current polygon has a child
    child_idx = hierarchy[idx][2]
    if child_idx >= 0:
        # Call this function recursively, negate the `add` parameter
        polygon = merge_polygons(polygon, child_idx, not add, contours, hierarchy)

    # Check if there is another polygon at the same hierarchy level
    next_idx = hierarchy[idx][0]
    if next_idx >= 0:
        # Call this function recursively
        polygon = merge_polygons(polygon, next_idx, add, contours, hierarchy)

    return polygon

def get_trans_params(transform):
    """
    Extract translation and rotation parameters from a given transformation matrix.

    Parameters
    ----------
    transform : ndarray
        Transformation matrix.

    Returns
    -------
    ndarray
        1D array containing translation and rotation parameters.

    Notes
    -----
    This function extracts translation and rotation parameters from a transformation matrix
    and rearranges them into a 1D array.

    """

    # Convert the transformation matrix to a numpy array
    transform = np.array(transform)

    # Initialize an array to hold the translation and rotation parameters
    trans_params = np.zeros(6)

    # Extract and rearrange the parameters
    trans_params[:2] = transform[:2]
    trans_params[2:4] = transform[3:5]
    trans_params[4] = transform[2]
    trans_params[5] = transform[5]

    return trans_params

def save_result(case_name, model_name, y_result, input_names, train=True, whole=True):
    """
    Save the results of a model run, including model outputs and polygons derived from the model outputs.

    Parameters
    ----------
    case_name : str
        Name of the case or scenario.
    model_name : str
        Name of the model.
    y_result : ndarray
        Model outputs after thresholding.
    input_names : ndarray
        Paths of the test set files used for training.
    train : bool, optional
        True if for training, False otherwise, by default True.
    whole : bool, optional
        True if for the whole region, False for only the test set, by default True.

    Returns
    -------
    None

    Notes
    -----
    This function saves the model outputs, metadata, and polygons derived from the model outputs.

    """

    # Filter UserWarnings for a cleaner output
    warnings.filterwarnings("ignore", category=UserWarning)

    # Save path for result storage
    save_path = os.path.join(base_path, "result", case_name, model_name)

    # Create folders for storing model results
    if train:
        result_path = os.path.join(save_path, "train_region/model_result/")
        gpkg_path = os.path.join(save_path, "train_region/gpkg/")
        merged_gpkg_path = os.path.join(save_path, "merged_gpkg")
    else:
        result_path = os.path.join(save_path, "test_region/model_result/")
        gpkg_path = os.path.join(save_path, "test_region/gpkg/")
        merged_gpkg_path = os.path.join(save_path, "merged_gpkg")

    os.makedirs(result_path, exist_ok=True)
    os.makedirs(gpkg_path, exist_ok=True)
    os.makedirs(merged_gpkg_path, exist_ok=True)

    # Prepare the model outputs for saving
    result = np.array(y_result[..., 0])
    result_ = (256 * result[:]/256).astype(np.uint8)

    # Initialize a GeoDataFrame to store merged polygons
    merged_multipolygon = gpd.GeoDataFrame()

    # Iterate through each input image
    for i in range(len(input_names)):
        
        # Step 1. Save model results and metadata
        file_name = input_names[i].split("/")[-1]
        src_file_path = input_names[i]
        dst_file_path = os.path.join(result_path, file_name)

        img = cv2.flip(255 * result_[i, ...], 0).astype(np.uint8)

        with rasterio.open(src_file_path) as src:
            nodata = src.nodata
            crs = src.crs
            transform = src.transform

        if not whole:
            cv2.imwrite(dst_file_path, img)

            with rasterio.open(dst_file_path, 'r+') as dst:
                dst.nodata = nodata
                dst.crs = crs
                dst.transform = transform

        # Step 2. Generate polygons from contours and hierarchy
        contours, hierarchy = get_contours_hierarchy(img)

        try:
            polygon = merge_polygons(MultiPolygon(), 0, True, contours, hierarchy)
        except IndexError as e:
            print("An IndexError occurred: " + str(e) + "{0} has only 1 or 2 point(s). To create a polygon from contours, a minimum of three points is required.".format(dst_file_path))
            polygon = MultiPolygon()

        trans_params = get_trans_params(transform)
        geopolygon = affine_transform(polygon, trans_params)

        # Step 3. Create GeoPackage (.gpkg) file with polygons
        gdf = gpd.GeoDataFrame(geometry=[geopolygon])
        merged_multipolygon = pd.concat([merged_multipolygon, gdf])

        gdf.crs = crs

        if not whole:
            gpkg_save_path = os.path.join(gpkg_path, file_name + ".gpkg")
            gdf.to_file(gpkg_save_path, driver="GPKG")

        if i % 100 == 0:
            print("Merged {0}th images...{1} images are left...GOGOGOGOGO".format(i, (len(input_names)-i)))

    merged_multipolygon.crs = "EPSG:3857"

    if train and whole:
        merged_gpkg_name = "{0}_{1}_train.gpkg".format(case_name, model_name)
    elif train and not whole:
        merged_gpkg_name = "{0}_{1}_onlytestset.gpkg".format(case_name, model_name)
    else:
        merged_gpkg_name = "{0}_{1}_test.gpkg".format(case_name, model_name)

    merged_multipolygon.to_file(os.path.join(merged_gpkg_path, merged_gpkg_name), driver='GPKG')

    print("Task finished!")
