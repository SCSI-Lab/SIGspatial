# ACM SIGspatial cup 2023
## Supraglacier lake detection from SCSI

## Contents
There are three main scripts in this repo:
 - DataProcessing.ipynb
 - post_process.ipynb
 - OurModel.ipynb
 - utils.py

## Data set
We use additionally sentinel-2 image from "https://dataspace.copernicus.eu/browser/".
Each date of image is same with SIGspatial's data.
Among the sentinel-2 image, We use 5 bands for Deep learning.(Blue / Green / Red / NIR / SWIR)
And we calculate NDWI, NDWIice, NDSI index and use it.
Finally we have 8 channel iamges.
- Blue
- Green
- Red
- NIR
- SWIR
- NDWI (Normalized Difference Water Index)
- NDWIice (Normalized Difference Water Index - ice)
- NDSI (Normalized Difference Snow Index)

You can use 8 channel data here, Our dataset saved in forder named OurDataset.
You can download them nby date and region. (Data e.g. Greenland_sentinel2_19-06-03_region1.tif, mean 19-06-03 date region num 1)

"https://drive.google.com/drive/folders/1Uua1HvDV0hWTl8zmdLOJ5lQij2ODxElm?usp=drive_link"

Our Lake_polygon.gpkg Also in Google Drive folder named OurResult/lake_final.gpkg

## Execution process
The following steps outline the process of training and evaluating a model using the ourModel class.

### Setting Hyperparameters and Paths
Set the hyperparameters and data paths.

params_path = "/home/u2018144071/SIG/params2.json"
img_channels = 8
size = 256
n_case = 2
channel_dict = {
    "NIR + NDWI + NDWI_ice + R + SWIR": [2,3,4,5,6],
}
rate = 0.03
test_dict = {"0603": [1,3,5]}

### Model Training and Evaluation
Iterate through the specified number of cases (n_case) to train and evaluate the model.
For each case, initialize the model, preprocess the data, and proceed with training and evaluation.

### Parameters
params_path (str): Path to the hyperparameters configuration file.
img_channels (int): Number of image channels.
size (int): Image size.
n_case (int): Number of cases to proceed with training.
channel_dict (dict): Dictionary specifying channel combinations to be used.
test_dict (dict): Dictionary containing test data for storing evaluation results.

### How to Run
Set the hyperparameters and data paths as per the execution process above.
Execute the model training and evaluation code using the ourModel class.
The model is trained and evaluated for each case, and the results are saved.

## Data-Processing
- We Pre-processing image for deep learning. Follow the code under condition.
- We split image into 256 size.
- Overlap 50% with each image on train image and test image.
- We nake mask in each train image by given lake polygon gpkg.

## Post-processing
- After the results came out, We Post-process our results.
- First, we eliminate the Lake-polygons which don't match contest standards.
- Second, we also eliminate the Lake-polygons which were predicted that will not be lakse based on our custom-dataset.
- Post-processing procedure is simple, Just follow our jupyter-notebook.
