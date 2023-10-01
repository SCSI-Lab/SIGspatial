# ACM SIGspatial cup 2023
## Supraglacier lake detection from SCSI 

## Dependencies
- Python
- Keras
- OpenCV
- Matplotlib
- Numpy
- Rasterio


## Contents
There are three main scripts in this repo:
- data_processing
- 
- 


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
- NDWI
- NDWIice
- NDSI)

## Pre-processing

