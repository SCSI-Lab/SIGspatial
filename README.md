# ACM SIGspatial cup 2023
## Supraglacier lake detection from SCSI 
후에 딥러닝 주요 모듈 및 쿠다버전 컴퓨터 gpu추가
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
- util.py
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
- NDWI (Normalized Difference Water Index)
- NDWIice (Normalized Difference Water Index - ice)
- NDSI (Normalized Difference Snow Index)

## Pre-processing

