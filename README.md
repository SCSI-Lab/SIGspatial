# ACM SIGspatial cup 2023
## Supraglacier lake detection from SCSI 
후에 딥러닝 주요 모듈 및 쿠다버전 컴퓨터 gpu추가 / 구글드라이브 링크 추가


## Contents
There are three main scripts in this repo:
- DataProcessing.ipynb
- post_process.ipynb
- Main.py

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

## Make data set
- We Pre-processing image for deep learning. Follow the code under condition.
- We split image into 256 size.
- Overlap 50% with each image on train image and test image.
- We nake mask in each train image by given lake polygon gpkg.

## Post-processing
- After the results came out, We Post-process our results.
- First, we eliminate the Lake-polygons which don't match contest standards.
- Second, we also eliminate the Lake-polygons which were predicted that will not be lakse based on our custom-dataset.
- Post-processing procedure is simple, Just follow our jupyter-notebook.
