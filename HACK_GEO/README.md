# Semantic Segmentation using PyTorch

This project provides a PyTorch implementation of semantic segmentation. It includes a training segmentation model on mac os and google colab and a Jupyter Notebook for testing the model on new images.

This Jupyter notebook contains the implementation of a semantic segmentation model using PyTorch for segmenting images. The model is trained on a dataset of images that are labeled with corresponding ground truth masks. The trained model can then be used to segment new images, i.e., to assign a label to each pixel in the image.

## Requirements
- Refer to the tips and tricks in confluence for installing pytorch on mac M1 
## Dataset
### Dataset 1
The Semantic Drone Dataset focuses on semantic understanding of urban scenes for increasing the safety of autonomous drone flight and landing procedures. The imagery depicts more than 20 houses from nadir (bird's eye) view acquired at an altitude of 5 to 30 meters above ground. A high resolution camera was used to acquire images at a size of 6000x4000px (24Mpx). The training set contains 400 publicly available images and the test set is made up of 200 private images.

PERSON DETECTION
For the task of person detection the dataset contains bounding box annotations of the training and test set.

SEMANTIC SEGMENTATION
We prepared pixel-accurate annotation for the same training and test set. The complexity of the dataset is limited to 20 classes as listed in the following table.

Table 1: Semanic classes of the Drone Dataset

tree, gras, other vegetation, dirt, gravel, rocks, water, paved area, pool, person, dog, car, bicycle, roof, wall, fence, fence-pole, window, door, obstacle
[official website](https://www.kaggle.com/datasets/bulentsiyah/semantic-drone-dataset?resource=download).

### Dataset 2


### Dataset 3 

https://uavid.nl/ 


## Usage

To test the pre-trained segmentation model on new images, open the `segmentation_test.ipynb` Jupyter Notebook and follow the instructions provided.

```bash
jupyter notebook segmentation_test.ipynb
