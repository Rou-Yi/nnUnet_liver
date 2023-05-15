# Model Overview
A pre-trained model for volumetric (3D) segmentation of brain tumor subregions from multimodal MRIs based on BraTS 2018 data.

## Workflow

The model is trained to segment 3 nested subregions of primary brain tumors (gliomas): the "enhancing tumor" (ET), the "tumor core" (TC), the "whole tumor" (WT) based on 4 aligned input MRI scans (T1c, T1, T2, FLAIR). 

![](https://developer.download.nvidia.com/assets/Clara/Images/clara_pt_brain_mri_segmentation_workflow.png)

- The ET is described by areas that show hyper intensity in T1c when compared to T1, but also when compared to "healthy" white matter in T1c. 
- The TC describes the bulk of the tumor, which is what is typically resected. The TC entails the ET, as well as the necrotic (fluid-filled) and the non-enhancing (solid) parts of the tumor. 
-  The WT describes the complete extent of the disease, as it entails the TC and the peritumoral edema (ED), which is typically depicted by hyper-intense signal in FLAIR.


## Data
The training data is from the [Multimodal Brain Tumor Segmentation Challenge (BraTS) 2018](https://www.med.upenn.edu/sbia/brats2018/data.html).

- Target: 3 tumor subregions
- Task: Segmentation
- Modality: MRI  
- Size: 285 3D volumes (4 channels each)

The provided labelled data was partitioned, based on our own split, into training (200 studies), validation (42 studies) and testing (43 studies) datasets.

# Training configuration
This model utilized a similar approach described in 3D MRI brain tumor segmentation
using autoencoder regularization, which was a winning method in BraTS2018 [1]. The training was performed with the following:

- Script: train.sh
- GPU: Atleast 16GB of GPU memory. 
- Actual Model Input: 224 x 224 x 144
- AMP: True
- Optimizer: Adam
- Learning Rate: 1e-4
- Loss: DiceLoss

## Input
Input: 4 channel MRI (4 aligned MRIs T1c, T1, T2, FLAIR at 1x1x1 mm)

1. Normalizing to unit std with zero mean
1. Randomly cropping to (224, 224, 144) 
1. Randomly spatial flipping
1. Randomly scaling and shifting intensity of the volume

## Output
Output: 3 channels
- Label 0: TC tumor subregion
- Label 1: WT tumor subregion
- Label 2: ET tumor subregion


# Model Performance
The model was trained with 200 cases with our own split, as shown in the datalist json file in config folder. 
The achieved Dice scores on the validation and testing data are: 
- Tumor core (TC): 0.8259 
- Whole tumor (WT): 0.9035 
- Enhancing tumor (ET): 0.7508 
- Average: 0.8267 


## Training Performance
A graph showing the training loss and the mean dice over 300 epochs.  
![](https://developer.download.nvidia.com/assets/Clara/Images/clara_pt_brain_mri_segmentation_train.png)

## Validation Performance
A graph showing the validation mean dice over 300 epochs.  

![](https://developer.download.nvidia.com/assets/Clara/Images/clara_pt_brain_mri_segmentation_val.png)


# Intended Use
The model needs to be used with NVIDIA hardware and software. For hardware, the model can run on any NVIDIA GPU with memory greater or equal to 16 GB. For software, this model is usable only as part of Transfer Learning & Annotation Tools in Clara Train SDK container.  Find out more about Clara Train at the [Clara Train Collections on NGC](https://ngc.nvidia.com/catalog/collections/nvidia:claratrainframework).

**The Clara pre-trained models are for developmental purposes only and cannot be used directly for clinical procedures.**

# License
[End User License Agreement](https://developer.nvidia.com/clara-train-eula) is included with the product. Licenses are also available along with the model application zip file. By pulling and using the Clara Train SDK container and downloading models, you accept the terms and conditions of these licenses.

# References
[1] Myronenko, Andriy. "3D MRI brain tumor segmentation using autoencoder regularization." International MICCAI Brainlesion Workshop. Springer, Cham, 2018. https://arxiv.org/abs/1810.11654.
