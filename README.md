## Introduction
This code combines MRI and WSI to effectively improve various indicators.
All code of this study will be fully disclosed after the article is accepted. During this period, any inquiries regarding the code or paper can be made to the author.
## Configuration Environment
CUDA=11.8  
python=3.9  
numpy=1.24.4  
torch=2.0.0  
torchvision=0.15.1  
pytorch-lightning=2.2.1
## Data preparation
MRI:BraTS2020  
WSI:TCGA(https://portal.gdc.cancer.gov/)
## MRI data preprocessing
python ./data/preprocess.py
## WSI data preprocessing
HistoQC:https://github.com/choosehappy/HistoQC  
### Extract patches from the whole slide images
python WSI_data/patch_extraction.py --cancer=GBM --num-cpus=6 --magnification=20 --patch-size=256 --stratify=idh,atrx,p19q --wsi_path --wsi_mask_path --output_path
## Training the model
python train_IDH_lightning.py
## Use SGM
![image](https://github.com/user-attachments/assets/ef7f5bf4-6adf-4ee8-a45d-22f55028c33e)
## No SGM
![image](https://github.com/user-attachments/assets/8c6f0d57-156a-48ea-b56a-2845bf1adc24)
