**Official PyTorch code for the TBC2024 paper "MAFBLiF: Multi-scale Attention Feature Fusion Based Blind Light Field Image Quality Assessment". Please refer to the [paper](https://ieeexplore.ieee.org/document/10623345) for details.**

![image]([https://github.com/ZhiliangMa/MPU6500-HMC5983-AK8975-BMP280-MS5611-10DOF-IMU-PCB/blob/main/img/IMU-V5-TOP.jpg](https://github.com/oldblackfish/MAFBLiF/blob/main/fig/framework.png))

**Note: First, we convert the dataset into H5 files using MATLAB. Then, we train and test the model in Python.**

### Generate Dataset in MATLAB
Take the NBU-LF1.0 dataset for instance, convert the dataset into h5 files, and then put them into './Datasets/NBU_MLI_7x32x32/':
```
 ./MAFBLiF/Datasets/Generateh5_for_NBU_Dataset.m
```
    
### Train
Train the model using the following command:
```
python Train.py  --trainset_dir ./Datasets/NBU_MLI_7x32x32/
```

### Test Overall Performance
Test the overall performance using the following command:
```
python Test.py
```

### Test Individual Distortion Type Performance
Test the individual distortion type performance using the following command:
```
 python Test_Dist.py
```
### Acknowledgement
This project is based on [DeeBLiF](https://github.com/ZhengyuZhang96/DeeBLiF). Thanks for the awesome work.

### Citation
Please cite the following paper if you use this repository in your reseach.
```
@ARTICLE{10623345,
  author={Zhou, Rui and Jiang, Gangyi and Cui, Yueli and Chen, Yeyao and Xu, Haiyong and Luo, Ting and Yu, Mei},
  journal={IEEE Transactions on Broadcasting}, 
  title={MAFBLiF: Multi-Scale Attention Feature Fusion-Based Blind Light Field Image Quality Assessment}, 
  year={2024},
  volume={},
  number={},
  pages={1-13},
  keywords={Measurement;Feature extraction;Image quality;Visualization;Tensors;Electronic mail;Distortion measurement;Light field;blind image quality assessment;multi-scale attention;spatial-angular features;pooling},
  doi={10.1109/TBC.2024.3434699}}

```
### Contact
For any questions, feel free to contact: 2211100079@nbu.edu.cn or blackfish5254@gmail.com
