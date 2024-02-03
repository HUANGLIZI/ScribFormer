# ScribFormer [TMI 2024]

This repository is the official implementation of the paper ScribFormer: Transformer Makes CNN Work Better for Scribble-based Medical Image Segmentation. [ResearchGate](https://www.researchgate.net/publication/377930252_ScribFormer_Transformer_Makes_CNN_Work_Better_for_Scribble-based_Medical_Image_Segmentation)

## Datasets

### ACDC
1. The ACDC dataset with mask annotations can be downloaded from [ACDC](https://www.creatis.insa-lyon.fr/Challenge/acdc/).
2. The scribble annotations of ACDC have been released in [ACDC scribbles](https://vios-s.github.io/multiscale-adversarial-attention-gates/data). 
3. The pre-processed ACDC data used for training could be directly downloaded from [ACDC_dataset](https://github.com/HiLab-git/WSL4MIS/tree/main/data/ACDC).

### MSCMR
1. The MSCMR dataset with mask annotations can be downloaded from [MSCMRseg](https://zmiclab.github.io/zxh/0/mscmrseg19/data.html). 
2. The scribble annotations of MSCMRseg have been released in [MSCMR_scribbles](https://github.com/BWGZK/CycleMix/tree/main/MSCMR_scribbles). 
3. The scribble-annotated MSCMR dataset used for training could be directly downloaded from [MSCMR_dataset](https://github.com/BWGZK/CycleMix/tree/main/MSCMR_dataset).

## Requirements

Some important required packages include:
* Python 3.8
* CUDA 11.3
* [Pytorch](https://pytorch.org) 1.10.1.
* torchvision 0.11.2
* Some basic python packages such as Numpy, Scikit-image, SimpleITK, Scipy ......

Follow official guidance to install [Pytorch](https://pytorch.org).

## Training

To train the model, run this command:

```train
python train.py --root_path <dataset_path> --exp <path_to_save_model> --bilinear --linear_layer --fold MAAGfold --max_epoches 200 --weight_cam_subloss 0.25 0.5 0.75 1 --weight_cam 0.1
```

## Evaluation

To evaluate the model, run this command:

```eval
python test.py --bilinear --linear_layer --fold MAAGfold --exp <path_to_save_model> --save_prediction
```

## Citation

```bash
@article{li2023lvit,
  title={ScribFormer: Transformer Makes CNN Work Better for Scribble-based Medical Image Segmentation},
  author={Li, Zihan and Zheng, Yuan and Shan, Dandan and Yang, Shuzhou and Li, Qingde and Wang, Beizhan and Hong, Qingqi and Shen, Dinggang},
  journal={IEEE Transactions on Medical Imaging},
  year={2024},
  publisher={IEEE}
}
```
