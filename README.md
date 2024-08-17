# OVT-B-Dataset
This repository contains the download link and usage instructions for the new dataset.

## Download

You can download the dataset from the following link:
[BaiDuYun](https://pan.baidu.com/s/1hy44z_om609jIhXjRxXCug?pwd=8yy3) (8yy3)


## Description

This dataset OVT-B can be used as a new benchmark to the research of OVMOT.

## Usage

Instructions on how to use the dataset:

1. Download the dataset and annotation.
2. Extract the files.
3. Copy the ```CLASS```, ```base_id```, and ```novel_id``` from ovtb_classname.py and add them to the classname.py file under the roi_head folder of the ov detector.
4. Modify the ```data_root``` in the configs to the path where the OVT-B folder is located. Change ```ann_file``` to the path of ovtb_ann.json, ```img_prefix``` to data_root+'OVT-B', and ```prompt_path``` to the path of ovtb_class.pth.
5. Then test/evaluate by TAO-type/COCO-type dataset eval tools/codes.

## Organization

```
├── OVT-B
│   ├── AnimalTrack
│   │   ├── subdir
│   │   │   ├── img.jpg
│   │   │   └── ...
│   ├── GMOT-40
│   ├── ImageNet-VID
│   ├── LVVIS
│   ├── OVIS
│   ├── UVO
│   ├── YouTube-VIS-2021
├── ovtb_ann.json
├── ovtb_class.pth
├── ovtb_classname.py
├── ovtb_prompt.pth
└── OVTB-format.txt
```

## Data sample illustration

![Sample 1](assets/ovtb_frame.png)
![Sample 2](assets/ovtb_frame2.png)
![Sample 3](assets/ovtb_frame3.png)
![Sample 4](assets/ovtb_frame4.png)

## Citation

If you use this dataset in your research, please cite it as follows:
<!-- ```
@dataset{OVT-B_2024,
title = {OVT-B: A New Large-Scale Benchmark for Open-Vocabulary Multi-Object Tracking},
year = {2024}
}
``` -->
