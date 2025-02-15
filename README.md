## PointOBB-v3:  Expanding Performance Boundaries of Single Point-Supervised Oriented Object Detection Supervision



### Paper
The paper is available at [PointOBB-v3](https://arxiv.org/abs/2501.13898). You are also welcome to check out the conference version [PointOBB](https://openaccess.thecvf.com/content/CVPR2024/html/Luo_PointOBB_Learning_Oriented_Object_Detection_via_Single_Point_Supervision_CVPR_2024_paper.html).


![Pipeline Image](PointOBB/docs/pipeline.jpg)
![Pipeline Image](PointOBB/docs/e2e.jpg)


### Train/Test
Please see [`PointOBB/README.md`](PointOBB/README.md).


### Weight

DIOR-R

|         Backbone         |  mAP  | Angle |  Config | Detector |                                                                                                                                                                              Download                                                                                                                                                                              |
| :----------------------: | :---: | :---: | :---:  | :------: |  :------------------------------------------------------------------------------------------------------------: |
| ResNet50 (1024,1024,200) | 41.82 | le90  | [pointobbv3-dior](PointOBB/configs2/pointobb/pointobbv3_r50_fpn_2x_dior_two_stage.py)|    Oriented RCNN  |  [model](https://drive.google.com/file/d/1ZqBQivJ19QFA-VVCRaYAOuPkgA8PtjNA/view?usp=sharing) |


DOTA-v1.0

|         Backbone         |  mAP  | Angle |  Config | Detector |                                                                                                                                                                              Download                                                                                                                                                                              |
| :----------------------: | :---: | :---: | :-----: | :------: |  :------------------------------------------------------------------------------------------------------------: |
| ResNet50 (1024,1024,200) | 50.44 | le90  | [pointobbv3-dota](PointOBB/configs2/pointobb/pointobbv3_r50_fpn_2x_dota_two_stage.py)|    Oriented RCNN |  [model](https://drive.google.com/file/d/1bFhYBdIMy6yBCyAmTVHcZP6UD3w9cbx8/view?usp=sharing) |


DOTA-v1.5

|         Backbone         |  mAP  | Angle |  Config | Detector |                                                                                                                                                                              Download                                                                                                                                                                              |
| :----------------------: | :---: | :---: | :-----: | :------: |  :------------------------------------------------------------------------------------------------------------: |
| ResNet50 (1024,1024,200) | 38.08 | le90  | [pointobbv3-dota15](PointOBB/configs2/pointobb/pointobbv3_r50_fpn_2x_dota15_two_stage.py)|    Oriented RCNN |  [model](https://drive.google.com/file/d/1bHMmClalEtupq4CJ-6sZBfqW6RfTLEqF/view?usp=sharing) |


DOTA-v2.0

|         Backbone         |  mAP  | Angle |  Config | Detector |                                                                                                                                                                              Download                                                                                                                                                                              |
| :----------------------: | :---: | :---: | :-----: | :------: |  :------------------------------------------------------------------------------------------------------------: |
| ResNet50 (1024,1024,200) | 24.86 | le90  | [pointobbv3-dota20](PointOBB/configs2/pointobb/pointobbv3_r50_fpn_2x_dota20_two_stage.py)|    Oriented RCNN |  [model](https://drive.google.com/file/d/1u__zL9CXyGyeAZVq9hQF-r1qhduzUG0C/view?usp=sharing) |

FAIR1M

|         Backbone         |  mAP  | Angle |  Config | Detector |                                                                                                                                                                              Download                                                                                                                                                                              |
| :----------------------: | :---: | :---: | :-----: | :------: |  :------------------------------------------------------------------------------------------------------------: |
| ResNet50 (1024,1024,200) | 20.19 | le90  | [pointobbv3-fair](PointOBB/configs2/pointobb/pointobbv3_r50_fpn_2x_fair_two_stage.py)|    Oriented RCNN |  [model](https://drive.google.com/file/d/1atuyx7-aZYSOndPkhpr7_4ygDmQNEWuq/view?usp=sharing) |


STAR

|         Backbone         |  mAP  | Angle |  Config | Detector |                                                                                                                                                                              Download                                                                                                                                                                              |
| :----------------------: | :---: | :---: | :-----: | :------: |  :------------------------------------------------------------------------------------------------------------: |
| ResNet50 (1024,1024,200) | 16.73 | le90  | [pointobbv3-star](PointOBB/configs2/pointobb/pointobbv3_r50_fpn_2x_star_two_stage.py)|    Oriented RCNN |  [model](https://drive.google.com/file/d/1m4NAIdTv5vBf9b_4DAJBETRjQw79khZ6/view?usp=sharing) |


RSAR

|         Backbone         |  mAP  | Angle |  Config | Detector |                                                                                                                                                                              Download                                                                                                                                                                              |
| :----------------------: | :---: | :---: | :-----: | :------: |  :------------------------------------------------------------------------------------------------------------: |
| ResNet50 (1024,1024,200) | 22.84 | le90  | [pointobbv3-rsar](PointOBB/configs2/pointobb/pointobbv3_r50_fpn_2x_rsar_two_stage.py)|    Oriented RCNN |  [model](https://drive.google.com/file/d/1bf4wzAApTUFm05sRXmXSyOWzrohW4wx-/view?usp=sharing) |


# The code of end-to-end version is in the other branch!

### Citation
If you find this work helpful, please consider to cite:
```
@article{zhang2025pointobb,
   title     = {PointOBB-v3: Expanding Performance Boundaries of Single Point-Supervised Oriented Object Detection},
   author     = {Zhang, Peiyuan and Luo, Junwei and Yang, Xue and Yu, Yi and Li, Qingyun and Zhou, Yue and Jia, Xiaosong and Lu, Xudong and Chen, Jingdong and Li, Xiang and others},
   journal    = {arXiv preprint arXiv:2501.13898},
   year       = {2025}
}
```
```
@InProceedings{luo2024pointobb,
   title     = {PointOBB: Learning Oriented Object Detection via Single Point Supervision},
   author    = {Luo, Junwei and Yang, Xue and Yu, Yi and Li, Qingyun and Yan, Junchi and Li, Yansheng},
   booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
   pages     = {16730-16740},
   year      = {2024}
}
```

-----

Special thanks to the codebase contributors of MMRotate and P2BNet!
```
@inproceedings{zhou2022mmrotate,
  title   = {MMRotate: A Rotated Object Detection Benchmark using PyTorch},
  author  = {Zhou, Yue and Yang, Xue and Zhang, Gefan and Wang, Jiabao and Liu, Yanyi and
             Hou, Liping and Jiang, Xue and Liu, Xingzhao and Yan, Junchi and Lyu, Chengqi and
             Zhang, Wenwei and Chen, Kai},
  booktitle={Proceedings of the 30th ACM International Conference on Multimedia},
  year={2022}
}
```

```
@inproceedings{P2BNet,
  title     = {Point-to-Box Network for Accurate Object Detection via Single Point Supervision},
  author    = {Pengfei Chen, Xuehui Yu, Xumeng Han, Najmul Hassan, Kai Wang, Jiachen Li, Jian Zhao, Humphrey Shi, Zhenjun Han, and Qixiang Ye},
  booktitle = {ECCV},
  year      = {2022}
}
```
