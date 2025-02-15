<p align="center">
  <h1 align="center">PointOBB-v3： Expanding Performance Boundaries of Single Point-Supervised Oriented Object Detection</h1>
  <p align="center">
      <a href='https://scholar.google.com.hk/citations?user=rQbW67AAAAAJ&hl' style='text-decoration: none' >Peiyuan Zhang</a><sup></sup>&emsp;
      <a href='https://scholar.google.com.hk/citations?hl=zh-CN&user=6XibZaYAAAAJ' style='text-decoration: none' >Junwei Luo</a><sup></sup>&emsp;
      <a href='https://yangxue0827.github.io/' style='text-decoration: none' >Xue Yang</a><sup></sup>&emsp;
      <a href='https://scholar.google.com/citations?user=OYtSc4AAAAAJ&hl=en' style='text-decoration: none' >Yi Yu</a><sup></sup>&emsp; 
      <a href='https://scholar.google.com/citations?hl=en&user=TvsTun4AAAAJ' style='text-decoration: none' >Qingyun Li</a><sup></sup>&emsp;   
      <a href='https://scholar.google.com.hk/citations?user=v-aQ8GsAAAAJ&hl=zh-CN' style='text-decoration: none' >Yue Zhou</a><sup></sup>&emsp;
      <a href='https://jiaxiaosong1002.github.io/' style='text-decoration: none' >Xiaosong Jia</a><sup></sup>&emsp;
      <a href='https://scholar.google.com/citations?user=G9jWIggAAAAJ&hl=en' style='text-decoration: none' >Xudong Lu</a><sup></sup>&emsp;
      <a href='https://scholar.google.com/citations?user=8SCEv-YAAAAJ&hl=en' style='text-decoration: none' >Jingdong Chen</a><sup></sup>&emsp;
      <a href='https://scholar.google.com/citations?user=oamjJdYAAAAJ&hl=zh-CN' style='text-decoration: none' >Xiang Li</a><sup></sup>&emsp;
      <a href='https://scholar.google.com/citations?user=ga230VoAAAAJ&hl=en' style='text-decoration: none' >Junchi Yan</a><sup></sup>&emsp;
      <a href='https://scholar.google.com/citations?user=wn9hc6UAAAAJ&hl=zh-CN' style='text-decoration: none' >Yansheng Li</a><sup></sup>&emsp;      
      <div align="center">
      <a href='https://arxiv.org/abs/2501.13898'><img src='https://img.shields.io/badge/arXiv-2501.09720-brown.svg?logo=arxiv&logoColor=white'></a>
	  </div>
    <p align='center'>
        If you find our work helpful, please consider giving us a ⭐!
    </p>
   </p>
</p>

The paper is available at [PointOBB-v3](https://arxiv.org/abs/2501.13898). You are also welcome to check out the conference version [PointOBB](https://openaccess.thecvf.com/content/CVPR2024/html/Luo_PointOBB_Learning_Oriented_Object_Detection_via_Single_Point_Supervision_CVPR_2024_paper.html).

**📌 Note: This branch contains the code for the end-to-end version. For the two-stage version, please refer to [`two-stage`](https://github.com/VisionXLab/PointOBB-v3/tree/end_to_end) branch.**

<img width="989" alt="image" src="https://github.com/user-attachments/assets/e320a8ce-6c98-438b-9b92-0c922536b5ab" />

### Train/Test
Please see [`PointOBB/README.md`](PointOBB/README.md).

### Weight

DIOR-R

|         Backbone         |  mAP  | Angle |  Config | Detector |                                                                                                                                                                              Download                                                                                                                                                                              |
| :----------------------: | :---: | :---: | :---:  | :------: |  :------------------------------------------------------------------------------------------------------------: |
| ResNet50 (1024,1024,200) | 37.60 | le90  | [pointobbv3-dior](PointOBB/configs2/pointobb/pointobbv3_r50_fpn_2x_dior_e2e.py)|    PointOBBv3_e2e  |  [model](https://drive.google.com/file/d/14WGrXK98J9hNchSb_bp5dODFeL20Mn0E/view?usp=sharing) |


DOTA-v1.0

|         Backbone         |  mAP  | Angle |  Config | Detector |                                                                                                                                                                              Download                                                                                                                                                                              |
| :----------------------: | :---: | :---: | :-----: | :------: |  :------------------------------------------------------------------------------------------------------------: |
| ResNet50 (1024,1024,200) | 41.29 | le90  | [pointobbv3-dota](PointOBB/configs2/pointobb/pointobbv3_r50_fpn_2x_dota_e2e.py)|    PointOBBv3_e2e |  [model](https://drive.google.com/file/d/1GtsQKRIWqf-3St9RyCcl1kJ98Ve6W-sR/view?usp=sharing) |


DOTA-v1.5

|         Backbone         |  mAP  | Angle |  Config | Detector |                                                                                                                                                                              Download                                                                                                                                                                              |
| :----------------------: | :---: | :---: | :-----: | :------: |  :------------------------------------------------------------------------------------------------------------: |
| ResNet50 (1024,1024,200) | 31.25 | le90  | [pointobbv3-dota15](PointOBB/configs2/pointobb/pointobbv3_r50_fpn_2x_dota15_e2e.py)|    PointOBBv3_e2e |  [model](https://drive.google.com/file/d/1w7MVj4lYJnx8TlHYQQGX9I3pETNgW4Cx/view?usp=sharing) |


DOTA-v2.0

|         Backbone         |  mAP  | Angle |  Config | Detector |                                                                                                                                                                              Download                                                                                                                                                                              |
| :----------------------: | :---: | :---: | :-----: | :------: |  :------------------------------------------------------------------------------------------------------------: |
| ResNet50 (1024,1024,200) | 22.82 | le90  | [pointobbv3-dota20](PointOBB/configs2/pointobb/pointobbv3_r50_fpn_2x_dota20_e2e.py)|    PointOBBv3_e2e |  [model](https://drive.google.com/file/d/1Ac91PTE9WkVYcAXjvynI29pL1irGJWgQ/view?usp=sharing) |

FAIR1M

|         Backbone         |  mAP  | Angle |  Config | Detector |                                                                                                                                                                              Download                                                                                                                                                                              |
| :----------------------: | :---: | :---: | :-----: | :------: |  :------------------------------------------------------------------------------------------------------------: |
| ResNet50 (1024,1024,200) | 11.42 | le90  | [pointobbv3-fair](PointOBB/configs2/pointobb/pointobbv3_r50_fpn_2x_fair_e2e.py)|    PointOBBv3_e2e |  [model](https://drive.google.com/file/d/15no3FJ_7JBHeCBbICMpljBi_fd3jovAo/view?usp=sharing) |


STAR

|         Backbone         |  mAP  | Angle |  Config | Detector |                                                                                                                                                                              Download                                                                                                                                                                              |
| :----------------------: | :---: | :---: | :-----: | :------: |  :------------------------------------------------------------------------------------------------------------: |
| ResNet50 (1024,1024,200) | 11.31 | le90  | [pointobbv3-star](PointOBB/configs2/pointobb/pointobbv3_r50_fpn_2x_star_e2e.py)|    PointOBBv3_e2e |  [model](https://drive.google.com/file/d/1E-9WIBUq8NQd685f5jVRlhu9KVT3I6QK/view?usp=sharing) |


RSAR

|         Backbone         |  mAP  | Angle |  Config | Detector |                                                                                                                                                                              Download                                                                                                                                                                              |
| :----------------------: | :---: | :---: | :-----: | :------: |  :------------------------------------------------------------------------------------------------------------: |
| ResNet50 (1024,1024,200) | 15.84 | le90  | [pointobbv3-rsar](PointOBB/configs2/pointobb/pointobbv3_r50_fpn_2x_rsar_e2e.py)|    PointOBBv3_e2e |  [model](https://drive.google.com/file/d/1nZ5RAIJP1WDDGxvWZTNo-fB8t2vOUarT/view?usp=sharing) |

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
