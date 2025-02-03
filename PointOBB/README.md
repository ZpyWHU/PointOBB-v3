Our GPUs: 2 * A100 (80GB)


# Prerequisites  &  Prepare Dataset
Please follow our conference version for details: [https://github.com/Luo-Z13/pointobb](https://github.com/Luo-Z13/pointobb)

# Train/Inference

1. **Train**

To train the model, follow these steps:
```bash
cd PointOBB
## train with single GPU, note adjust learning rate or batch size accordingly
# DIOR
python tools/train.py --config configs2/pointobb/pointobbv3_r50_fpn_2x_dior_two_stage.py --work-dir ./work_dir/pointobbv3_dior/ --cfg-options evaluation.save_result_file='./work_dir/pointobbv3_dior/pseudo_obb_result.json'

# DOTA
# python tools/train.py --config configs2/pointobb/pointobbv3_r50_fpn_2x_dota_two_stage.py --work-dir ./work_dir/pointobbv3_dior/ --cfg-options evaluation.save_result_file='./work_dir/pointobbv3_dior/pseudo_obb_result.json'

......
```



2. **Inference** (two-stage)
  
To inference (generate pseudo obb label), follow these steps:
```bash
# obtain COCO format pseudo label for the training set 
# (在训练集上推理,从单点生成旋转框的伪标签)
sh test_p.sh
# convert COCO format to DOTA format 
# (将伪标签从COCO格式转换为DOTA格式)
sh tools_cocorbox2dota.sh
# train standard oriented object detectors 
# (使用伪标签训练mmrotate里的标准旋转检测器)
# Please use algorithms in mmrotate (https://github.com/open-mmlab/mmrotate)
```






