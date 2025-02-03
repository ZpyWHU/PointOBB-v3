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

2. **Inference** 

To inference, follow these steps:
```bash
## Don't forget to change the dataset_type in config when testing on DOTA-v1.0/v1.5/v2.0 or FAIR1M or STAR !!!
# DIOR
python tools/test.py configs2/pointobb/pointobbv3_r50_fpn_2x_dior_e2e.py ./work_dir/pointobbv3/dior/epoch_24.pth --eval mAP

# DOTA
# python tools/test.py configs2/pointobb/pointobbv3_r50_fpn_2x_dota_e2e.py ./work_dir/pointobbv3/dota10/epoch_24.pth --format-only --eval-options submission_dir='./Task1/'

......
```






