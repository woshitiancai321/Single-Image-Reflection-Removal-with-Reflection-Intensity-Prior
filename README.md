# Paper
Single Image Reflection Removal with Reflection Intensity Prior [Paper](https://arxiv.org/abs/2312.03798).
# Installation
Please refer [github](https://github.com/Janspiry/Image-Super-Resolution-via-Iterative-Refinement)
# Test/Evaluation
```python
python sr.py -p val -c config/REAL.json -data_path ./data/reflect/real20
python sr.py -p val -c config/CDR.json -data_path ./data/reflect/CDR/strong/
python sr.py -p val -c config/IBCLN.json -data_path ./data/reflect/IBCLN/test/
```
Kindly update the path corresponding to the "resume_state" keyword in the .json file to the downloaded checkpoint, for instance
```json
"resume_state":"/home/percv-d10/handongshen/git/han/github/image-resolution-unet/ckp/CDR".
```
# Checkpoint
Checkpoint can get from .
Piror checkpoint is in prior_ckp file. Other is in ckp.
You can obtain the checkpoint file from [checkpoint](https://drive.google.com/file/d/1cDLpUMpvChAsdr0V6SEBVao4RenIy3YD/view?usp=sharing). The previous checkpoint is stored in the "prior_ckp" file, while others can be found in the "ckp" file.
请将两个文件夹放在当前目录之下


The checkpoint can be obtained from [checkpoint](https://drive.google.com/file/d/1cDLpUMpvChAsdr0V6SEBVao4RenIy3YD/view?usp=sharing). Place the downloaded checkpoint file in the current directory.

Please ensure that two folders are present in the current directory: "prior_ckp" containing the prior knowledge checkpoint and "ckp" containing the other checkpoints.
