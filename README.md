# PWTF

Optical flow aims to measure pixel displacements between consecutive video frames, characterizing the continuous motion field within dynamic scenes. However, existing methods fail to simultaneously achieve robust long-range displacement estimation and precise fine-grained matching. To address this crucial trade-off issue, we propose PWTF, a novel method that Perceives Wide motion before Tracking Fine details. Specifically, PWTF first explores enhancing feature representation by three sets of cross-modal cues. Furthermore, PWTF introduces the Wide-Range Motion Perception module (WRMP), which utilizes Transformer to estimate pixel correlations and perceive large displacements from a global perspective. Finally, PWTF proposes the Fine-Grained Tracking (FGT) module, where FGT is an improved ConvGRU that utilizes potential global displacements provided by WRMP and is capable of tracking subtle displacements in high-dimensional features. Overall, these components collectively form a cohesive wide-to-fine pixel displacements measure architecture. Experiment results show that PWTF demonstrates more stable performance compared to the existing frameworks in terms of performing large-scale motion and detail tracking in the visualized comparison results. Additionally, PWTF achieves a new state-of-the-art in zero-shot generalization on the KITTI dataset.

<img src="PWTF.pdf">


## Requirements
Our code is developed with pytorch 2.7.0, CUDA 12.8 and python 3.12. 
```Shell
conda create --name waft python=3.12
conda activate waft
pip install -r requirements.txt
```

Please also install [xformers](https://github.com/facebookresearch/xformers) following instructions.

## Datasets
To evaluate/train WAFT, you will need to download the required datasets: [FlyingChairs](https://lmb.informatik.uni-freiburg.de/resources/datasets/FlyingChairs.en.html#flyingchairs), [FlyingThings3D](https://lmb.informatik.uni-freiburg.de/resources/datasets/SceneFlowDatasets.en.html), [Sintel](http://sintel.is.tue.mpg.de/), [KITTI](http://www.cvlibs.net/datasets/kitti/eval_scene_flow.php?benchmark=flow), [HD1K](http://hci-benchmark.iwr.uni-heidelberg.de/), [TartanAir](https://theairlab.org/tartanair-dataset/), and [Spring](https://spring-benchmark.org/). Please also check [SEA-RAFT](https://github.com/princeton-vl/SEA-RAFT) for more details.



## Acknowledgements

This project relies on code from existing repositories: [RAFT](https://github.com/princeton-vl/RAFT), [DPT](https://github.com/isl-org/DPT), [Flowformer](https://github.com/drinkingcoder/FlowFormer-Official), [ptlflow](https://github.com/hmorimitsu/ptlflow), [DepthAnythingV2](https://github.com/DepthAnything/Depth-Anything-V2), and [DINOv3](https://ai.meta.com/dinov3/). We thank the original authors for their excellent work.
