# HAFDec
[PRCV] HAFDec: Lightweight Self-Supervised Monocular Depth Estimation with Hierarchical Adaptive Fusion Decoder (2.5M params, 5.5 GFLOPs), achieving Abs Rel 0.107 on KITTI and strong zero-shot generalization on Make3D.

# Overview
<img width="1429" height="747" alt="image" src="https://github.com/user-attachments/assets/ca49559e-c09d-4642-81ad-784fbd740843" />

# KITTI Test Results
You can download the trained models using the links below.

<img width="1247" height="1003" alt="image" src="https://github.com/user-attachments/assets/f72d1780-39ca-4f74-945e-308eb7ca0afa" />

| Model             | Params | ImageNet Pretrained | Input size | Abs Rel | Sq Rel | RMSE  | RMSE log | delta < 1.25 | delta < 1.25² | delta < 1.25³ |
|-------------------|--------|---------------------|------------|---------|--------|-------|----------|--------------|---------------|---------------|
| [HAFDec(pretrain)](通过网盘分享的文件：pretrain
链接: https://pan.baidu.com/s/119yF3WndSO6hoqXpFuMCZQ?pwd=w6hd 提取码: w6hd)  |  2.5M  | yes                 | 640x192    | 0.107   | 0.774  | 4.580 | 0.183    | 0.886        | 0.962        | 0.983         |
| [HAFDec](通过网盘分享的文件：train from scratch
链接: https://pan.baidu.com/s/18U0ECsqa4ZJ9V0sfaE0x0A?pwd=uctw 提取码: uctw)            | 2.5M   | no               | 640x192    | 0.122  | 0.926  | 4.846| 0.198   | 0.861       | 0.955        | 0.980        |
