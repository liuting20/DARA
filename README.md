<div align=center>
  
# DARA: Domain- and Relation-aware Adapters Make Parameter-efficient Tuning for Visual Grounding

**[Ting Liu](https://github.com/liuting20), [Xuyang Liu](https://xuyang-liu16.github.io/), [Siteng Huang](https://kyonhuang.top/), [Honggang Chen](https://sites.google.com/view/honggangchen/), Quanjun Yin, Long Qin, [Donglin Wang](https://milab.westlake.edu.cn/), Yue Hu**

<p>
<a href='https://arxiv.org/pdf/2405.06217'><img src='https://img.shields.io/badge/Paper-arXiv-red'></a>
<a href='https://ieeexplore.ieee.org/document/10688132'><img src='https://img.shields.io/badge/Paper-ICME-blue'></a>
</p>

</div>

## :sparkles: Overview
<p align="center"> <img src="overview.png" width="1000" align="center"> </p>

In this paper, we explore applying parameter-efficient transfer learning (PETL) to efficiently transfer the pre-trained vision-language knowledge to VG. Specifically, we propose **DARA**, a novel PETL method comprising **D**omain-aware **A**dapters (DA Adapters) and **R**elation-aware **A**dapters (RA Adapters) for VG. DA Adapters first transfer intra-modality representations to be more fine-grained for the VG domain. Then RA Adapters share weights to bridge the relation between two modalities, improving spatial reasoning. Empirical results on widely-used benchmarks demonstrate that DARA achieves the best accuracy while saving numerous updated parameters compared to the full fine-tuning and other PETL methods. Notably, with only **2.13%** tunable backbone parameters, DARA improves average accuracy by **0.81%** across the three benchmarks compared to the baseline model. Note that the tunale parameters are lower than reported in the paper by optimization.


### :point_right: Installation
1.  Clone this repository.
    ```
    git clone https://github.com/liuting20/DARA.git
    ```

2.  Prepare for the running environment. 

    ```
     conda env create -f environment.yaml      pip install -r requirements.txt
    ```

### :point_right: Getting Started

Please refer to [GETTING_STARGTED.md](GETTING_STARTED.md) to learn how to prepare the datasets and pretrained checkpoints.


### :point_right: Training and Evaluation

1.  Training
    ```
    CUDA_VISIBLE_DEVICES=0 python -u train.py --batch_size 64 --lr_bert 0.00001 --aug_crop --aug_scale --aug_translate --backbone resnet50 --detr_model ./checkpoints/detr-r50-referit.pth --bert_enc_num 12 --detr_enc_num 6 --dataset unc --max_query_len 20 --output_dir outputs/referit_r50 --epochs 90 --lr_drop 60
    ```

    We recommend to set --max_query_len 40 for RefCOCOg, and --max_query_len 20 for other datasets. 
    
    We recommend to set --epochs 180 (--lr_drop 120 acoordingly) for RefCOCO+, and --epochs 90 (--lr_drop 60 acoordingly) for other datasets. 

2.  Evaluation
    ```
    CUDA_VISIBLE_DEVICES=0 python -u eval.py --batch_size 64 --num_workers 4 --bert_enc_num 12 --detr_enc_num 6 --backbone resnet50 --dataset unc --max_query_len 20 --eval_set testA --eval_model ./outputs/referit_r50/best_checkpoint.pth --output_dir ./outputs/referit_r50
    ```

### :thumbsup: Acknowledge
This codebase is partially based on [TransVG](https://github.com/djiajunustc/TransVG).


## :pushpin: Citation
Please consider citing our paper in your publications, if our findings help your research.
```bibtex
@misc{liu2024dara,
      title={{DARA}: Domain- and Relation-aware Adapters Make Parameter-efficient Tuning for Visual Grounding}, 
      author={Ting Liu and Xuyang Liu and Siteng Huang and Honggang Chen and Quanjun Yin and Long Qin and Donglin Wang and Yue Hu},
      year={2024},
      eprint={2405.06217},
      archivePrefix={arXiv},
      primaryClass={cs.CV}
}
```

## :e-mail: Contact
For any question about our paper or code, please contact [Ting Liu](mailto:liuting20@nudt.edu.cn) or [Xuyang Liu](mailto:liuxuyang@stu.scu.edu.cn).
