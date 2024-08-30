export CUDA_VISIBLE_DEVICES=4,5,6,7


## RefCOCO
## single gpu
# CUDA_VISIBLE_DEVICES=7 python -u train.py --batch_size 64 --lr_bert 0.00001 --aug_crop --aug_scale --aug_translate --backbone resnet50 --detr_model /share/home/liuting/transvg_data/checkpoints/detr-r50-unc.pth --bert_enc_num 12 --detr_enc_num 6 --dataset unc --max_query_len 20 --output_dir outputs/refcoco --data_root /share/home/liuting/transvg_data/ln_data --split_root /share/home/liuting/transvg_data/data
## distribute
# python -m torch.distributed.launch --nproc_per_node=4 --use_env train.py --batch_size 16 --lr_bert 0.00001 --aug_crop --aug_scale --aug_translate --backbone resnet50 --detr_model /share/home/liuting/transvg_data/checkpoints/detr-r50-unc.pth --bert_enc_num 12 --detr_enc_num 6 --dataset unc --max_query_len 20 --output_dir outputs/refcoco --data_root /share/home/liuting/transvg_data/ln_data --split_root /share/home/liuting/transvg_data/data


## RefCOCO+
## single gpu
# CUDA_VISIBLE_DEVICES=7 python -u train.py --batch_size 64 --lr_bert 0.00001 --aug_crop --aug_scale --aug_translate --backbone resnet50 --detr_model /share/home/liuting/transvg_data/checkpoints/detr-r50-unc.pth --bert_enc_num 12 --detr_enc_num 6 --dataset unc+ --max_query_len 20 --output_dir outputs/coco+ --epochs 180 --lr_drop 120 --data_root /share/home/liuting/transvg_data/ln_data --split_root /share/home/liuting/transvg_data/data
## distribute
# python -m torch.distributed.launch --nproc_per_node=4 --use_env train.py --batch_size 16 --lr_bert 0.00001 --aug_crop --aug_scale --aug_translate --backbone resnet50 --detr_model /share/home/liuting/transvg_data/checkpoints/detr-r50-unc.pth --bert_enc_num 12 --detr_enc_num 6 --dataset unc+ --max_query_len 20 --output_dir outputs/coco+ --epochs 180 --lr_drop 120 --data_root /share/home/liuting/transvg_data/ln_data --split_root /share/home/liuting/transvg_data/data


## RefCOCOg g-split
## single gpu
# CUDA_VISIBLE_DEVICES=7 python -u train.py --batch_size 64 --lr_bert 0.00001 --aug_crop --aug_scale --aug_translate --backbone resnet50 --detr_model /share/home/liuting/transvg_data/checkpoints/detr-r50-gref.pth --bert_enc_num 12 --detr_enc_num 6 --dataset gref --max_query_len 40 --output_dir outputs/cocog-g --data_root /share/home/liuting/transvg_data/ln_data --split_root /share/home/liuting/transvg_data/data
## distribute
# python -m torch.distributed.launch --nproc_per_node=4 --use_env train.py --batch_size 16 --lr_bert 0.00001 --aug_crop --aug_scale --aug_translate --backbone resnet50 --detr_model /share/home/liuting/transvg_data/checkpoints/detr-r50-gref.pth --bert_enc_num 12 --detr_enc_num 6 --dataset gref --max_query_len 40 --output_dir outputs/cocog-g --data_root /share/home/liuting/transvg_data/ln_data --split_root /share/home/liuting/transvg_data/data


## RefCOCOg umd-split
## single gpu
CUDA_VISIBLE_DEVICES=7 python -u train.py --batch_size 64 --lr_bert 0.00001 --aug_crop --aug_scale --aug_translate --backbone resnet50 --detr_model /share/home/liuting/transvg_data/checkpoints/detr-r50-gref.pth --bert_enc_num 12 --detr_enc_num 6 --dataset gref_umd --max_query_len 40 --output_dir outputs/cocog-u --data_root /share/home/liuting/transvg_data/ln_data --split_root /share/home/liuting/transvg_data/data
## distribute
# python -m torch.distributed.launch --nproc_per_node=4 --use_env train.py --batch_size 16 --lr_bert 0.00001 --aug_scale --aug_translate --aug_crop --backbone resnet50 --detr_model /share/home/liuting/transvg_data/checkpoints/detr-r50-gref.pth --bert_enc_num 12 --detr_enc_num 6 --dataset gref_umd --max_query_len 40 --output_dir outputs/refcocog_u --data_root /share/home/liuting/transvg_data/ln_data --split_root /share/home/liuting/transvg_data/data