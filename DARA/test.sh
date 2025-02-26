export CUDA_VISIBLE_DEVICES=0,1,2,3



# # RefCOCO
## single gpu
CUDA_VISIBLE_DEVICES=0 python -u eval.py --batch_size 64 --bert_enc_num 12 --detr_enc_num 6 --backbone resnet50 --dataset unc --max_query_len 20 --eval_set testA --eval_model ./outputs/refcoco/best_checkpoint.pth --output_dir ./outputs_test/coco_r50_A
## distribute
python -m torch.distributed.launch --nproc_per_node=4 --use_env eval.py --batch_size 16 --num_workers 4 --bert_enc_num 12 --detr_enc_num 6 --backbone resnet50 --dataset unc --max_query_len 20 --eval_set testB --eval_model ./outputs/refcoco/best_checkpoint.pth --output_dir ./outputs_test/coco_r50_B


## RefCOCO+
## single gpu
CUDA_VISIBLE_DEVICES=0 python -u eval.py --batch_size 64 --bert_enc_num 12 --detr_enc_num 6 --backbone resnet50 --dataset unc+ --max_query_len 20 --eval_set testA --eval_model ./outputs/coco+/best_checkpoint.pth --output_dir ./outputs_test/coco+_r50_A
## distribute
#python -m torch.distributed.launch --nproc_per_node=4 --use_env eval.py --batch_size 16 --num_workers 4 --bert_enc_num 12 --detr_enc_num 6 --backbone resnet50 --dataset unc+ --max_query_len 20 --eval_set testB --eval_model ./outputs/coco+/best_checkpoint.pth --output_dir ./outputs_test/coco+_r50_B


# # RefCOCOg g-split
## single gpu
CUDA_VISIBLE_DEVICES=0 python -u eval.py --batch_size 64 --bert_enc_num 12 --detr_enc_num 6 --backbone resnet50 --dataset gref_umd --max_query_len 40 --eval_set test --eval_model ./outputs/refcocog_u/best_checkpoint.pth --output_dir ./outputs_test/refcocog_usplit_r50
## distribute
python -m torch.distributed.launch --nproc_per_node=4 --use_env eval.py --batch_size 16 --num_workers 4 --bert_enc_num 12 --detr_enc_num 6 --backbone resnet50 --dataset gref --max_query_len 40 --eval_set val --eval_model ./outputs/refcocog_u/best_checkpoint.pth --output_dir ./outputs/refcocog_gsplit_r50


# # RefCOCOg u-split
## single gpu
CUDA_VISIBLE_DEVICES=0 python -u eval.py --batch_size 64 --bert_enc_num 12 --detr_enc_num 6 --backbone resnet50 --dataset gref_umd --max_query_len 40 --eval_set test --eval_model ./outputs/refcocog_u/best_checkpoint.pth --output_dir ./outputs_test/refcocog_usplit_r50
## distribute
python -m torch.distributed.launch --nproc_per_node=4 --use_env eval.py --batch_size 16 --num_workers 4 --bert_enc_num 12 --detr_enc_num 6 --backbone resnet50 --dataset gref_umd --max_query_len 40 --eval_set test --eval_model ./outputs/refcocog_u/best_checkpoint.pth --output_dir ./outputs/refcocog_usplit_r50

