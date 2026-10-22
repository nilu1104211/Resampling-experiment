#!/bin/bash
export BASE_PATH=$(pwd)
export PYTHONPATH=$BASE_PATH
######## noise3 pretrain
data_dir="../self_sample_data/my_data/PU1K_raw_meshes/train/train_gt_pc/600" # 9988 points
result_dir="_result/ablation/600"

mode_list=("denoising")


for mode in "${mode_list[@]}"; do
	for pc_file in $data_dir/*.xyz; do
		if [ -f "$pc_file" ]; then
			pc_name=`basename "$pc_file" .xyz`
			# pc_file="$data_dir/test/input_2048/input_2048/$pc_name.xyz" # 2048
			pc_file="$data_dir/$pc_name.xyz" # 40,000

			# --- Pretrained Method
				python -u $BASE_PATH/multi_pretrain_main.py \
				--do-pretrain False \
				--pretrain-lr 0.0005 \
				--pretrain-iter 10010 \
				--pretrain-export-interval 600 \
				--target-pretrain-weight 1200 \
				--lr 0.0005 \
				--iterations 10010 \
				--export-interval 50 \
				--pc "$pc_file" \
				--init-var 0.15 \
				--D1 2497 --D2 2497 \
				--save-path "$result_dir" \
				--sampling-mode $mode \
				--batch-size 2 \
				--k 10 \
				--p1 0.85 --p2 0.2 \
				--force-normal-estimation \
				--mse
		fi
	done
done
: << 'CHECK_PRETRAIN_INFERENCE'
python -u $BASE_PATH/inference.py \
--lr 0.001 --name pretrain_result \
--iterations 16 \
--export-interval 100 \
--pc "$pc_file" \
--init-var 0.15 \
--D1 2497 --D2 2497 \
--save-path "$result_dir/finetune1/$pc_name/$mode" \
--generator "$result_dir/finetune1/$pc_name/$mode/generators/pretrain_model10.pt" \
--sampling-mode $mode \
--batch-size 8 \
--k 10 \
--force-normal-estimation \
--mse
CHECK_PRETRAIN_INFERENCE
# ---------------------------

