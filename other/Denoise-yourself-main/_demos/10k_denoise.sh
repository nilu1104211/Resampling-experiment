#### noise3
#!/bin/bash
export BASE_PATH=$(pwd)
export PYTHONPATH=$BASE_PATH
# 10k -> 40k
# 디렉토리 경로 설정
data_dir="/home/jihyeon/self_sample_data/my_data/PU1K_raw_meshes/sampling/10000"
result_dir="_result/pu1k_10k/noise3"

mode="density"

for pc_file in $data_dir/*.xyz; do
	if [ -f "$pc_file" ]; then
		pc_name=`basename "$pc_file" .xyz`
		# echo $pc_name
		noise_file="$data_dir/noise/0.002/$pc_name.xyz"
		# --- Pretrained Method
		python -u $BASE_PATH/denoise_pretrain_finetune_main.py \
		--do-pretrain True \
		--pretrain-lr 0.0005 \
		--pretrain-iter 10010 \
		--pretrain-export-interval 600 \
		--target-pretrain-weight 1200 \
		--lr 0.0005 \
		--iterations 10010 \
		--export-interval 600 \
		--pc "$pc_file" \
		--noise-pc "$noise_file" \
		--init-var 0.15 \
		--D1 2497 --D2 2497 \
		--save-path "$result_dir/$pc_name/$mode" \
		--sampling-mode $mode \
		--batch-size 8 \
		--k 10 \
		--p1 0.85 --p2 0.2 \
		--force-normal-estimation \
		--mse

		python -u $BASE_PATH/inference.py \
		--lr 0.001 --name result \
		--iterations 16 \
		--export-interval 100 \
		--pc "$pc_file" \
		--init-var 0.15 \
		--D1 2497 --D2 2497 \
		--save-path "$result_dir/$pc_name/$mode" \
		--generator "$result_dir/$pc_name/$mode/generators/model1200.pt" \
		--sampling-mode $mode \
		--batch-size 8 \
		--k 10 \
		--force-normal-estimation \
		--mse
	fi
done