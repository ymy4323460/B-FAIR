for dataset_id in nonlinearinvdata_25_16 nonlinearinvdata_50_16 # #nonliearlogitdata_10_32_0.5_0.5
    do
    for down in gmfBPR MLP
        do
#     pretrain step
        python3 train_baselines_syn_ndcg.py \
                --debias_mode Pretrain \
                --feature_data True \
                --dataset $dataset_id \
                --epoch_max 15 \
                --user_size 5000 \
                --item_size 32 \
                --user_item_size 5000 32 \
                --user_dim 32 \
                --item_dim 16  \
                --user_emb_dim 32 \
                --item_emb_dim 32 \
                --ipm_layer_dims 64 32 8 \
                --ctr_layer_dims 64 32 8

        python3 train_baselines_syn_ndcg.py \
                --debias_mode Pretrain \
                --pretrain_mode imputation \
                --feature_data True \
                --dataset $dataset_id \
                --epoch_max 15 \
                --user_size 5000 \
                --item_size 32 \
                --user_item_size 5000 32 \
                --user_dim 32 \
                --item_dim 16  \
                --user_emb_dim 32 \
                --item_emb_dim 32 \
                --ipm_layer_dims 64 32 8 \
                --ctr_layer_dims 64 32 8


        python3 train_baselines_syn_ndcg.py \
                --debias_mode Pretrain \
                --pretrain_mode uniform_imputation \
                --feature_data True \
                --dataset $dataset_id \
                --epoch_max 15 \
                --user_size 5000 \
                --item_size 32 \
                --user_item_size 5000 32 \
                --user_dim 32 \
                --item_dim 16  \
                --user_emb_dim 32 \
                --item_emb_dim 32 \
                --ipm_layer_dims 64 32 8 \
                --ctr_layer_dims 64 32 8 \


        python3 train_baselines_syn_ndcg.py \
                --train_mode train \
                --debias_mode ATT \
                --feature_data True \
                --dataset $dataset_id \
                --downstream $down \
                --use_weight True \
                --epoch_max 100 \
                --user_size 5000 \
                --item_size 32 \
                --user_item_size 5000 32 \
                --user_dim 32 \
                --item_dim 16  \
                --user_emb_dim 32 \
                --item_emb_dim 32 \
                --ipm_layer_dims 64 32 8 \
                --ctr_layer_dims 64 32 8

        python3 train_baselines_syn_ndcg.py \
                --train_mode train \
                --debias_mode CVIB \
                --feature_data True \
                --dataset $dataset_id \
                --downstream $down \
                --use_weight True \
                --epoch_max 100 \
                --user_size 5000 \
                --item_size 32 \
                --user_item_size 5000 32 \
                --user_dim 32 \
                --item_dim 16  \
                --user_emb_dim 32 \
                --item_emb_dim 32 \
                --ipm_layer_dims 64 32 8 \
                --ctr_layer_dims 64 32 8

        python3 train_baselines_syn_ndcg.py \
                --train_mode train \
                --debias_mode Propensitylearnt_Mode \
                --feature_data True \
                --dataset $dataset_id \
                --downstream $down \
                --use_weight True \
                --epoch_max 100 \
                --user_size 5000 \
                --item_size 32 \
                --user_item_size 5000 32 \
                --user_dim 32 \
                --item_dim 16  \
                --user_emb_dim 32 \
                --item_emb_dim 32 \
                --ipm_layer_dims 64 32 8 \
                --ctr_layer_dims 64 32 8

        python3 train_baselines_syn_ndcg.py \
                --train_mode train \
                --debias_mode SNIPSlearnt_Mode \
                --feature_data True \
                --dataset $dataset_id \
                --downstream $down \
                --use_weight True \
                --epoch_max 100 \
                --user_size 5000 \
                --item_size 32 \
                --user_item_size 5000 32 \
                --user_dim 32 \
                --item_dim 16  \
                --user_emb_dim 32 \
                --item_emb_dim 32 \
                --ipm_layer_dims 64 32 8 \
                --ctr_layer_dims 64 32 8

        python3 train_baselines_syn_ndcg.py \
                --train_mode train \
                --debias_mode Direct \
                --feature_data True \
                --dataset $dataset_id \
                --downstream $down \
                --use_weight True \
                --epoch_max 100 \
                --user_size 5000 \
                --item_size 32 \
                --user_item_size 5000 32 \
                --user_dim 32 \
                --item_dim 16  \
                --user_emb_dim 32 \
                --item_emb_dim 32 \
                --ipm_layer_dims 64 32 8 \
                --ctr_layer_dims 64 32 8

        python3 train_baselines_syn_ndcg.py \
                --train_mode train \
                --debias_mode Propensity_DR_Mode \
                --feature_data True \
                --dataset $dataset_id \
                --downstream $down \
                --use_weight True \
                --epoch_max 100 \
                --user_size 5000 \
                --item_size 32 \
                --user_item_size 5000 32 \
                --user_dim 32 \
                --item_dim 16  \
                --user_emb_dim 32 \
                --item_emb_dim 32 \
                --ipm_layer_dims 64 32 8 \
                --ctr_layer_dims 64 32 8
    done
done
