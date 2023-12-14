for down in mlpBPR NeuBPR gmfBPR MLP
    do
#    python3 train_baselines_pcic.py --epoch_max 20 --debias_mode Pretrain --user_emb_dim 32 --item_emb_dim 32 --ctr_layer_dims 64 32 8
#    python3 train_baselines_pcic.py --epoch_max 20 --pretrain_mode imputation --debias_mode Pretrain --user_emb_dim 32 --item_emb_dim 32 --ctr_layer_dims 64 32 8
#    python3 train_baselines_pcic.py --epoch_max 20 --pretrain_mode uniform_imputation --debias_mode Pretrain --user_emb_dim 32 --item_emb_dim 32 --ctr_layer_dims 64 32 8
#
#    python3 train_baselines_pcic.py --use_weight True --epoch_max 100 --train_mode train --debias_mode ATT --user_emb_dim 32 --item_emb_dim 32 --ctr_layer_dims 64 32 8 --downstream $down
#    python3 train_baselines_pcic.py --use_weight True --epoch_max 100 --train_mode train --debias_mode CVIB --user_emb_dim 32 --item_emb_dim 32 --ctr_layer_dims 64 32 8 --downstream $down
#
#    python3 train_baselines_pcic.py --use_weight True --epoch_max 100 --train_mode train --debias_mode Propensitylearnt_Mode --user_emb_dim 32 --item_emb_dim 32 --ctr_layer_dims 64 32 8 --downstream $down
#    python3 train_baselines_pcic.py --use_weight True --epoch_max 100 --train_mode train --debias_mode SNIPSlearnt_Mode --user_emb_dim 32 --item_emb_dim 32 --ctr_layer_dims 64 32 8 --downstream $down
#
#    python3 train_baselines_pcic.py --use_weight True --epoch_max 100 --train_mode train --debias_mode Direct --user_emb_dim 32 --item_emb_dim 32 --ctr_layer_dims 64 32 8 --downstream $down
#    python3 train_baselines_pcic.py --use_weight True --epoch_max 100 --train_mode train --debias_mode Propensity_DR_Mode --user_emb_dim 32 --item_emb_dim 32 --ctr_layer_dims 64 32 8 --downstream $down
#
#    python3 train_baselines_pcic.py --use_weight True --epoch_max 100 --train_mode train --debias_mode Uniform_DR_Mode --user_emb_dim 32 --item_emb_dim 32 --ctr_layer_dims 64 32 8 --downstream $down

#
#    python3 train_baselines.py --epoch_max 20 --debias_mode Pretrain --user_emb_dim 32 --item_emb_dim 32 --ctr_layer_dims 64 32 8 --user_size 15400 --item_size 1000 --user_item_size 15400 1000 --dataset yahoo
#    python3 train_baselines.py --epoch_max 20 --pretrain_mode imputation --debias_mode Pretrain --user_emb_dim 32 --item_emb_dim 32 --ctr_layer_dims 64 32 8 --user_size 15400 --item_size 1000 --user_item_size 15400 1000 --dataset yahoo
#    python3 train_baselines.py --epoch_max 20 --pretrain_mode uniform_imputation --debias_mode Pretrain --user_emb_dim 32 --item_emb_dim 32 --ctr_layer_dims 64 32 8 --user_size 15400 --item_size 1000 --user_item_size 15400 1000 --dataset yahoo
#
#    python3 train_baselines.py --use_weight True --epoch_max 100 --train_mode train --debias_mode ATT --user_emb_dim 32 --item_emb_dim 32 --ctr_layer_dims 64 32 8 --downstream $down --user_size 15400 --item_size 1000 --user_item_size 15400 1000 --dataset yahoo
#    python3 train_baselines.py --use_weight True --epoch_max 100 --train_mode train --debias_mode CVIB --user_emb_dim 32 --item_emb_dim 32 --ctr_layer_dims 64 32 8 --downstream $down --user_size 15400 --item_size 1000 --user_item_size 15400 1000 --dataset yahoo
#
#    python3 train_baselines.py --use_weight True --epoch_max 100 --train_mode train --debias_mode Propensitylearnt_Mode --user_emb_dim 32 --item_emb_dim 32 --ctr_layer_dims 64 32 8 --downstream $down --user_size 15400 --item_size 1000 --user_item_size 15400 1000 --dataset yahoo
#    python3 train_baselines.py --use_weight True --epoch_max 100 --train_mode train --debias_mode SNIPSlearnt_Mode --user_emb_dim 32 --item_emb_dim 32 --ctr_layer_dims 64 32 8 --downstream $down --user_size 15400 --item_size 1000 --user_item_size 15400 1000 --dataset yahoo
#
#    python3 train_baselines.py --use_weight True --epoch_max 100 --train_mode train --debias_mode Direct --user_emb_dim 32 --item_emb_dim 32 --ctr_layer_dims 64 32 8 --downstream $down --user_size 15400 --item_size 1000 --user_item_size 15400 1000 --dataset yahoo
#    python3 train_baselines.py --use_weight True --epoch_max 100 --train_mode train --debias_mode Propensity_DR_Mode --user_emb_dim 32 --item_emb_dim 32 --ctr_layer_dims 64 32 8 --downstream $down --user_size 15400 --item_size 1000 --user_item_size 15400 1000 --dataset yahoo
#
#    python3 train_baselines.py --use_weight True --epoch_max 100 --train_mode train --debias_mode Uniform_DR_Mode --user_emb_dim 32 --item_emb_dim 32 --ctr_layer_dims 64 32 8 --downstream $down --user_size 15400 --item_size 1000 --user_item_size 15400 1000 --dataset yahoo
#
#    python3 train_baselines.py --epoch_max 20 --debias_mode Pretrain --feature_data True --dataset coat --user_dim 14 --item_dim 33 --user_emb_dim 32 --item_emb_dim 32 --ipm_layer_dims 64 32 8 --ctr_layer_dims 64 32 8 --user_size 290 --item_size 300
#    python3 train_baselines.py --epoch_max 20 --debias_mode Pretrain --pretrain_mode imputation --feature_data True --dataset coat --user_dim 14 --item_dim 33 --user_emb_dim 32 --item_emb_dim 32 --ipm_layer_dims 64 32 8 --ctr_layer_dims 64 32 8 --user_size 290 --item_size 300
#    python3 train_baselines.py --epoch_max 20 --debias_mode Pretrain --pretrain_mode uniform_imputation --feature_data True --dataset coat --user_dim 14 --item_dim 33 --user_emb_dim 32 --item_emb_dim 32 --ipm_layer_dims 64 32 8 --ctr_layer_dims 64 32 8 --user_size 290 --item_size 300
#
#    python3 train_baselines_coat.py --use_weight True  --epoch_max 100 --train_mode train --debias_mode ATT --feature_data True --dataset coat --user_dim 14 --item_dim 33 --user_emb_dim 32 --item_emb_dim 32 --ipm_layer_dims 64 32 8 --ctr_layer_dims 64 32 8 --user_size 290 --item_size 300 --user_item_size 290 300 --downstream $down
#    python3 train_baselines_coat.py --use_weight True  --epoch_max 100 --train_mode train --debias_mode CVIB --feature_data True --dataset coat --user_dim 14 --item_dim 33  --user_emb_dim 32 --item_emb_dim 32 --ipm_layer_dims 64 32 8 --ctr_layer_dims 64 32 8 --user_size 290 --item_size 300  --user_item_size 290 300 --downstream $down
#
#
#    python3 train_baselines_coat.py --use_weight True  --epoch_max 100 --train_mode train --debias_mode Propensitylearnt_Mode --feature_data True --dataset coat --user_dim 14 --item_dim 33 --user_emb_dim 32 --item_emb_dim 32 --ipm_layer_dims 64 32 8 --ctr_layer_dims 64 32 8 --user_size 290 --item_size 300 --user_item_size 290 300 --downstream $down
#    python3 train_baselines_coat.py --use_weight True  --epoch_max 100 --train_mode train --debias_mode SNIPSlearnt_Mode --feature_data True --dataset coat --user_dim 14 --item_dim 33  --user_emb_dim 32 --item_emb_dim 32 --ipm_layer_dims 64 32 8 --ctr_layer_dims 64 32 8 --user_size 290 --item_size 300  --user_item_size 290 300 --downstream $down
#
#    python3 train_baselines_coat.py --use_weight True  --epoch_max 100 --train_mode train --debias_mode Direct --feature_data True --dataset coat --user_dim 14 --item_dim 33  --user_emb_dim 32 --item_emb_dim 32 --ipm_layer_dims 64 32 8 --ctr_layer_dims 64 32 8 --user_size 290 --item_size 300  --user_item_size 290 300 --downstream $down
#    python3 train_baselines_coat.py --use_weight True  --epoch_max 100 --train_mode train --debias_mode Propensity_DR_Mode --feature_data True --dataset coat --user_dim 14 --item_dim 33  --user_emb_dim 32 --item_emb_dim 32 --ipm_layer_dims 64 32 8 --ctr_layer_dims 64 32 8 --user_size 290 --item_size 300 --user_item_size 290 300  --downstream $down
#
#    python3 train_baselines_coat.py --use_weight True  --epoch_max 100 --train_mode train --debias_mode Uniform_DR_Mode --feature_data True --dataset coat --user_dim 14 --item_dim 33  --user_emb_dim 32 --item_emb_dim 32 --ipm_layer_dims 64 32 8 --ctr_layer_dims 64 32 8 --user_size 290 --item_size 300 --user_item_size 290 300  --downstream $down


# ACL
#    python3 train_baselines_pcic.py --epoch_max 20 --pretrain_mode imputation --debias_mode Pretrain --user_emb_dim 32 --item_emb_dim 32 --ctr_layer_dims 64 32 8
#
#    python3 train_baselines_pcic.py --use_weight True --epoch_max 100 --train_mode train --debias_mode ACL --user_emb_dim 32 --item_emb_dim 32 --ctr_layer_dims 64 32 8 --downstream $down
#
#    python3 train_baselines.py --epoch_max 20 --pretrain_mode uniform_imputation --debias_mode Pretrain --user_emb_dim 32 --item_emb_dim 32 --ctr_layer_dims 64 32 8 --user_size 15400 --item_size 1000 --user_item_size 15400 1000 --dataset yahoo
#
#    python3 train_baselines.py --use_weight True --epoch_max 100 --train_mode train --debias_mode ACL --user_emb_dim 32 --item_emb_dim 32 --ctr_layer_dims 64 32 8 --downstream $down --user_size 15400 --item_size 1000 --user_item_size 15400 1000 --dataset yahoo
#
#    python3 train_baselines.py --epoch_max 20 --debias_mode Pretrain --pretrain_mode imputation --feature_data True --dataset coat --user_dim 47 --item_dim 33 --user_emb_dim 32 --item_emb_dim 32 --ipm_layer_dims 64 32 8 --ctr_layer_dims 64 32 8 --user_size 290 --item_size 300
#
#    python3 train_baselines_coat.py --use_weight True  --epoch_max 100 --train_mode train --debias_mode ACL --feature_data True --dataset coat --user_dim 47 --item_dim 33 --user_emb_dim 32 --item_emb_dim 32 --ipm_layer_dims 64 32 8 --ctr_layer_dims 64 32 8 --user_size 290 --item_size 300 --user_item_size 290 300 --downstream $down

#
#    python3 train_baselines.py --epoch_max 40 --debias_mode Pretrain --user_emb_dim 32 --item_emb_dim 32 --ctr_layer_dims 64 32 8 --user_size 6040 --item_size 3952 --user_item_size 6040 3952 --dataset movielens --user_group_size 2 --item_group_size 18 --test_sensitive_group item
#    python3 train_baselines.py --epoch_max 40 --pretrain_mode imputation --debias_mode Pretrain --user_emb_dim 32 --item_emb_dim 32 --ctr_layer_dims 64 32 8 --user_size 6040 --item_size 3952 --user_item_size 6040 3952 --dataset movielens --user_group_size 2 --item_group_size 18 --test_sensitive_group item
#    python3 train_baselines.py --epoch_max 40 --pretrain_mode uniform_imputation --debias_mode Pretrain --user_emb_dim 32 --item_emb_dim 32 --ctr_layer_dims 64 32 8 --user_size 6040 --item_size 3952 --user_item_size 6040 3952 --dataset movielens --user_group_size 2 --item_group_size 18 --test_sensitive_group item
#
#    python3 train_baselines.py --use_weight True --epoch_max 100 --train_mode train --debias_mode ATT --user_emb_dim 32 --item_emb_dim 32 --ctr_layer_dims 64 32 8 --downstream $down --user_size 6040 --item_size 3952 --user_item_size 6040 3952 --dataset movielens --user_group_size 2 --item_group_size 18 --test_sensitive_group item
#    python3 train_baselines.py --use_weight True --epoch_max 100 --train_mode train --debias_mode CVIB --user_emb_dim 32 --item_emb_dim 32 --ctr_layer_dims 64 32 8 --downstream $down --user_size 6040 --item_size 3952 --user_item_size 6040 3952 --dataset movielens --user_group_size 2 --item_group_size 18 --test_sensitive_group item
#
#    python3 train_baselines.py --use_weight True --epoch_max 100 --train_mode train --debias_mode Propensitylearnt_Mode --user_emb_dim 32 --item_emb_dim 32 --ctr_layer_dims 64 32 8 --downstream $down --user_size 6040 --item_size 3952 --user_item_size 6040 3952 --dataset movielens --user_group_size 2 --item_group_size 18 --test_sensitive_group item
#    python3 train_baselines.py --use_weight True --epoch_max 100 --train_mode train --debias_mode SNIPSlearnt_Mode --user_emb_dim 32 --item_emb_dim 32 --ctr_layer_dims 64 32 8 --downstream $down --user_size 6040 --item_size 3952 --user_item_size 6040 3952 --dataset movielens --user_group_size 2 --item_group_size 18 --test_sensitive_group item
#
#    python3 train_baselines.py --use_weight True --epoch_max 100 --train_mode train --debias_mode Direct --user_emb_dim 32 --item_emb_dim 32 --ctr_layer_dims 64 32 8 --downstream $down --user_size 6040 --item_size 3952 --user_item_size 6040 3952 --dataset movielens --user_group_size 2 --item_group_size 18 --test_sensitive_group item
#    python3 train_baselines.py --use_weight True --epoch_max 100 --train_mode train --debias_mode Propensity_DR_Mode --user_emb_dim 32 --item_emb_dim 32 --ctr_layer_dims 64 32 8 --downstream $down --user_size 6040 --item_size 3952 --user_item_size 6040 3952 --dataset movielens --user_group_size 2 --item_group_size 18 --test_sensitive_group item

#    python3 train_baselines.py --use_weight True --epoch_max 100 --train_mode train --debias_mode Uniform_DR_Mode --user_emb_dim 32 --item_emb_dim 32 --ctr_layer_dims 64 32 8 --downstream $down --user_size 6040 --item_size 3952 --user_item_size 6040 3952 --dataset movielens


#
#    python3 train_baselines.py --epoch_max 40 --debias_mode Pretrain --user_emb_dim 32 --item_emb_dim 32 --ctr_layer_dims 64 32 8 --user_size 1338 --item_size 21 --user_item_size 1338 21 --dataset insurance --user_group_size 2 --item_group_size 21 --test_sensitive_group item --data_path data_nonuniform_gender.csv
#    python3 train_baselines.py --epoch_max 40 --pretrain_mode imputation --debias_mode Pretrain --user_emb_dim 32 --item_emb_dim 32 --ctr_layer_dims 64 32 8 --user_size 1338 --item_size 21 --user_item_size 1338 21 --dataset insurance --user_group_size 2 --item_group_size 21 --test_sensitive_group item --data_path data_nonuniform_gender.csv
#    python3 train_baselines.py --epoch_max 40 --pretrain_mode uniform_imputation --debias_mode Pretrain --user_emb_dim 32 --item_emb_dim 32 --ctr_layer_dims 64 32 8 --user_size 1338 --item_size 21 --user_item_size 1338 21 --dataset insurance --user_group_size 2 --item_group_size 21 --test_sensitive_group item --data_path data_nonuniform_gender.csv
#
#    python3 train_baselines.py --use_weight True --epoch_max 100 --train_mode train --debias_mode ATT --user_emb_dim 32 --item_emb_dim 32 --ctr_layer_dims 64 32 8 --downstream $down --user_size 1338 --item_size 21 --user_item_size 1338 21 --dataset insurance --user_group_size 2 --item_group_size 21 --test_sensitive_group item --data_path data_nonuniform_gender.csv
#    python3 train_baselines.py --use_weight True --epoch_max 100 --train_mode train --debias_mode CVIB --user_emb_dim 32 --item_emb_dim 32 --ctr_layer_dims 64 32 8 --downstream $down --user_size 1338 --item_size 21 --user_item_size 1338 21 --dataset insurance --user_group_size 2 --item_group_size 21 --test_sensitive_group item --data_path data_nonuniform_gender.csv
#
#    python3 train_baselines.py --use_weight True --epoch_max 100 --train_mode train --debias_mode Propensitylearnt_Mode --user_emb_dim 32 --item_emb_dim 32 --ctr_layer_dims 64 32 8 --downstream $down --user_size 1338 --item_size 21 --user_item_size 1338 21 --dataset insurance --user_group_size 2 --item_group_size 21 --test_sensitive_group item --data_path data_nonuniform_gender.csv
#    python3 train_baselines.py --use_weight True --epoch_max 100 --train_mode train --debias_mode SNIPSlearnt_Mode --user_emb_dim 32 --item_emb_dim 32 --ctr_layer_dims 64 32 8 --downstream $down --user_size 1338 --item_size 21 --user_item_size 1338 21 --dataset insurance --user_group_size 2 --item_group_size 21 --test_sensitive_group item --data_path data_nonuniform_gender.csv
#
#    python3 train_baselines.py --use_weight True --epoch_max 100 --train_mode train --debias_mode Direct --user_emb_dim 32 --item_emb_dim 32 --ctr_layer_dims 64 32 8 --downstream $down --user_size 1338 --item_size 21 --user_item_size 1338 21 --dataset insurance --user_group_size 2 --item_group_size 21 --test_sensitive_group item --data_path data_nonuniform_gender.csv
#    python3 train_baselines.py --use_weight True --epoch_max 100 --train_mode train --debias_mode Propensity_DR_Mode --user_emb_dim 32 --item_emb_dim 32 --ctr_layer_dims 64 32 8 --downstream $down --user_size 1338 --item_size 21 --user_item_size 1338 21 --dataset insurance --user_group_size 2 --item_group_size 21 --test_sensitive_group item --data_path data_nonuniform_gender.csv


#    python3 train_baselines.py --epoch_max 40 --debias_mode Pretrain --user_emb_dim 32 --item_emb_dim 32 --ctr_layer_dims 64 32 8 --user_size 1338 --item_size 21 --user_item_size 1338 21 --dataset insurance --user_group_size 8 --item_group_size 21 --test_sensitive_group item --data_path data_nonuniform_marital.csv
#    python3 train_baselines.py --epoch_max 40 --pretrain_mode imputation --debias_mode Pretrain --user_emb_dim 32 --item_emb_dim 32 --ctr_layer_dims 64 32 8 --user_size 1338 --item_size 21 --user_item_size 1338 21 --dataset insurance --user_group_size 8 --item_group_size 21 --test_sensitive_group item --data_path data_nonuniform_marital.csv
#    python3 train_baselines.py --epoch_max 40 --pretrain_mode uniform_imputation --debias_mode Pretrain --user_emb_dim 32 --item_emb_dim 32 --ctr_layer_dims 64 32 8 --user_size 1338 --item_size 21 --user_item_size 1338 21 --dataset insurance --user_group_size 8 --item_group_size 21 --test_sensitive_group item --data_path data_nonuniform_marital.csv
#
#    python3 train_baselines.py --use_weight True --epoch_max 100 --train_mode train --debias_mode ATT --user_emb_dim 32 --item_emb_dim 32 --ctr_layer_dims 64 32 8 --downstream $down --user_size 1338 --item_size 21 --user_item_size 1338 21 --dataset insurance --user_group_size 8 --item_group_size 21 --test_sensitive_group item --data_path data_nonuniform_marital.csv
#    python3 train_baselines.py --use_weight True --epoch_max 100 --train_mode train --debias_mode CVIB --user_emb_dim 32 --item_emb_dim 32 --ctr_layer_dims 64 32 8 --downstream $down --user_size 1338 --item_size 21 --user_item_size 1338 21 --dataset insurance --user_group_size 8 --item_group_size 21 --test_sensitive_group item --data_path data_nonuniform_marital.csv
#
#    python3 train_baselines.py --use_weight True --epoch_max 100 --train_mode train --debias_mode Propensitylearnt_Mode --user_emb_dim 32 --item_emb_dim 32 --ctr_layer_dims 64 32 8 --downstream $down --user_size 1338 --item_size 21 --user_item_size 1338 21 --dataset insurance --user_group_size 8 --item_group_size 21 --test_sensitive_group item --data_path data_nonuniform_marital.csv
#    python3 train_baselines.py --use_weight True --epoch_max 100 --train_mode train --debias_mode SNIPSlearnt_Mode --user_emb_dim 32 --item_emb_dim 32 --ctr_layer_dims 64 32 8 --downstream $down --user_size 1338 --item_size 21 --user_item_size 1338 21 --dataset insurance --user_group_size 8 --item_group_size 21 --test_sensitive_group item --data_path data_nonuniform_marital.csv
#
#    python3 train_baselines.py --use_weight True --epoch_max 100 --train_mode train --debias_mode Direct --user_emb_dim 32 --item_emb_dim 32 --ctr_layer_dims 64 32 8 --downstream $down --user_size 1338 --item_size 21 --user_item_size 1338 21 --dataset insurance --user_group_size 8 --item_group_size 21 --test_sensitive_group item --data_path data_nonuniform_marital.csv
#    python3 train_baselines.py --use_weight True --epoch_max 100 --train_mode train --debias_mode Propensity_DR_Mode --user_emb_dim 32 --item_emb_dim 32 --ctr_layer_dims 64 32 8 --downstream $down --user_size 1338 --item_size 21 --user_item_size 1338 21 --dataset insurance --user_group_size 8 --item_group_size 21 --test_sensitive_group item --data_path data_nonuniform_marital.csv


    python3 train_baselines.py --epoch_max 40 --debias_mode Pretrain --user_emb_dim 32 --item_emb_dim 32 --ctr_layer_dims 64 32 8 --user_size 6040 --item_size 3952 --user_item_size 6040 3952 --dataset movielens --user_group_size 7 --item_group_size 18 --test_sensitive_group item --data_path data_nonuniform_movie_tag_age.csv
    python3 train_baselines.py --epoch_max 40 --pretrain_mode imputation --debias_mode Pretrain --user_emb_dim 32 --item_emb_dim 32 --ctr_layer_dims 64 32 8 --user_size 6040 --item_size 3952 --user_item_size 6040 3952 --dataset movielens --user_group_size 7 --item_group_size 18 --test_sensitive_group item --data_path data_nonuniform_movie_tag_age.csv
    python3 train_baselines.py --epoch_max 40 --pretrain_mode uniform_imputation --debias_mode Pretrain --user_emb_dim 32 --item_emb_dim 32 --ctr_layer_dims 64 32 8 --user_size 6040 --item_size 3952 --user_item_size 6040 3952 --dataset movielens --user_group_size 7 --item_group_size 18 --test_sensitive_group item --data_path data_nonuniform_movie_tag_age.csv

    python3 train_baselines.py --use_weight True --epoch_max 100 --train_mode train --debias_mode ATT --user_emb_dim 32 --item_emb_dim 32 --ctr_layer_dims 64 32 8 --downstream $down --user_size 6040 --item_size 3952 --user_item_size 6040 3952 --dataset movielens --user_group_size 7 --item_group_size 18 --test_sensitive_group item --data_path data_nonuniform_movie_tag_age.csv
    python3 train_baselines.py --use_weight True --epoch_max 100 --train_mode train --debias_mode CVIB --user_emb_dim 32 --item_emb_dim 32 --ctr_layer_dims 64 32 8 --downstream $down --user_size 6040 --item_size 3952 --user_item_size 6040 3952 --dataset movielens --user_group_size 7 --item_group_size 18 --test_sensitive_group item --data_path data_nonuniform_movie_tag_age.csv

    python3 train_baselines.py --use_weight True --epoch_max 100 --train_mode train --debias_mode Propensitylearnt_Mode --user_emb_dim 32 --item_emb_dim 32 --ctr_layer_dims 64 32 8 --downstream $down --user_size 6040 --item_size 3952 --user_item_size 6040 3952 --dataset movielens --user_group_size 7 --item_group_size 18 --test_sensitive_group item --data_path data_nonuniform_movie_tag_age.csv
    python3 train_baselines.py --use_weight True --epoch_max 100 --train_mode train --debias_mode SNIPSlearnt_Mode --user_emb_dim 32 --item_emb_dim 32 --ctr_layer_dims 64 32 8 --downstream $down --user_size 6040 --item_size 3952 --user_item_size 6040 3952 --dataset movielens --user_group_size 7 --item_group_size 18 --test_sensitive_group item --data_path data_nonuniform_movie_tag_age.csv

    python3 train_baselines.py --use_weight True --epoch_max 100 --train_mode train --debias_mode Direct --user_emb_dim 32 --item_emb_dim 32 --ctr_layer_dims 64 32 8 --downstream $down --user_size 6040 --item_size 3952 --user_item_size 6040 3952 --dataset movielens --user_group_size 7 --item_group_size 18 --test_sensitive_group item --data_path data_nonuniform_movie_tag_age.csv
    python3 train_baselines.py --use_weight True --epoch_max 100 --train_mode train --debias_mode Propensity_DR_Mode --user_emb_dim 32 --item_emb_dim 32 --ctr_layer_dims 64 32 8 --downstream $down --user_size 6040 --item_size 3952 --user_item_size 6040 3952 --dataset movielens --user_group_size 7 --item_group_size 18 --test_sensitive_group item --data_path data_nonuniform_movie_tag_age.csv

done


#
#for down in MLP NeuBPR
#    do
#    python3 train_baselines.py --epoch_max 15 --debias_mode Pretrain --user_emb_dim 32 --item_emb_dim 32 --ctr_layer_dims 64 32 8
#    python3 train_baselines.py --epoch_max 15 --pretrain_mode imputation --debias_mode Pretrain --user_emb_dim 32 --item_emb_dim 32 --ctr_layer_dims 64 32 8
#    python3 train_baselines.py --epoch_max 15 --pretrain_mode uniform_imputation --debias_mode Pretrain --user_emb_dim 32 --item_emb_dim 32 --ctr_layer_dims 64 32 8
#
#
#    python3 train_baselines.py --use_weight True  --epoch_max 50 --train_mode train --debias_mode Propensitylearnt_Mode --user_emb_dim 32 --item_emb_dim 32 --ctr_layer_dims 64 32 8 --downstream $down
#    python3 train_baselines.py --use_weight True  --epoch_max 50 --train_mode train --debias_mode SNIPSlearnt_Mode --user_emb_dim 32 --item_emb_dim 32 --ctr_layer_dims 64 32 8 --downstream $down
#
#    python3 train_baselines.py --use_weight True  --epoch_max 50 --train_mode train --debias_mode Direct --user_emb_dim 32 --item_emb_dim 32 --ctr_layer_dims 64 32 8 --downstream $down
#    python3 train_baselines.py --use_weight True  --epoch_max 50 --train_mode train --debias_mode Propensity_DR_Mode --user_emb_dim 32 --item_emb_dim 32 --ctr_layer_dims 64 32 8 --downstream $down
#
#    python3 train_baselines.py --use_weight True  --epoch_max 50 --train_mode train --debias_mode Uniform_DR_Mode --user_emb_dim 32 --item_emb_dim 32 --ctr_layer_dims 64 32 8 --downstream $down
#
#
#    python3 train_baselines.py --epoch_max 50 --debias_mode Pretrain --feature_data True --dataset coat --user_dim 14 --item_dim 33 --user_emb_dim 32 --item_emb_dim 32 --ipm_layer_dims 64 32 8 --ctr_layer_dims 64 32 8 --user_size 290 --item_size 300
#    python3 train_baselines.py --epoch_max 50 --debias_mode Pretrain --pretrain_mode imputation --feature_data True --dataset coat --user_dim 14 --item_dim 33 --user_emb_dim 32 --item_emb_dim 32 --ipm_layer_dims 64 32 8 --ctr_layer_dims 64 32 8 --user_size 290 --item_size 300
#    python3 train_baselines.py --epoch_max 50 --debias_mode Pretrain --pretrain_mode uniform_imputation --feature_data True --dataset coat --user_dim 14 --item_dim 33 --user_emb_dim 32 --item_emb_dim 32 --ipm_layer_dims 64 32 8 --ctr_layer_dims 64 32 8 --user_size 290 --item_size 300
#
#
#
#    python3 train_baselines_coat.py --use_weight True  --epoch_max 150 --train_mode train --debias_mode Propensitylearnt_Mode --feature_data True --dataset coat --user_dim 14 --item_dim 33 --user_emb_dim 32 --item_emb_dim 32 --ipm_layer_dims 64 32 8 --ctr_layer_dims 64 32 8 --user_size 290 --item_size 300 --user_item_size 290 300 --downstream $down
#    python3 train_baselines_coat.py --use_weight True  --epoch_max 150 --train_mode train --debias_mode SNIPSlearnt_Mode --feature_data True --dataset coat --user_dim 14 --item_dim 33  --user_emb_dim 32 --item_emb_dim 32 --ipm_layer_dims 64 32 8 --ctr_layer_dims 64 32 8 --user_size 290 --item_size 300  --user_item_size 290 300 --downstream $down
#
#    python3 train_baselines_coat.py --use_weight True  --epoch_max 150 --train_mode train --debias_mode Direct --feature_data True --dataset coat --user_dim 14 --item_dim 33  --user_emb_dim 32 --item_emb_dim 32 --ipm_layer_dims 64 32 8 --ctr_layer_dims 64 32 8 --user_size 290 --item_size 300  --user_item_size 290 300 --downstream $down
#    python3 train_baselines_coat.py --use_weight True  --epoch_max 150 --train_mode train --debias_mode Propensity_DR_Mode --feature_data True --dataset coat --user_dim 14 --item_dim 33  --user_emb_dim 32 --item_emb_dim 32 --ipm_layer_dims 64 32 8 --ctr_layer_dims 64 32 8 --user_size 290 --item_size 300 --user_item_size 290 300  --downstream $down
#
#    python3 train_baselines_coat.py --use_weight True  --epoch_max 150 --train_mode train --debias_mode Uniform_DR_Mode --feature_data True --dataset coat --user_dim 14 --item_dim 33  --user_emb_dim 32 --item_emb_dim 32 --ipm_layer_dims 64 32 8 --ctr_layer_dims 64 32 8 --user_size 290 --item_size 300 --user_item_size 290 300  --downstream $down
#done
# python3 train_causale.py --train_mode dev --debias_mode Propensitylearnt_Mode --user_emb_dim 64 --item_emb_dim 64 --ctr_layer_dims 32 16 8



# python3 train_baselines.py --debias_mode Pretrain --feature_data True --dataset coat --epoch_max 200 --dataset coat --user_dim 14 --item_dim 33  --user_emb_dim 32 --item_emb_dim 32 --ipm_layer_dims 64 32 16  --ctr_layer_dims 32 16 8 --user_size 290 --item_size 300
# python3 train_baselines.py --debias_mode Pretrain --pretrain_mode imputation --feature_data True --dataset coat --epoch_max 100 --dataset coat --user_dim 14 --item_dim 33  --user_emb_dim 32 --item_emb_dim 32 --ipm_layer_dims 64 32 16  --ctr_layer_dims 32 16 8
# python3 train_baselines.py --debias_mode Pretrain --pretrain_mode uniform_imputation --feature_data True --dataset coat --epoch_max 100 --dataset coat --user_dim 14 --item_dim 33  --user_emb_dim 32 --item_emb_dim 32 --ipm_layer_dims 64 32 16  --ctr_layer_dims 32 16 8
# python3 train_baselines.py --train_mode save_imputation --feature_data True --dataset coat


# python3 train_baselines_coat.py --train_mode dev --debias_mode Propensitylearnt_Mode --feature_data True --dataset coat --epoch_max 100 --dataset coat --user_dim 14 --item_dim 33  --user_emb_dim 32 --item_emb_dim 32 --ipm_layer_dims 64 32 16  --ctr_layer_dims 32 16 8 --user_size 290 --item_size 300
# python3 train_baselines_coat.py --train_mode dev --debias_mode SNIPSlearnt_Mode --feature_data True --dataset coat --epoch_max 100 --dataset coat --user_dim 14 --item_dim 33  --user_emb_dim 32 --item_emb_dim 32 --ipm_layer_dims 64 32 16  --ctr_layer_dims 32 16 8 --user_size 290 --item_size 300

# python3 train_baselines_coat.py --train_mode dev --debias_mode Direct --feature_data True --dataset coat --epoch_max 100 --dataset coat --user_dim 14 --item_dim 33  --user_emb_dim 32 --item_emb_dim 32 --ipm_layer_dims 64 32 16  --ctr_layer_dims 32 16 8 --user_size 290 --item_size 300
# python3 train_baselines_coat.py --train_mode dev --debias_mode Propensity_DR_Mode --feature_data True --dataset coat --epoch_max 100 --dataset coat --user_dim 14 --item_dim 33  --user_emb_dim 32 --item_emb_dim 32 --ipm_layer_dims 64 32 16  --ctr_layer_dims 32 16 8 --user_size 290 --item_size 300

# python3 train_baselines_coat.py --train_mode dev --debias_mode Uniform_DR_Mode --feature_data True --dataset coat --epoch_max 100 --dataset coat --user_dim 14 --item_dim 33  --user_emb_dim 32 --item_emb_dim 32 --ipm_layer_dims 64 32 16  --ctr_layer_dims 32 16 8 --user_size 290 --item_size 300


# python3 train_causale.py --train_mode dev --debias_mode Propensitylearnt_Mode --user_emb_dim 64 --item_emb_dim 64 --ctr_layer_dims 32 16 8