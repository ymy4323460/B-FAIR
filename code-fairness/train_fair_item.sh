#!/usr/bin/env bash
#for lambda_impression in 0.001 0.01 0.05 0.1 0.3 0.5 0.7 1 3 5 7 10
#do
#    python3 train_model.py \
#        --debias_mode IPM_Embedding \
#        --epoch_max 200 \
#        --dataset huawei \
#        --user_dim 300 \
#        --item_dim 1 \
#        --feature_data True \
#        --impression_or_click click  \
#        --user_emb_dim 64 \
#        --item_emb_dim 32 \
#        --ipm_layer_dims 64 32 32 \
#        --ctr_layer_dims 64 32 32 \
#        --lambda_impression $lambda_impression
#done
#for lambda_impression in 0.001 0.01 0.05 0.1 0.3 0.5 0.7 1 3 5 7 10
#do
#    python3 train_model.py \
#        --debias_mode Noweight \
#        --epoch_max 150 \
#        --dataset huawei \
#        --user_dim 29 \
#        --item_dim 211 \
#        --feature_data True \
#        --impression_or_click impression \
#        --user_emb_dim 29 \
#        --item_emb_dim 32 \
#        --ipm_layer_dims 64 32 8 \
#        --ctr_layer_dims 64 32 8 \
#        --iter_save 10 \
#        --clip_value 0.2 0.8 \
#        --lambda_impression $lambda_impression
#done

#movielens
for lambda_impression in 0.001
do
  for confoun in False
  do
    for down in mlpBPR NeuBPR gmfBPR MLP
    do
      for debias in True
      do
        for data in insurance
        do
          python3 train_fair.py \
                --debias_mode Fairness \
                --is_debias $debias \
                --downstream $down \
                --epoch_max 150 \
                --dataset $data\
                --user_dim 1\
                --item_dim 1\
                --user_size 1338 \
                --item_size 21 \
                --user_group_size 2\
                --item_group_size 21\
                --user_item_size 1338 21 \
                --user_emb_dim 32 \
                --item_emb_dim 32 \
                --ipm_layer_dims 64 32 8 \
                --ctr_layer_dims 64 32 8 \
                --iter_save 10 \
                --clip_value 0.2 0.8 \
                --ctr_classweight 1 1 \
                --embedding_classweight 1 1 \
                --lambda_impression $lambda_impression\
                --test_sensitive_group item \
                --data_path data_nonuniform_gender_product.csv

          python3 train_fair.py \
                --debias_mode Fairness \
                --is_debias $debias \
                --downstream $down \
                --epoch_max 150 \
                --dataset $data\
                --user_dim 1\
                --item_dim 1\
                --user_size 1338 \
                --item_size 21 \
                --user_group_size 9\
                --item_group_size 21\
                --user_item_size 1338 21 \
                --user_emb_dim 32 \
                --item_emb_dim 32 \
                --ipm_layer_dims 64 32 8 \
                --ctr_layer_dims 64 32 8 \
                --iter_save 10 \
                --clip_value 0.2 0.8 \
                --ctr_classweight 1 1 \
                --embedding_classweight 1 1 \
                --lambda_impression $lambda_impression\
                --test_sensitive_group item \
                --data_path data_nonuniform_marital_product.csv
#
#          python train_adv.py \
#              --debias_mode Adversarial \
#              --is_debias $debias \
#              --confounder $confoun \
#              --lambda_confounder $confoun_lamb \
#              --downstream $down \
#              --epoch_max 150 \
#              --dataset yahoo \
#              --user_dim 1 \
#              --item_dim 1 \
#              --user_emb_dim 32 \
#              --item_emb_dim 32 \
#              --ipm_layer_dims 64 32 8 \
#              --ctr_layer_dims 64 32 8 \
#              --iter_save 10 \
#              --clip_value 0.2 0.8 \
#              --ctr_classweight 1 1 \
#              --embedding_classweight 1 1 \
#              --lambda_impression $lambda_impression
        done
      done
    done
  done
done

#synthetic_data
#for lambda_impression in 0.001
#do
#  for confoun in False
#  do
#    for down in gmfBPR MLP
#    do
#      for debias in True
#      do
#        for data in nonlinearinvdata_25_32
#        do
#          python3 train_fair.py \
#                --debias_mode Fairness \
#                --is_debias $debias \
#                --downstream $down \
#                --epoch_max 150 \
#                --dataset $data\
#                --user_dim 64 \
#                --item_dim 32 \
#                --feature_data True \
#                --user_emb_dim 32 \
#                --item_emb_dim 32 \
#                --ipm_layer_dims 64 32 8 \
#                --ctr_layer_dims 64 32 8 \
#                --iter_save 10 \
#                --clip_value 0.2 0.8 \
#                --ctr_classweight 1 1 \
#                --embedding_classweight 1 1 \
#                --no_disentangle False \
#                --lambda_impression $lambda_impression
##
##          python train_adv.py \
##              --debias_mode Adversarial \
##              --is_debias $debias \
##              --confounder $confoun \
##              --lambda_confounder $confoun_lamb \
##              --downstream $down \
##              --epoch_max 150 \
##              --dataset yahoo \
##              --user_dim 1 \
##              --item_dim 1 \
##              --user_emb_dim 32 \
##              --item_emb_dim 32 \
##              --ipm_layer_dims 64 32 8 \
##              --ctr_layer_dims 64 32 8 \
##              --iter_save 10 \
##              --clip_value 0.2 0.8 \
##              --ctr_classweight 1 1 \
##              --embedding_classweight 1 1 \
##              --lambda_impression $lambda_impression
#        done
#      done
#    done
#  done
#done






for confoun in False
do
  for down in mlpBPR NeuBPR gmfBPR MLP
  do
    for debias in False
    do
      for data in insurance #nonlinearinvdata_25_32 nonlinearinvdata_50_32
      do
        python3 train_fair.py \
                --debias_mode Adversarial \
                --is_debias $debias \
                --confounder $confoun \
                --downstream $down \
                --epoch_max 100 \
                --dataset $data\
                --user_dim 1\
                --item_dim 1\
                --user_size 1338 \
                --item_size 21 \
                --user_group_size 2\
                --item_group_size 21\
                --user_item_size 1338 21 \
                --user_emb_dim 32 \
                --item_emb_dim 32 \
                --ipm_layer_dims 64 32 8 \
                --ctr_layer_dims 64 32 8 \
                --iter_save 10 \
                --clip_value 0.2 0.8 \
                --ctr_classweight 3 1 \
                --embedding_classweight 1 1\
                --test_sensitive_group item \
                --data_path data_nonuniform_gender_product.csv

        python3 train_fair.py \
                --debias_mode Adversarial \
                --is_debias $debias \
                --confounder $confoun \
                --downstream $down \
                --epoch_max 100 \
                --dataset $data\
                --user_dim 1\
                --item_dim 1\
                --user_size 1338 \
                --item_size 21 \
                --user_group_size 9\
                --item_group_size 21\
                --user_item_size 1338 21 \
                --user_emb_dim 32 \
                --item_emb_dim 32 \
                --ipm_layer_dims 64 32 8 \
                --ctr_layer_dims 64 32 8 \
                --iter_save 10 \
                --clip_value 0.2 0.8 \
                --ctr_classweight 3 1 \
                --embedding_classweight 1 1\
                --test_sensitive_group item \
                --data_path data_nonuniform_marital_product.csv

#        python3 train_fair.py \
#                --debias_mode Adversarial \
#                --is_debias $debias \
#                --confounder $confoun \
#                --downstream $down \
#                --epoch_max 100 \
#                --dataset $data\
#                --user_dim 1\
#                --item_dim 1\
##                --user_size 6040 \
##                --item_size 3952 \
#                --user_group_size 2\
#                --item_group_size 18\
#                --user_item_size 6040 3952 \
#                --user_emb_dim 32 \
#                --item_emb_dim 32 \
#                --ipm_layer_dims 64 32 8 \
#                --ctr_layer_dims 64 32 8 \
#                --iter_save 10 \
#                --clip_value 0.2 0.8 \
#                --ctr_classweight 3 1 \
#                --embedding_classweight 1 1\
#                --test_sensitive_group item

#        python3 train_adv.py \
#                --debias_mode Adversarial \
#                --is_debias $debias \
#                --confounder $confoun \
#                --downstream $down \
#                --epoch_max 100 \
#                --dataset $data\
#                --user_dim 1\
#                --item_dim 1\
#                --user_size 6040 \
#                --item_size 3952 \
#                --user_item_size 6040 3952 \
#                --user_emb_dim 32 \
#                --item_emb_dim 32 \
#                --ipm_layer_dims 64 32 8 \
#                --ctr_layer_dims 64 32 8 \
#                --iter_save 10 \
#                --clip_value 0.2 0.8 \
#                --ctr_classweight 3 1 \
#                --embedding_classweight 1 1
#        python3 train_syn_adv.py \
#                --debias_mode Adversarial \
#                --is_debias $debias \
#                --confounder $confoun \
#                --downstream $down \
#                --epoch_max 100 \
#                --dataset $data\
#                --user_dim 64 \
#                --item_dim 32 \
#                --feature_data True \
#                --user_emb_dim 32 \
#                --item_emb_dim 32 \
#                --ipm_layer_dims 64 32 8 \
#                --ctr_layer_dims 64 32 8 \
#                --iter_save 10 \
#                --clip_value 0.2 0.8 \
#                --ctr_classweight 1 1 \
#                --embedding_classweight 1 1


#        python train_adv.py \
#            --debias_mode Adversarial \
#            --is_debias $debias \
#            --confounder $confoun \
#            --downstream $down \
#            --epoch_max 30 \
#            --dataset yahoo \
#            --user_dim 1 \
#            --item_dim 1 \
#            --user_emb_dim 32 \
#            --item_emb_dim 32 \
#            --ipm_layer_dims 64 32 8 \
#            --ctr_layer_dims 64 32 8 \
#            --iter_save 10 \
#            --ctr_classweight 1 1 \
#            --embedding_classweight 1 1 \
#            --clip_value 0.2 0.8
        done
    done
  done
done

#
#for lambda_impression in 0.3 0.1 0.05 0.01 0.001
#do
#  for confoun in True False
#  do
#    for down in MLP gmfBPR bprBPR mlpBPR NeuBPR
#    do
#      for debias in True
#      do
#        python3 train_adv.py \
#            --debias_mode Adversarial \
#            --is_debias $debias \
#            --confounder $confoun \
#            --downstream $down \
#            --epoch_max 150 \
#            --dataset huawei \
#            --user_dim 29 \
#            --item_dim 211 \
#            --feature_data True \
#            --user_emb_dim 32 \
#            --item_emb_dim 32 \
#            --ipm_layer_dims 64 32 8 \
#            --ctr_layer_dims 64 32 8 \
#            --iter_save 10 \
#            --clip_value 0.2 0.8 \
#            --lambda_impression $lambda_impression
#      done
#    done
#  done
#done
#
## baseline
#
#for confoun in False
#do
#  for down in MLP gmfBPR bprBPR mlpBPR NeuBPR
#  do
#    for debias in False
#    do
#      python3 train_adv.py \
#          --debias_mode Adversarial \
#          --is_debias $debias \
#          --confounder $confoun \
#          --downstream $down \
#          --epoch_max 150 \
#          --dataset huawei \
#          --user_dim 29 \
#          --item_dim 211 \
#          --feature_data True \
#          --user_emb_dim 32 \
#          --item_emb_dim 32 \
#          --ipm_layer_dims 64 32 8 \
#          --ctr_layer_dims 64 32 8 \
#          --iter_save 10 \
#          --clip_value 0.2 0.8
#    done
#  done
#done
