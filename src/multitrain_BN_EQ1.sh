for center_weight in 0.7 1.0 1.3
do
  for hm_weight in 0.6 0.8 1.0
  do
    for lamda_regular_weight in 0.3 0.5 0.8
    do
    python main_with_eval.py ctdetgt --arch resdcngt_18 --dataset pascal --num_epochs 15 --lr_step 5 --flip_test --center_weight $center_weight --save_dir multi_norm_centerloss_bn_eq1_${center_weight}_hm${hm_weight}_lamdaR${lamda_regular_weight} --lr_step_ratio 0.22 --load_model /storage/caijihuzhuo/CenterNet/exp/ctdetgt/norm_centerloss_bn_eq1_1.0_hm08_lamdaR05/ap50_0.000_ap0.727_ar0.000.pth --lr 7e-6 --eq1  --hm_weight $hm_weight --lamda_regular_weight $lamda_regular_weight
    sleep 30
    done
  done
done
