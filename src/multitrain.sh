for center_weight in 0.2 0.1 0.05 0.01
do
  python main_with_eval.py ctdetgt --arch dlagt_34 --dataset shangqi --batch_size 4 --master_batch -1 --lr 1.25e-4 --gpus 0 --num_workers 8 --num_epochs 100 --lr_step 60,80 --val_intervals 1 --center_weight $center_weight --save_dir lamda_${center_weight}
  sleep 30
done



python main_with_eval.py ctdetgt --arch resdcngt_18 --dataset pascal --num_epochs 30 --lr_step 6,13,20 --flip_test --center_weight 1.0 --save_dir norm_centerloss_bn_eq1_1.0_hm08_lamdaR05 --lr_step_ratio 0.22 --load_model /storage/caijihuzhuo/CenterNet/exp/ctdetgt/norm_centerloss_bn_eq1_1.0_hm08_lamdaR05/ap50_0.000_ap0.671_ar0.000.pth --lr 1.5e-4 --eq1  --hm_weight 0.8 --lamda_regular_weight 0.5
