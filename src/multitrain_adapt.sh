for weight in 0.6 0.4 0.2 0.8
do
  python main_with_eval.py ctdetadapt --arch resdcn4cshare_18 --exp_id fusion_384 --dataset flirfusion --num_epochs 70 --lr_step 45,60 --batch_size 32 --eq1 --flip_test --adapt_thermal_weight ${weight} --save_dir adapt_${weight} --val_intervals 1
  sleep 30
done


