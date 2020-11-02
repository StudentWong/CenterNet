for center_weight in 0.0
do
  python main_with_eval.py ctdetgt --arch resdcngt_18 --dataset pascal --batch_size 32 --master_batch -1 --lr 5e-4 --gpus 0 --num_workers 8 --num_epochs 70 --lr_step 45,60 --val_intervals 1 --center_weight $center_weight --save_dir voc_lamda_${center_weight}
  sleep 30
  #python main_with_eval.py ctdetgt --arch resdcngt_18 --dataset pascal --batch_size 32 --master_batch -1 --lr 5e-4 --gpus 0 --num_workers 8 --num_epochs 70 --lr_step 45,60 --val_intervals 1 --center_weight $center_weight --save_dir voc_eq1_lamda_${center_weight} --eq1
  #sleep 30
done

python main_with_eval.py ctdet --arch resdcn_18 --dataset pascal --batch_size 32 --master_batch -1 --lr 5e-4 --gpus 0 --num_workers 8 --num_epochs 70 --lr_step 45,60 --val_intervals 1 --save_dir voc_original
