for center_weight in 0.2 0.1 0.05 0.01
do
  python main_with_eval.py ctdetgt --arch dlagt_34 --dataset shangqi --batch_size 4 --master_batch -1 --lr 1.25e-4 --gpus 0 --num_workers 8 --num_epochs 100 --lr_step 60,80 --val_intervals 1 --center_weight $center_weight --save_dir lamda_${center_weight}
  sleep 30
done