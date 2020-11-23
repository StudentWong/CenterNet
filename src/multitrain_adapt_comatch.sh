for weight in 1.0 0.8
do
  for cut in 0.0 0.2
  do
    for att_w in 0.0 1
    do
      for att_ft in 0.0 0.5
      do
      python main_with_eval.py ctdetadapt --arch resdcn4cshareco_18 --exp_id fusion_384 --dataset flirfusion --num_epochs 70 --lr_step 45,60 --batch_size 32 --eq1 --flip_test --adapt_thermal_weight ${weight} --save_dir adapt_Tweight${weight}_cut${cut}_attw${att_w}_attft${att_ft} --val_intervals 1 --share_cut ${cut} --comatch_att_weight ${att_w} --comatch_ft_weight ${att_ft}
      sleep 30
      done
    done
  done
done


