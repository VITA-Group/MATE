python train_film.py --gpu 6,7 \
  --save-path ./experiments/CIFAR_FS_MetaOptNet_SVM_FiLM-SVM-WGrad_dual-BN_nomultiplier_clamp_he-init \
  --save-epoch 100 --train-shot 5 --dataset CIFAR_FS --val-episodes-per-batch 4 \
  --head SVM --network ResNet_FiLM \
  --task-embedding FiLM_SVM_WGrad --dual-BN \
  --film-reg-type MSGAN --film-reg-level 1.0
