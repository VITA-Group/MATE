# LOAD and TRAIN

# Load-train
python train_film.py --gpu 0,1 \
  --save-path ./experiments/CIFAR_FS_MetaOptNet_SVM_FiLM-SVM-WGrad_dual-BN_load-naive_lr0.01 \
  --save-epoch 10 --train-shot 5 --dataset CIFAR_FS --val-episodes-per-batch 4 \
  --head SVM --network ResNet_FiLM \
  --task-embedding FiLM_SVM_WGrad --dual-BN --lr 0.01 \
  --load-naive-backbone --load ./experiments/CIFAR_FS_MetaOptNet_SVM_baseline/epoch_22.pth

# Load-train + msgan
python train_film.py --gpu 6,7 \
  --save-path ./experiments/CIFAR_FS_MetaOptNet_SVM_FiLM-SVM-WGrad_dual-BN_load-naive_msgan1e-6_lr0.01 \
  --save-epoch 10 --train-shot 5 --dataset CIFAR_FS --val-episodes-per-batch 4 \
  --head SVM --network ResNet_FiLM \
  --task-embedding FiLM_SVM_WGrad --dual-BN --lr 0.01 \
  --film-reg-type MSGAN --film-reg-level 1e-6 \
  --load-naive-backbone --load ./experiments/CIFAR_FS_MetaOptNet_SVM_baseline/epoch_22.pth

# Load-train + normfilm
python train_film.py --gpu 0,1 \
  --save-path ./experiments/CIFAR_FS_MetaOptNet_SVM_FiLM-SVM-WGrad_dual-BN_load-naive_normfilm_lr0.01 \
  --save-epoch 10 --train-shot 5 --dataset CIFAR_FS --val-episodes-per-batch 4 \
  --head SVM --network ResNet_FiLM \
  --task-embedding FiLM_SVM_WGrad --dual-BN --lr 0.01 \
  --film-normalize \
  --load-naive-backbone --load ./experiments/CIFAR_FS_MetaOptNet_SVM_baseline/epoch_22.pth

# Load-train + msgan + normfilm
python train_film.py --gpu 2,4 \
  --save-path ./experiments/CIFAR_FS_MetaOptNet_SVM_FiLM-SVM-WGrad_dual-BN_load-naive_msgan1e-6_normfilm_lr0.01 \
  --save-epoch 10 --train-shot 5 --dataset CIFAR_FS --val-episodes-per-batch 4 \
  --head SVM --network ResNet_FiLM \
  --task-embedding FiLM_SVM_WGrad --dual-BN --lr 0.01 \
  --film-reg-type MSGAN --film-reg-level 1e-6 --film-normalize \
  --load-naive-backbone --load ./experiments/CIFAR_FS_MetaOptNet_SVM_baseline/epoch_22.pth