# Baseline
python train_film.py --gpu 6,7 \
  --save-path ./experiments/CIFAR_FS_MetaOptNet_SVM_FiLM-SVM-WGrad_dual-BN \
  --save-epoch 100 --train-shot 5 --dataset CIFAR_FS --val-episodes-per-batch 4 \
  --head SVM --network ResNet_FiLM \
  --task-embedding FiLM_SVM_WGrad --dual-BN

python train_film.py --gpu 6,7 \
  --save-path ./experiments/CIFAR_FS_MetaOptNet_SVM_FiLM-SVM-WGrad_dual-BN_msgan-reg-1.0 \
  --save-epoch 100 --train-shot 5 --dataset CIFAR_FS --val-episodes-per-batch 4 \
  --head SVM --network ResNet_FiLM \
  --task-embedding FiLM_SVM_WGrad --dual-BN \
  --film-reg-type MSGAN --film-reg-level 1.0

python train_film.py --gpu 6,7 \
  --save-path ./experiments/CIFAR_FS_MetaOptNet_SVM_FiLM-SVM-WGrad_dual-BN_msgan-reg-1e-2 \
  --save-epoch 100 --train-shot 5 --dataset CIFAR_FS --val-episodes-per-batch 4 \
  --head SVM --network ResNet_FiLM \
  --task-embedding FiLM_SVM_WGrad --dual-BN \
  --film-reg-type MSGAN --film-reg-level 1e-2

python train_film.py --gpu 4,5 \
  --save-path ./experiments/CIFAR_FS_MetaOptNet_SVM_FiLM-SVM-WGrad_dual-BN_msgan-reg-1e-4 \
  --save-epoch 100 --train-shot 5 --dataset CIFAR_FS --val-episodes-per-batch 4 \
  --head SVM --network ResNet_FiLM \
  --task-embedding FiLM_SVM_WGrad --dual-BN \
  --film-reg-type MSGAN --film-reg-level 1e-4

python train_film.py --gpu 2,3 \
  --save-path ./experiments/CIFAR_FS_MetaOptNet_SVM_FiLM-SVM-WGrad_dual-BN_msgan-reg-1e-6 \
  --save-epoch 100 --train-shot 5 --dataset CIFAR_FS --val-episodes-per-batch 4 \
  --head SVM --network ResNet_FiLM \
  --task-embedding FiLM_SVM_WGrad --dual-BN \
  --film-reg-type MSGAN --film-reg-level 1e-6
