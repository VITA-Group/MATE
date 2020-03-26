# Baseline
python train_film.py --gpu 0,1,2,3 \
  --save-path ./experiments/miniImageNet_MetaOptNet_SVM_FiLM-KME_dual-BN \
  --train-shot 15 --dataset miniImageNet --eps 0.1 --val-episodes-per-batch 1 \
  --head SVM --network ResNet_FiLM \
  --task-embedding FiLM_SVM_WGrad --dual-BN

python train_film.py --gpu 0,1,2,3 \
  --save-path ./experiments/miniImageNet_MetaOptNet_SVM_FiLM-KME_dual-BN_msgan-reg-1e-2 \
  --train-shot 15 --dataset miniImageNet --eps 0.1 --val-episodes-per-batch 1 \
  --head SVM --network ResNet_FiLM \
  --task-embedding FiLM_SVM_WGrad --dual-BN \
  --film-reg-type MSGAN --film-reg-level 1e-2

python train_film.py --gpu 0,1,2,3 \
  --save-path ./experiments/miniImageNet_MetaOptNet_SVM_FiLM-KME_dual-BN_msgan-reg-1e-4 \
  --train-shot 15 --dataset miniImageNet --eps 0.1 --val-episodes-per-batch 1 \
  --head SVM --network ResNet_FiLM \
  --task-embedding FiLM_SVM_WGrad --dual-BN \
  --film-reg-type MSGAN --film-reg-level 1e-4

python train_film.py --gpu 0,1,2,3 \
  --save-path ./experiments/miniImageNet_MetaOptNet_SVM_FiLM-KME_dual-BN_msgan-reg-1e-6 \
  --train-shot 15 --dataset miniImageNet --eps 0.1 --val-episodes-per-batch 1 \
  --head SVM --network ResNet_FiLM \
  --task-embedding FiLM_SVM_WGrad --dual-BN \
  --film-reg-type MSGAN --film-reg-level 1e-6
