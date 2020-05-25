python test.py --gpu 0 \
--load ./experiments/CIFAR_FS_RFS_SVM_ResNet12_baseline/resnet12_last.pth \
--way 5 --shot 1 --query 15 --dataset CIFAR_FS --episode 1000 \
--head SVM --network ResNetRFS

python train_film.py --gpu 0,1 \
  --save-path ./experiments/CIFAR_FS_RFS_SVM_FiLM-SVM-WGrad_dual-BN_load-naive-fix_lr0.01 \
  --save-epoch 10 --train-shot 5 --dataset CIFAR_FS --val-episodes-per-batch 4 \
  --head SVM --network ResNetRFS_FiLM \
  --task-embedding FiLM_SVM_WGrad --dual-BN --train-film-dualBN --lr 0.01 \
  --load-naive-backbone --load ./experiments/CIFAR_FS_RFS_SVM_ResNet12_baseline/resnet12_last.pth
