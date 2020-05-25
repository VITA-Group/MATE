python test.py --gpu 0 \
--load ./experiments/CIFAR_FS_RFS_SVM_ResNet12_baseline/resnet12_last.pth \
--way 5 --shot 1 --query 15 --dataset CIFAR_FS --episode 1000 \
--head SVM --network ResNetRFS
