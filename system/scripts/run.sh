# !/bin/sh

# python -u main.py -data fmnist -m resnet18 -algo LocalSuperiorSoups -gr 3 --n_verts 3 -did 1 -eg 1 --div_coeff 1e-1 --aff_coeff 1e-5 -go lss_fmnist_resnet18 -nc 5 -lr 1e-3 --num_classes 10 | tee ../tmp/Digit5/lss_fmnist_resnet18.output &

# python -u main.py -data Cifar10-dir5em1 -m resnet18 -algo LocalSuperiorSoups -gr 1 -did 2 -eg 1 -go lss_Cifar10_resnet18 -nc 5 -lr 5e-4 --num_classes 10 | tee ../tmp/Cifar10/lss_Cifar10_resnet18.output &

python -u main.py -data Digit5 -m resnet18 -algo LocalSuperiorSoups -gr 1 -did 3 -eg 1 -go lss_digit5_resnet18 -nc 5 -lr 5e-4 --num_classes 10 | tee ../tmp/Digit5/lss_digit5_resnet18.output &

# python -u main.py -data DomainNet -m resnet18 -algo LocalSuperiorSoups -gr 1 -did 4 -eg 1 -go lss_DomainNet_resnet18 -nc 5 -lr 5e-4 --num_classes 10 | tee ../tmp/DomainNet/lss_DomainNet_resnet18.output &

# echo "Running scripts in parallel"
wait # This will wait until all scripts finish
echo "Script done running"
