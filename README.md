# CS453_Project

Leveraging neural coverage to train robust ML models

Team4 [Seongho Keum, Kwangmin Kim, Dongwon Shin, Sungwoo Jeon]


## How to train model

dataset: MNIST, CIFAR10 

model: MLP, LENET

```
python train_model.py --dataset CIFAR10 --model LENET --epochs 10 --batch 64 --save-path ./weights/
```

## How to optimize noise

please run
```
pip install opencv-python
```

We have two configurable parameters: 'lagrangian' and 'step_number'.

The 'lagrangian' balances two factors: 1) the size of perturbation, and 2) maximizing neuron coverage. If the 'lagrangian' value is high, we aim to minimize the perturbation size. Conversely, if the 'lagrangian' value is low, we prioritize maximizing neuron coverage.

we save two types of sample images "original images" and "perturbed images".

We can also retrain models using perturbed images. (for --epochs-noise times) 

For example, you can run
```
python add_noise.py --dataset MNIST --model MLP --model-path weights/MLP_MNIST.pt --step-num 100 --lagrangian 0.01 --epochs-noise 5 --save-path ./weights/
```
or
```
python add_noise.py --dataset CIFAR10 --model MLP --model-path weights/MLP_CIFAR10.pt --step-num 100 --lagrangian 0.01 --epochs-noise 5 --save-path ./weights/
```

Then, you can obtain both the original and updated neuron coverage as well as the training accuracies as follows.
```
Coverage Before attack: 19.37%
Coverage After attack: 23.00%
```