# 2018-05-07 MC network sparse coding
On refait le code avec 16 orientations entre 0 et pi.

L'architecture du réseau est composé d'une entrée en 256*256 (plus de réduction donc, pour éviter l'aliasing), suivi d'une couche ReLu leaky pour sparsifier la réprésentation d'entrée. Ensuite un layer de convolutionné et de maxpooling, suivi de relus pour la sortie, donc :

    Net(
        (relu1): Linear(in_features=256, out_features=200, bias=True)

        (conv1): Conv2d(1, 6, kernel_size=(20, 20), stride=(1, 1))
        (pool): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)

        (fc1): Linear(in_features=90, out_features=30, bias=True)
        (fc2): Linear(in_features=30, out_features=20, bias=True)

        (outlayer): Linear(in_features=20, out_features=16, bias=True)
    )

Par contre j'ai toujours un problème avec l'utilisation de la cross entropy en sortie, les formes des tenseurs de sorties du réseau sont bizzares (4dim alors qu'elles devraient être 2D : batch x predicted).

TODO : comparer avec l'exemple stock de CIFAR10 qui utilise le même optimizer.
