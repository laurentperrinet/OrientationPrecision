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

TODO : comparer avec l'exemple stock de CIFAR10 qui utilise le même optimizer. ERR : mauvais resize entre CONV2D et Linear FC1.

# 2018-05-09 Fixing bugs and binarizing CHAMP
### MC V3
Le network n'apprend pas mais au moins il marche.

Pour le faire tourner en cuda :

    model.cuda()
    data, target = data.cuda(), target.cuda()

L'archi qui semble marcher un peu mieux :

    Net(
    (relu1): Linear(in_features=128, out_features=200, bias=True)

    (conv1): Conv2d(1, 6, kernel_size=(20, 20), stride=(1, 1))
    (pool): MaxPool2d(kernel_size=(2, 2), stride=(2, 2), dilation=(1, 1), ceil_mode=False)

    (fc1): Linear(in_features=29160, out_features=2000, bias=True)
    (fc2): Linear(in_features=2000, out_features=200, bias=True)
    (fc3): Linear(in_features=200, out_features=100, bias=True)

    (outlayer): Linear(in_features=100, out_features=16, bias=True)
    )

Celle qui marche vraiment c'est celle qui utilise 3 couches de ReLu. A mon avis c'est la convolution dans du 1D qui fais buguer les choses sur l'autre modèle (a vérifier en le faisant tourner chez moi).

### CHAMP Binarization
Attention à l'enumerate dans les tenseurs. On peut remplacer les valeurs de chaque sous-tenseurs comme on le ferait avec une liste, par exemple :

    L1.dictionary[a][b][c][d] = 0.0

Pour l'erreur :

    torch.mm received an invalid combination of arguments - got
    (torch.FloatTensor, Variable), but expected one of:

Il suffit de wrap le dico comme une variable (L1.dictionary)

### Integration where+orientation
Pour intégrer le modèle saccadique avec la détection d'angle, il faut créer une représentation de la quantité d'information de l'image pour diriger la focalisation.

*L'entropie mesure l'incertitude dans un processus stochastique : plus un objet est incertain plus il est entropique. En image, on peut s'attendre à ce que les zones avec le plus de variances soient celles qui possèdent le plus d'entropie.*

# A lire
[Entropy in an image - Python](http://bugra.github.io/work/notes/2014-05-16/entropy-perplexity-image-text/)
