# 2018-05-14 - Feedforward, lateral et unsync MC detection
Pour rendre la question un peu plus bio-like, on est passé sur 16 orientations de motionclouds. Maintenant, on va jouer sur les types d'interactions du réseau pour poser des questions bio. L'unsync permet de poser la question de fonctionnement asynchrone du réseau visuel (épilepsie, gradient de chlore KCC2..) et le lateral permet d'aller sur de l'OBV1.

### Feedforward
Pas de soucis, marche tout seul, il faut juste lui mettre une résolution d'input > 32x32 sinon ça donne rien.

### Unsync
Les tenseurs de PyTorch ont tous un flag param.requieres_grad, qui indique si il est True que le gradient est calculé sur le tenseur. Pour freeze un layer par exemple :

    for param in conv1.parameters():
        param.requires_grad = False

Ecrit d'une manière peu digeste, un tuto sur le [sujet](https://github.com/mortezamg63/Accessing-and-modifying-different-layers-of-a-pretrained-model-in-pytorch).

Les résultats montrent que la précision finale unsync est plus faible que celle en sync et que la descente de loss est absolument dégueulasse sans synchro.

# A lire
* [Entropy in an image - Python](http://bugra.github.io/work/notes/2014-05-16/entropy-perplexity-image-text/)
* [Matching Pursuit - InVibe](http://blog.invibe.net/posts/2015-05-22-a-hitchhiker-guide-to-matching-pursuit.html)
* [PyTorch - RetinaNet](https://github.com/kuangliu/pytorch-retinanet/blob/master/fpn.py) - Interactions latérales en PyTorch (un peu douteux), voir le [papier](https://arxiv.org/abs/1708.02002)
