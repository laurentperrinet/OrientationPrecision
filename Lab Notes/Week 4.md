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

Ca marche avec un convolutionel et deux couches linéaires derrière !

### Lateral
Regarder la doc PyTorch sur les RNN.

### PyTorch
Pour recharger un réseau entrainé avec CUDA sur du CPU : [ce lien](https://discuss.pytorch.org/t/error-when-moving-gpu-trained-model-to-cpu/10980)

# 2018-05-15 - Recurrent Net
Utiliser l'implémentation PyTorch de RNN ira surement plus vite que de tenter de les coder soi-même (quoi que, voir le code [source](https://pytorch.org/docs/master/_modules/torch/nn/modules/rnn.html)).

# A lire
* [Entropy in an image - Python](http://bugra.github.io/work/notes/2014-05-16/entropy-perplexity-image-text/) - Implemented
* [Matching Pursuit - InVibe](http://blog.invibe.net/posts/2015-05-22-a-hitchhiker-guide-to-matching-pursuit.html)
* [PyTorch - RetinaNet](https://github.com/kuangliu/pytorch-retinanet/blob/master/fpn.py) - Interactions latérales en PyTorch (un peu douteux), voir le [papier](https://arxiv.org/abs/1708.02002) - Même carrément douteux
* [PyTorch tutorial repo](https://github.com/ritchieng/the-incredible-pytorch) - Notamment des notebooks de RNN.

# La présentation approche
* [Rules for a good poster](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC1876493/)
* [NYU advice for posters](http://www.personal.psu.edu/drs18/postershow/)
* [Servier Images](https://smart.servier.com/)
