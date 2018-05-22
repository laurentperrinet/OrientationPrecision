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

Un problème avec les RNN est qu'ils sortent des tuples (h,c) et qu'on ne peut pas squisher d'autres layers linéaires par dessus. Il va falloir ruser pour faire un bon ring.

# 2018-05-16 - Towards a ring
L'implémentation PyTorch des RNN est un  peu foireuse mais avec du debugging ça passe. Il faut jouer sur la taille des layers hidden et des lengths d'input pour obtenir des tuples correct.

On peut mettre un CNN avant le RNN même si c'est pas prévu pour. Dans ce cas la, il faut écrire un foward pass custom, qui change les dimensions pour donner au RNN des tenseurs de bonne dimension, comme ceci :

    class BiRNN(nn.Module):
        def __init__(self, input_size, hidden_size, num_layers, num_classes):
            super(BiRNN, self).__init__()

            #hyperparams
            self.hidden_size = hidden_size
            self.num_layers = num_layers

            #biRNN
            self.rnn = nn.RNN(input_size, hidden_size, num_layers, batch_first=True, bidirectional=True)

            #CNN
            self.conv = nn.Conv2d(1,6,20)
            self.pool = nn.MaxPool2d(2,2)

            #Out en FC
            self.fc = nn.Linear(hidden_size*2, num_classes)  # 2 for bidirection

        def forward(self, x):
            #rnn Variable
            h0 = torch.zeros(self.num_layers*2, x.size(0), self.hidden_size)
            h0 = Variable(h0)

            #cnn
            x = self.pool(F.relu(self.conv(x)))

            #delete one dimension to feed to RNN
            x = x[:,-1,:,:]

            #rnn --> out: tensor of shape (batch_size, seq_length, hidden_size*2)
            out, _ = self.rnn(x, h0)  

            #final output resize for FC
            out = self.fc(out[:, -1, :])

            return out

Pas encore testé pour voir si il peut apprendre quoi que ce soit avec tout ces resize.

# 2018-05-17 BiRNN, benchmarking, getting the damn thing to work
## Performances
Pour 20 epochs de training avec des batchs de 8 :
* LSTM - MNIST  : 97% accuracy (CUDA)
* LSTM - MC     : 90% accuracy (CUDA)
* RNN  - MNIST  : 93% accuracy (CUDA, 5 epochs, early stop requis)
* RNN -  MC     : 20% (chance = 6,25%) (CUDA, 50 epochs)

On dirait que le RNN est pas super sur MC, à vérifier avec CUDA. Si c'est le cas, il va falloir refaire le BiRNN-CNN pour remplacer les unités du RNN en LSTM (ce qui devrait pas être trop dur).

## BiRNN + CNN
Il faut simplement remplacer le call du nn.RNN en nn.LSTM et lui feed (h0, c0) dans le forward au lieu de h0 seul.

## Matching pursuit
Vu qu'il reste plus qu'a entrainer le BiRNN+CNN, je commence a implémenter les saccades avec l'algo de matching puirsuit de Laurent. Pour l'install :

       pip install LogGabor
       pip install SparseEdges==20171205

# 2018-05-18 MP+Edges classification
Pour intégrer les deux, le mieux serait d'extraire le gagnant de chaque étape de match, de le classifier et de ranger le tout dans une liste avec les infos de MP + la classification du network.

# 2018-05-21 MP+Edges part II
Pour les arrays complexes de numpy on peut utiliser arr.real et arr.imag pour séparer les deux composants.

Les résultats des singles predictions sont pas folles mais au moins elles fonctionnent dans l'ensemble. Surtout en linéaire

# 2018-05-23 Ring
TODO : reussir a faire marcher le ring et faire des prédictions on the fly pour l'intégration avec les stream

Et surtout changer pour un network avec un layer convolutionnel parce que la c'est vraiment trop horrible à regarder.
# A lire
* [Entropy in an image - Python](http://bugra.github.io/work/notes/2014-05-16/entropy-perplexity-image-text/) - Implemented
* [Matching Pursuit - InVibe](http://blog.invibe.net/posts/2015-05-22-a-hitchhiker-guide-to-matching-pursuit.html)
* [PyTorch - RetinaNet](https://github.com/kuangliu/pytorch-retinanet/blob/master/fpn.py) - Interactions latérales en PyTorch (un peu douteux), voir le [papier](https://arxiv.org/abs/1708.02002) - Même carrément douteux
* [PyTorch tutorial repo](https://github.com/ritchieng/the-incredible-pytorch) - Notamment des notebooks de RNN.
* [Réseau smashé de CNN et RNN pour la classification multilabel](https://arxiv.org/pdf/1604.04573.pdf) - Sympathique en application et en théorie
* [A nice presentation about place cell using BiRNN](http://slideplayer.com/slide/10066142/)

# La présentation approche
* [Rules for a good poster](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC1876493/)
* [NYU advice for posters](http://www.personal.psu.edu/drs18/postershow/)
* [Servier Images](https://smart.servier.com/)
