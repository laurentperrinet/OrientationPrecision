# 2018-05-23 Ring
TODO : reussir a faire marcher le ring et faire des prédictions on the fly pour l'intégration avec les stream.
Et surtout changer pour un network avec un layer convolutionnel parce que la c'est vraiment trop horrible à regarder.

Le problème avec le réseau actuel est qu'il classifie tout les angles comme un angle de classe 6 (environ pi/4, pi/3). Il faudrait ajouter un layer de dropout en sortie ou réduire le nombre d'épochs pour éviter l'overfit.

### Convolutional layers
Une solution pour éviter cet overfitting serait aussi de combiner dropout et augmentation du nombre de "colonnes", càd de dicos appris par le layer convolutionnel.

    self.conv1 = nn.Conv2d(1, 16, 20)

### Ring
L'apprentissage du Ring (convo en 16+RNN) ne montre pas de convergence en 15 epochs, a voir en bruteforce avec un GPU, mais c'est douteux.

Super nouvelle, le passage du RNN au LSTM est plug and play, il suffit juste de rajouter le tuple (h0,c0) en input et il n'y a aucun problème de resize. En LSTM, on dirait que le réseau converge, au moins sur les deux premières époques.
Pour l'instant le mieux qu'on obtient est :

    Accuracy LSTM+CNN = 3 x Accuracy RNN+CNN
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
