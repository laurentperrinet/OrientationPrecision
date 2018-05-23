# 2018-05-23 Ring
Le problème avec le réseau actuel est qu'il classifie tout les angles comme un angle de classe 6 (environ pi/4, pi/3). Il faudrait ajouter un layer de dropout en sortie ou réduire le nombre d'épochs pour éviter l'overfit ?

Pour faire du dropout avec pytorch sur les LSTM :

    self.rnn = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, bidirectional=True, dropout = 0.2)

Pour rajouter un layer de dropout qui drop p éléments du tenseur (il les transforme en 0):

    torch.nn.Dropout(p=0.5, inplace=False)


Faut bien sur le désactiver quand on teste pour eviter de rajouter encore du drop, donc

    model.eval()

## Convolutional layers
Une solution pour éviter cet overfitting serait aussi de combiner dropout et augmentation du nombre de "colonnes", càd de dicos appris par le layer convolutionnel.

    self.conv1 = nn.Conv2d(1, 16, 20)

## Benchmark RNN vs LSTM for lateral connexions
L'apprentissage du convo en 6+RNN ne montre pas de convergence en 15 epochs sur un CPU. Pas non plus en 100 sur un GPU (31% acc.). Avec 20% de dropout dans le RNN, on arrive à 44% de chance, sans convergence non plus. Avec 20% de dropout entre le RNN et le FC de sortie, on a 28% d'accuracy. (Trois conditions testés sur 5 seeds et moyennées).

Super nouvelle, le passage du RNN au LSTM est plug and play, il suffit juste de rajouter le tuple (h0,c0) en input et il n'y a aucun problème de resize.

Pour le ring convo en 6 + LSTM, en 100 epochs sur GPU on est a 80%(!). En convo 16+LSTM, c'est 86% mais avec 0 de loss à la fin. Ce n'est pas un overfit puisqu'un dropout de 20% dans le LSTM empêche de faire tomber le loss a 0 mais ne réduit pas les performances.
Enfin, un dropout de 20% avant la sortie donne a peu près le même résultat (87% acc.)

Dernière des trois architectures possible, le fait de connecter la moitié des CNN avec l'autre moitié des CNN pour faire une espèce de ring qui connecte 8 a 8 les deux moitiés du cortex. Ca marche et ça prédit à 86% sur le test avec le drop dans la sortie. Avec le drop dans les connexions latérales (le LSTM), on a 86% aussi, mais ça parait plus bioréaliste que la selection se fasse dans les connexions latérales.

## Ring
TODO : rajouter la visualisation des dictionnaires et faire l'intégration ring+saccades, pour le modèle ring LSTM et pour le modèle ring split LSTM.
TODO : Faire de l'exploration d'hyperparametres pour booster l'accuracy :
* dropout
* forme des couches
* taille de l'entrée (128 > 64 ?)
* taille de la couche cachée du LSTM

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
