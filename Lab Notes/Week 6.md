# 2018-05-28 - CNN with BCE
Pour se rapprocher de la justification psychophysique, on va faire un nouveau réseau convolutionnel simple et observer la variation de performance en fonction de Btheta.
Le label est cos(theta-theta0)/4 btheta carré, avec theta le vrai theta et theta 0 la moyenne des thetas.

Il faut donc commencer par générer des MotionClouds organisés par Btheta. On va utiliser le même système que dans le clouds_boundary, c'est a dire diviser les folders en 50 chunks de Btheta croissant, avec au sein de ces folders 8 thetas possibles, pour 6 MC dans chaque thetas. Donc :

    50x8x6 (train) + 50x8x3 = 3600 MC

Pas de convergence alors que ça marchait la veille. Ce n'est pas un problème de topologie du réseau, on a beau faire varier la taille des couches dans tout les sens rien ne change. Jouer sur le LR non plus.

Le one_hot_encoding marche bien, on va essayer de le mettre avant et d'utiliser cette liste one_hot_encoded de labels plutot que de les encoder dans la boucle d'entrainement. Ca a l'air de marcher un peu mieux, maintenant il faut reorganiser le code autour de ça.

TODO : Essayer la BCE/BCEWLL sur le LSTM ? La littérature soutient celle utilisation plutôt.

Pour faire un prior on peut donner des weights au BCELoss
# A lire

* d

# La présentation approche
* [Rules for a good poster](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC1876493/)
* [NYU advice for posters](http://www.personal.psu.edu/drs18/postershow/)
* [Servier Images](https://smart.servier.com/)
