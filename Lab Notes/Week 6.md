# 2018-05-28 - CNN with BCE
## Changer de label
Pour se rapprocher de la justification psychophysique, on va faire un nouveau réseau convolutionnel simple et observer la variation de performance en fonction de Btheta.
Le label est cos(theta-theta0)/4 btheta carré, avec theta le vrai theta et theta 0 la moyenne des thetas.

Il faut donc commencer par générer des MotionClouds organisés par Btheta. On va utiliser le même système que dans le clouds_boundary, c'est a dire diviser les folders en 50 chunks de Btheta croissant, avec au sein de ces folders 8 thetas possibles, pour 6 MC dans chaque thetas. Donc :

    50x8x6 (train) + 50x8x3 = 3600 MC

Le code du nouveau label :

    B_theta = np.mean(chunk) * 180/np.pi
    print('Mean B_theta = %s' %B_theta)

    Theta_zero = np.mean(np.linspace(0,np.pi,16))
    print('Theta_zero = %s' % Theta_zero)

    target_list =[]
    for i,t in enumerate(target):
       Theta = data_set.classes[target[i]]
       target_list.append(float(Theta))
    print('List of thetas = %s' % target_list)

    new_label = []
    for lab in range(len(target_list)):
       new_lab = math.cos(target_list[lab]-Theta_zero)
       new_lab = new_lab/4 * (B_theta) ** 2
       new_label.append(float(new_lab))

    print('New labels %s' % new_label)
    new_label = torch.LongTensor(new_label)
    new_label = Variable(new_label).cuda()

## BCELoss
Pas de convergence alors que ça marchait la veille, en utilisant le vieux label. Ce n'est pas un problème de topologie du réseau, on a beau faire varier la taille des couches dans tout les sens rien ne change. Jouer sur le LR non plus.

Le one_hot_encoding marche bien, on va essayer de le mettre avant et d'utiliser cette liste one_hot_encoded de labels plutot que de les encoder dans la boucle d'entrainement. Ca a l'air de marcher un peu mieux, maintenant il faut reorganiser le code autour de ça.

TODO : Essayer la BCE/BCEWLL sur le LSTM ? La littérature soutient celle utilisation plutôt.

Pour faire un prior on peut donner des weights au BCELoss

# 2018-05-29 - Wrapping BCE up
Le network simple marche avec le nouveau label, maintenant il reste a faire le plotting de (Theta-Theta0)², la variance, en fonction de l'augmentation de B_theta. Ensuite on essayera en BCE.

J'ai changé le network pour remettre celui de CIFAR qui marche bien mieux ( 2CNN, 3 RELU).

Par contre pas possible de faire marcher ça avec le Cross Entropy Loss, donc il faut utiliser le BCE.

* d

# La présentation approche
* [Rules for a good poster](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC1876493/)
* [NYU advice for posters](http://www.personal.psu.edu/drs18/postershow/)
* [Servier Images](https://smart.servier.com/)
