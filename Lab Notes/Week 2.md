# 2018-04-30 - B_theta with real eyes
Le code pour tester les orientatations et voir les perfs comparés au réseau de neurones marche : l'erreur était dans matplotlib. Pour montrer une image :

    plt.imshow(Image.read('path'))
    plt.show()

*A priori* on retrouve la même chute de détection mais plus vers 50° que 25° (à tester sur plus qu'une personne).

J'ai installé CHAMP et dépendances, il demande trop de RAM pour être utilisé sur le laptop.

# 2018-05-01 - GPU at home
J'ai installé Ubuntu 18.04 sur le GPU chez moi pour pouvoir faire tourner PyTorch le WE. Pour démarrer :

    F11 > Ubuntu > hugo

Pour faire tourner CUDA sur U-18.04, il faut prendre la 9.0 et refuser les install de drivers. Ensuite c'est classique.

Si jamais les drivers empêchent de démarrer, il faut modifier GRUB (e sur la distribution) et modifier

    ro quiet slash
    en
    ro nomodeset quiet slash




# A lire
* [Pytorch Logistic Regression](https://github.com/vinhkhuc/PyTorch-Mini-Tutorials) - Pour la suite
* [Optimal eye movement strategies in visual search](https://liberalarts.utexas.edu/files/1516227) - Intégration what+where, discuter avec Pierre Albigés.

# Extras
* [Markdown cheatsheet](https://support.zendesk.com/hc/fr/articles/203691016-Formatage-de-texte-avec-Markdown) - Pour faire des jolis rapports
* [LogGabor sur Wikipedia](https://www.wikiwand.com/en/Log_Gabor_filter) - En complément de la doc
* [Générer des graphs de réseau PyTorch](https://github.com/szagoruyko/pytorchviz) - Pour faire un joli rapport
