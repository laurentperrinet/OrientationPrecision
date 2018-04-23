# 2018-04-23 - Début du stage
Today's Journal Club : [Frontal eye field](https://www.frontiersin.org/articles/1a0.3389/fnint.2014.00066/full)
### GPU
Laurent m'a créé un compte sur le babbage pour accéder au GPU. Pour se connecter :

    ssh hugo@10.164.7.21

Pour installer un script uniquement sur mon home directory (donc sans permissions) :

    pip3 install --user

Pour installer pytorch sur mon ordinateur (sans GPU) :

    conda install pytorch-cpu torchvision -c pytorch

Sur babbage qui tourne Cuda 8:

    pip3 install torch torchvision

Pour tester sur babbage ( /!\ ipython3) :

    cd examples/mnist
    ipython3 main.py

### MotionClouds
J'ai installé [MotionClouds](http://motionclouds.invibe.net/) sur ma machine et sur babbage, avec la [dépendance pour la visualisation](http://vispy.org/installation.html).
Les visualisations de vidéos ne marchent pas (dépendances ?)

Le code pour générer des stimulus statitiques se trouve [ici](http://motionclouds.invibe.net/posts/static-motion-clouds.html#a-simple-application:-defining-a-set-of-stimuli-with-different-orientation-bandwidths).

### LogGabor
Pour générer des filtres Log Gabor en Python :

    pip install LogGabor

La [documentation](http://nbviewer.jupyter.org/github/bicv/LogGabor/blob/master/LogGabor.ipynb).



# A lire
* [Numpy Logspace](https://docs.scipy.org/doc/numpy/reference/generated/numpy.logspace.html) - From numpy docs
* [MotionClouds paper](https://www.physiology.org/doi/pdf/10.1152/jn.00737.2011) - Pour les définitions
* [MotionClouds Orientation](http://motionclouds.invibe.net/posts/static-motion-clouds.html#a-simple-application:-defining-a-set-of-stimuli-with-different-orientation-bandwidths) - Code relatif à la détection de l'orientation

# Extras
* [Markdown cheatsheet](https://support.zendesk.com/hc/fr/articles/203691016-Formatage-de-texte-avec-Markdown)
* [LogGabor sur Wikipedia](https://www.wikiwand.com/en/Log_Gabor_filter)
* [Online Gabor Filters] (http://matlabserver.cs.rug.nl/cgi-bin/matweb.exe)
