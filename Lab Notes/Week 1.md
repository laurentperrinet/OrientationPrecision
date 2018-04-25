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


# 2018-04-24 - Toying around with Pytorch and MC
### Pytorch
Pytorch est un framework de machine learning qui utilise des graphs de computations dynamiques et permet de gagner en performance sur Keras, Theano, TF... Un exemple de réseau convo :

    class Net(nn.Module):
      def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 10)

On peut définir la fonction de chaque layer :

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 5 * 5)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x
Et résumer le modèle

    model = Net()
    print(model)

### Imageio
Pour sauvegarder les figures en loseless et sans annotations de matplotlib, j'utilise imageio :

    pip install Imageio
    imageio.imwrite('./figs/B0 %s.png' %(B_theta*180/np.pi), im[:, :, 0])

### MotionClouds
Pour générer des MC statitiques avec B_Theta entre 1° et 15° :

    bw_values = np.pi*np.logspace(-7,-3.5, N_theta, base=2)

entre 15° et 30° :

    bw_values = np.pi*np.logspace(-3.5, -2.5, N_theta, base=2)

entre 30° et 45° :

    bw_values = np.pi*np.logspace(-2.5, -2, N_theta, base=2)

# 2018-04-25 - PyTorchConv vs MC
### Output MC
Les images statiques MC sortent en 256*256 et en greyscale (1D).

### PyTorch shapes
Les layers [Conv2D](http://pytorch.org/docs/master/nn.html) de PyTorch prennent en argument :

    Conv2d(inchannel, outchannel)

Les tenseurs qui sortent du DataLoader sont au format imagenumber x nbr_channels x width x height

La fonction tensor.view() permet de changer la forme d'un tenseur, comme numpy.reshape (il faut bien sur garder le même nombre d'éléments totaux). Par [exemple](https://stackoverflow.com/questions/42479902/how-view-method-works-for-tensor-in-torch) :

    x = x.view(-1, 16 * 4 * 4)

Permet de laisser a Pytorch calculer le nombre de lignes (-1) en lui donnant le nombre de colonnes. Dans le CNN, on s'en sert pour passer la feature map au fully connected layer.

Penser a calculer la bonne taille de sortie des convolutionnés et a faire des convolutions de taille raisonnable.

On a un size mismatch entre le fully connected et le premier linear et une erreur de format de sortie.
Pour arranger la forme du tenseur de sortie sur 2D :

    tensorname.permute(axis,axis)

# A lire
* [Numpy Logspace](https://docs.scipy.org/doc/numpy/reference/generated/numpy.logspace.html) - From numpy docs
* [Numpy Geomspace](https://docs.scipy.org/doc/numpy-1.14.0/reference/generated/numpy.geomspace.html) - Comme logspace sans devoir convertir les endpoints en base
* [MotionClouds paper](https://www.physiology.org/doi/pdf/10.1152/jn.00737.2011) - Pour les définitions des variables
* [MotionClouds Orientation](http://motionclouds.invibe.net/posts/static-motion-clouds.html#a-simple-application:-defining-a-set-of-stimuli-with-different-orientation-bandwidths) - Code relatif à la détection de l'orientation
* [Import custom dataset PyTorch](http://pytorch.org/tutorials/beginner/data_loading_tutorial.html) - Dataset class pour imports custom et Torchvision pour les MC statiques de 2018-04-24.

# Extras
* [Markdown cheatsheet](https://support.zendesk.com/hc/fr/articles/203691016-Formatage-de-texte-avec-Markdown) - Pour faire des jolis rapports
* [LogGabor sur Wikipedia](https://www.wikiwand.com/en/Log_Gabor_filter) - En complément de la doc
* [Générer des graphs de réseau PyTorch](https://github.com/szagoruyko/pytorchviz) - Pour faire un joli rapport
