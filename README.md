# Stage M1 Neurosciences
![Could not display banner](https://amidex.univ-amu.fr/sites/amidex.univ-amu.fr/files/logo_amidex-rgb.jpg)
### English below
Stage de première année de Master de Neurosciences avec [Laurent Perrinet](https://invibe.net/LaurentPerrinet/HomePage).
Poster disponible !

# GDR Vision 2018 Abstract
The selectivity of the visual system to oriented patterns is very well documented in a wide range of species, especially in mammals. In particular, neurons of the primary visual cortex are anatomically grouped by their preference to a given orientated visual stimulus. Interactions between such groups of neurons have been successfully modeled using recurrently-connected network of spiking neurons, so called "ring models". Nonetheless, this selectivity is most often studied with crystal-like patterns such as gratings. 

Here, we studied the ability of human observers to discriminate texture-like patterns for which we could quantitatively tune the precision of their orientated content and we propose a generic model to explain such results. The first contribution shows that the discrimination threshold as a function of the precision did not vary smoothly as would be expected, but more in a binary, "all or none" fashion. Our second contribution is to propose a novel model of orientation selectivity that is based on deep-learning techniques, which performance we evaluated in the same task. This model has human-like performance in term of accuracy and exhibits qualitatively similar psychophysical curves. One hypothesis that such a structure allows for the system to be robust to noise in its visual inputs.
# Organisation du dépot
Les notebooks se trouvent dans le folder /Notebooks et sont datés par AAAA-MM-JJ. Les notebooks qui correspondent a une version finale et fonctionnelle du code ne sont pas datés.

Dans le dossier Lab Notes se trouve le carnet de développement qui contient aussi quelques fixs pour les bugs les plus courants.

# Dépendances
Les notebooks requièrent les packages suivants :
* numpy
* matplotlib
* pytorch
* psychopy + pygame
### Pour la génération de stimuli visuels
* [MotionClouds](http://motionclouds.invibe.net/install.html)

### Pour la production de saccades 
* [LogGabor](https://pypi.org/project/LogGabor/)
* [SparseEdges](https://pypi.org/project/SparseEdges/20171205/#description)


# Internship M1 Neurosciences
A Neurosciences MSc. internship repo for a model built under the supervision of [Laurent Perrinet](ttps://invibe.net/LaurentPerrinet/HomePage).
Now with a poster ! 

# Repo structure
All notebooks are located in the /Notebooks folder and are dated in a YYYY-MM-DD format. Notebooks that are not dated are considered stable and can be use to check out the code.

In /LabNotes are located roadmap objectives and quick fixes for recurrent bugs that occured during developpement (in French).

# Dependancies
Usuals :
* numpy
* matplotlib 
* pytorch
Not so usual :
* psychopy, along with pygame
### To generate visual stimuli
* [MotionClouds](http://motionclouds.invibe.net/install.html)
### To generate saccadic gaze (edge extraction)
* [LogGabor](https://pypi.org/project/LogGabor/)
* [SparseEdges](https://pypi.org/project/SparseEdges/20171205/#description)
