# machineLearningWithPython
notes and snippets from the VHS-course for machine learning with Python

[todo: add the notes taken with pen&paper]
[todo: translate all the german words/phrases]

# day1: ML-course (20190627)

pca:
https://en.wikipedia.org/wiki/Kernel_principal_component_analysis

http://www.iangoodfellow.com/

unsupervised learning:
k-means (clusterierung) and PCA - almost everything you will need

nachteil/problem: man muss vorher die gruppenanzahl angeben

EM: expectation maximisation

- estimation of the amount of needed clusters is a np-complete issue

---------------------------------------

# unsupervised learning

* pca: https://en.wikipedia.org/wiki/Kernel_principal_component_analysis
* k-means

# supervised learning

- (classification) Label for Information: rot oder blau fuer Katze oder Hund
- https://de.wikipedia.org/wiki/%C3%9Cberwachtes_Lernen
- https://en.wikipedia.org/wiki/Supervised_learning
- https://datascience.stackexchange.com/questions/9573/supervised-learning-and-labels

- needs a "training set" - like 100 pictures of dog and cat; to spearate them into classes
- how can you derive the information then from the pictures (images) with 10k pixels? --> reduce dimension (feature extraction) -> cat less than 10 kg/ wolf higher than 10kg
- how to separate 0 from 1 on scanned paper? width of the glyph -> very good feature; already 80-90% accuracy (or amount of color; sum of pixels) --> this is called feature engineering ( https://en.wikipedia.org/wiki/Feature_engineering ; but outdated after 2010; useful in the area of Medicine; means "done by a human")
- examples of a ML algorithm: KNN (see: https://en.wikipedia.org/wiki/Types_of_artificial_neural_networks)
- model has several parameters: how to tune those parameters? so we have the least deviations; but those parameters are not set by us --> this means "learning"
- computer is never learning something outside of those parameters: a and b, or 
- ResNet: 450 mio parameters - https://cv-tricks.com/cnn/understand-resnet-alexnet-vgg-inception/
- it is always about finding the best model: but which line separates the given training groups best?
- one line fits better, because the average perpendicular length is higher

- SVM: support vector machine: https://de.wikipedia.org/wiki/Support_Vector_Machine ;  Vapnik/Schölkof
- how to test the algorithm? throw more data on it; mabye take 80 images for training and 20 for verification; or 70/30 - https://www.researchgate.net/post/Is_there_an_ideal_ratio_between_a_training_set_and_validation_set_Which_trade-off_would_you_suggest // https://en.wikipedia.org/wiki/Resampling_(statistics)#Jackknife
- overfitting of bad sampling: very bad selection of the training set, so that the model is proper there, but then the "real world" is totally different - https://en.wikipedia.org/wiki/Overfitting
- regression: create a curve from given 2d-data-points
- nearest neighbour
- K-NN: specialisation 2-NN for dog/cat; underfitting - never an opinion, smaller values of K - to jumpy; too big values - to rigid or not useful

to read for myself:
- https://cv-tricks.com/object-detection/faster-r-cnn-yolo-ssd/
- https://en.wikipedia.org/wiki/Residual_neural_network

# reinforcement learning
- third kind of ML (not part of this course)

-------------------------

# exmaple in Python:

- install: https://www.python.org/downloads/
- P 3.7; add python to path; 

# good hints for reading (in this order): - not that they really have to be bought from the big A
- Pedro domingos - the master algorithm - kaufen und lesen; umgangssprachlich verständlich: https://www.amazon.de/Master-Algorithm-Ultimate-Learning-Machine/dp/0465065708
- hands on machine learning aurelie geron - https://www.amazon.de/Hands-Machine-Learning-Scikit-Learn-TensorFlow/dp/1491962291
- deep learning - yoshua goodfellow: https://www.amazon.de/Deep-Learning-Adaptive-Computation-Machine/dp/0262035618

good coursera course: Andrew Ng: https://de.coursera.org/learn/machine-learning - also the founder of coursera


pip install --upgrade -user scipy numpy scklearn

IRIS data set: classify flowers

(lol: https://images.app.goo.gl/SdDZCh7gEP3JHLUw7 )

- jupyter notebooks

pip install --user jupyter
https://www.datacamp.com/community/tutorials/tutorial-jupyter-notebook - how to create and run the very first notebook
(install this and create a notebook ..)

IDEs: Keras / TensorFlow / PyTorch

--> ? maybe this ?
https://medium.com/syncedreview/andrew-ng-says-enough-papers-lets-build-ai-now-6b52d24bcd58


 https://medium.com/syncedreview/andrew-ng-says-enough-papers-lets-build-ai-now-6b52d24bcd58
 introduction for the reinforcement learning: https://hci.iwr.uni-heidelberg.de/system/files/private/downloads/541645681/dammann_reinfocement-learning-report.pdf
 
 
training set/test set
features/feature engineering
overfitting bias

**mail for the guy:** 
Pouyan.Mohajerani@gmail.com
--> send something later

MACHINE LEARNING YEARNING - 118 pages Andrew Ng - very easy to udnerstand; but it looks like thereis no printed version:
https://towardsdatascience.com/6-concepts-of-andrew-ngs-book-machine-learning-yearning-abaf510579d4
Andrew NG - Machine learning yearning: bookform (just pdf): https://github.com/ajaymache/machine-learning-yearning or as chapters on github https://github.com/ajaymache/machine-learning-yearning

# ----------------------------------------------------------
# Second day of ML-course (20190704)

- account for nvidio for cudnn (cudo) - 2 GiByte thingy
- folder namend "ML" was prepared by the guy

### what is overfitting?
resultat passt nur rein zufällig (?); testdaten passen, reale nicht

### ist underfitting oposite to overfitting
test- und realdaten passen nicht

wir: hautpsächlich supervised learning (curved fitting)
- versuch mit ax^2+bx+c auf 4 datenpunkte: immer underfitting
 
 - warum gerade 4er gerade bei sinx(x) overfitting: trifft genau die testdaten
 
TODO: check overfitting
TODO: check underfitting

- ML braucht Expertenwissen: mit diesem expert knowledge erzeugt man feautres -> feature engineering
- feature engineering war hauptmethode bis man CNN (convolutional neuro networks) hatte
- wie verwendet man aber expert knowledege (siehe Iris-Datenbank)?

Beispiel: Python notebook: IRIS.ipynb

-- einfach ein neues erzeugen; von cmd aus; mit perfektem PATH
$ jupyter notebook (ENTER)
dann: http://localhost:8888/tree
dann das notebook laden
(Jupyter: app dev in browser :))

do: menue: Kernel >> Restart&clear output 
(hints lead to https://www.kaggle.com/skalskip/iris-data-visualization-and-knn-classification )

- dies ist eine volle KI-app
- Anfang: load some libs
- mit CTRL+ENTER Zelle evaluieren
- was ist numpy? Lineare Algebra

- lol: https://stackoverflow.com/questions/13187778/convert-pandas-dataframe-to-numpy-array

- krass: dataset.describe()

- über kaggle und competitions wird am letzten abend gesprochen: like online gambling
- he says: wie glücksspiel; macht extrem süchtig

- sklearn: new player; very important
-- is sci kit learn

- PCA
- K means
- k-NN
--> all three are: shallow (seicht) learning; viele davon in SciKitLearn enthalten
- Deep Learning

- LabelEncoder: transofrmiert Versicolor, Virginica to 0, 1, 2 ..

- die ausgabe der letzten zeile wird in den zellen des jupyter-nb gezeigt
- was habenn wir? 150 daten: 4 features: und ein label

- wir wollen fuer diesen datensatz fuer ein neues test-datum dann wissen, welche blumenart dies ist
- warum als 0,1,2,3 gelabelt und nicht 0,1,2,7? plotten einfacher
- aber falls man ein curve-fitting will, dann wäre es besser wenn sie gleichabständig sind
- darunter steckt aber cost-function: wie schlecht ist ein fehler?

- found myself: 5 cent per hour?
https://aws.amazon.com/marketplace/pp/B06Y6BNHD3
krasser shit: https://aws.amazon.com/marketplace/pp/B07MFRDXTB?qid=1562259741087&sr=0-1&ref_=srh_res_product_title  ... von fast kostenlos bis 3$/h umgebungen
- wie macht man auswahl der testdaten? random 80% von Klasse 0, 50% von Klasse 0, ... und fügt sie dann zusammen
- 10 krekbspatientetn und 1000 ohne patienten: 8/800 training, 2/200 für test
- wird sich besser an gesunde anpassen
--> class imbalance (todo check)

stratification: https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.train_test_split.html
--> schichten bilden

subplot: tabellnähnlicher plot
- die drei farben sind die verschiedenen Iris-klassen
- second feature plotted against first feature
- plot: 1 versus 4; fast alle roten sind rechts, alle grünen sind mitte, blau sind rechts
- oder plot 4 versus 3: auch ein paar fehler, aber viel weniger
- ein learner sucht sich solche "gut trennbaren" geegenden im gesamten space so dass die geringste anzahl an fehlern auftauchtkurs@B112PC12 MINGW64 ~/Desktop/mpe/ml_course (master)

- Voraussage angeben: y_pred = classifier.predict(X_test[:,[0,1,3]])
- 3NN angewendet
- classifier prediction: fast 97% genauigkeit accurarcy über test-set
- falls man die nachbarn auf 50 nachbarn packt, kommt man auf accuracy of 80% ... ist schlechter
- bei 100-nn nur 70%
- welche features man wählt, ist extrem wichtig: 4 features
- aber mit zwei features ist man schon ausreichend gut bedient
--> erinnert an PCA (dimensionality reduction)

(wichtige) Frage: wie wählt man jetzt dieses K?
- da sModel hat innere und auessere parameter
- hyper-parameters passt man mit fit gut (automatisch) an:
https://en.wikipedia.org/wiki/Hyperparameter_(machine_learning)

- essen heute abend: das menü ist der hype-parameter; muss vorher festgelegt werden
- hyperparameter: ist die wahl vom essen
- parameter: wieviel von was - davon abhängig

- hyperparameter-wahl kann auch schon overfitting bedeuten
- aus trainignsdatensatz kann man auch einen teil herausnehmen: validation (dritte menge)
- validation-parameter-set um hyperparameter anpassen

- artikel auf linkedin: als weitere erläuterung
- hyperparameter wählen: welches modell, ..

- (lookup): cross validation
- erst 1,2 als validation und ret test; dann 9,10 als valid vorheriger teil test: durchschnitt dieser ganzen validations ist dann ergebnis für k -> cross validation
- 20% splits --> 5 fold validation
- looks like nvidia gpu is best: but quick search found some converter https://stackoverflow.com/questions/10043974/how-to-run-my-cuda-application-on-ati-or-intel-card-in-software-mode

NEUES PROJEKT:
- ramon cachal: zellen für bilderkennung; spanischer neurowissenschaftler
- erster mit vorschlag, dass gehirn nettzwerk von neuronen ist - kein festes objekt
https://neuroscientificallychallenged.com/blog/history-of-neuroscience-ramon-y-cajal
learn: https://www.coursera.org/learn/synapses ? 

- grundlegendes problem: kurve zu punkten anpassen
bertrand russel: turkey problem - humes problem of induction
alle leute, die ich kenne und die gestorben sind, waren nicht ich -> ich werde nicht sterben

- wissen über die welt in model einbauen: (lookup) Regularisation --> https://towardsdatascience.com/regularization-in-machine-learning-76441ddcf99a

- no free lunch!
- daten, annahmen, 
- which GPU for deep learning?
https://timdettmers.com/2019/04/03/which-gpu-for-deep-learning/ - looks like a good in depth check

generic programming

dendrite, axon, synapse - collect and integrate signals and forward that
- if the signal is bigh enough to be higher than a threshold, then the neuron fires
- looks like a really good comparison: https://towardsdatascience.com/deep-learning-versus-biological-neurons-floating-point-numbers-spikes-and-neurotransmitters-6eebfa3390e9

i do kaggle? https://www.kaggle.com/

https://www.kaggle.com/cdeotte/supervised-generative-dog-net

- aktivierungsfunktion
- mit 13 parametern kann man eine sehr vielgefaltete kurve erzeugen - nicht nur eine gerade (wie mit 3 punkten)

- entscheidung treffen als CEO einer firma: preis eines produktes steigern
-- mit leuten reden: marketing, finanzen, .... diese selbst lesen auch ihre "werte" vom "markt/nutzerbasis"
-- preise gesteigert: stellt sich am ende des jahres als fehlentscheidung heraus: was macht man dann?
-- marketing lag aber richtiger als finanzen: jetzt also marketing mehr vertrauen geben und finanzen weniger; marketing hat das aber auch nur weitergemeldet bekommen
- (reaserch) back-propagation! <--- very important https://de.wikipedia.org/wiki/Backpropagation
- sehr einfach algorithmus

- Initialisation of NN - zufällige werte am Anfang
-- dann in weiterer Iteration mit anderem Satz von Parametern machen
-- Backprop (check)
-- learning rate (check)

stuff for my own homelearning / certificates:
* https://www.nvidia.com/de-de/deep-learning-ai/education/
* https://www.academy.fraunhofer.de/de/weiterbildung/information-kommunikation/data-scientist-schulungen.html
* https://www.datascience-certificate.mathematik-informatik-statistik.uni-muenchen.de/information/index.html
--> Anmeldefrist für das Wintersemester 2019/2020: 18. August 2019 ... 6500 euro ... what the fuck? lernt man da gold zu zaubern?!? 

- backprop macht selber keine änderung im netz: zeigt nur die direktino von änderungen
-- gradient descent
-- recursive gradient descent
-- or adams
-- or msdrop (massdrop)

- learning rate als regulation --> form von regularisation
- begriff von "epochs" (nachschlagen)

# beispiel: 100k datenpunkte; 100k bilder von katzen und hunden - netzwerk trainieren
- aber: dafür bräuchte man sehr, sehr viel RAM
- stattdessen: teilt man in 1000 gruppen von 100 bilder --> "mini-batch" https://en.wikipedia.org/wiki/Online_machine_learning#Batch_learning
- https://machinelearningmastery.com/gentle-introduction-mini-batch-gradient-descent-configure-batch-size/

- https://machinelearningmastery.com/inspirational-applications-deep-learning/

## neues praxisbeispiel: Standard (Fully Connected) Neural Network
--> FCNN_MNIST_light.ipynb

MNIST dataset --> hangeschriebene Ziffern
(knn - scikit)
(KEras for fully connected neural network; for tensorflow) - Keras vereinfacht das Leben
- Rest des Kurses nur noch Keras

60k bilder: ein bild hat 28x28 pixel
0..255 for luminance

one hot encoding? https://hackernoon.com/what-is-one-hot-encoding-why-and-when-do-you-have-to-use-it-e3c6186d008f

bild mit 28x28 pixel
-> 768 werte im bild (jedes pixel ein feature)
-> linearisiert: vektor von 768 werten -> diese werden dann 768 input nodes
-> dann zwischen-schichten --> das ist wahl der hyperparamter
-> alles mit allem verbunden
-> am ende 10 nodes in einer schicht (wieder sichtbare schicht)
- wenn mehr als eine versteckte schicht --> "deep"
- wenn zuviele schichten, dann overfitting; also zentrale frage: wie traininert man dies jetzt?

- 784 -> 400, 200, 10 -> diese schichten (?)
- read: https://ml-cheatsheet.readthedocs.io/en/latest/layers.html
- softmax layer
- in notebook: "Training The Model" ist dann wirklich das training des modells, das eigentliche ML
- in notebook: "Testing the Model" ist dann vergleich der vorhersagen mit dem model

- accuracy von 92% am ende ...

- mit 30 epochs dann 96% genauigkeit

- auto encoder: alle werte eingeben; output sind auch alle werte; in der mitte bottleneck mit nur 20 werten -> zum training eineinziges bild -> gleiches bild wird erwartet; was in der mitte passiert ist dimensionality reduction -> nur 20 werte werden gebraucht
--> ergebnis wie PCA (informationsflaschenhals -> durch die enge bekommt man die essenz)
read: https://de.wikipedia.org/wiki/Autoencoder

# homework
How to use validation data for Keras?
( https://stackoverflow.com/questions/46308374/what-is-validation-data-used-for-in-a-keras-sequential-model )
How to draw then the history data with Keras?

--> read and do: learning curves
https://stackoverflow.com/questions/37664783/how-to-plot-a-learning-curve-for-a-keras-experiment
how to plot them?
https://en.wikipedia.org/wiki/Autoencoder

--> add this https://de.linkedin.com/in/pmohajerani

# ----------------------------------------------------------
# tThird day of ML-course (20190711)

- get those two archives and unzip them (inside the cats-vs-dogs)
- vorher: ursprünglich parameter und Hyperparameter
- Zifferklassifikation hatte keine Features, nur rohe Daten: wie soll man da welche bestimmen?
- eigentlich Arbeit ist eine Linie (Kurve) zwischen den Gruppen zu zeichnen
- (Question to myself: how to separate three classes from each other? always reducible to 1vs1 problem?)
--> Anpassung der Kurven: fit (via Python, notebooks) - aber die Frage welchen Algo (ist Hyperparameter) man wählt, ist dem Dev überlassen
- es gibt kein Backprop für HyperParameter
 
- Thomas Bayes
P (H|B) = P(B|H)*P(H)/P(B)
https://de.wikipedia.org/wiki/Bayessches_Netz

looks great t read: https://towardsdatascience.com/@artem.oppermann

- Beispiel: zwei Bekannte mit gleichem Krebs; Daten vorhanden, recht viele Metadaten
-- beide haben gleiches Medikament genommen: fuer einen wirkte es, fuer den anderen nicht: was jetzt?
-- Daten: 4 Billionen FeatureWerte (Schätzung)
-- wenn man jetzt eine Kurve zwischen diese 4 Mrd Daten findet, hätte man Krebs geheilt -> sehr kompliziertes Klassifikationsproblem

- also: https://de.wikipedia.org/wiki/Markow-Kette
- read: https://towardsdatascience.com/backpropagating-ais-future-377816fc07fa

- read: https://towardsdatascience.com/5-useful-statistics-data-scientists-need-to-know-5b4ac29a7da9


books:
https://www.amazon.in/Life-3-0-Being-Artificial-Intelligence/dp/1101946598
https://www.amazon.in/Superintelligence-Dangers-Strategies-Nick-Bostrom/dp/0199678111

back to the mnist example:
https://jupyter-notebook.readthedocs.io/en/stable/examples/Notebook/Running%20Code.html

google: AutoML - projekt (check this)

- MNIST notebook: einfach mal einen weiteren layer einfuegen: sauschnell

* for the visualisation: https://keras.io/visualization/ - also how to use the matplotlib
* waurm ist 98% erkennung noch nicht gut genug?
-- bei überweisungsträgererkennung wäre selbst 99,5% sauschlecht, weil einfach dann 1 von 200 falsch ist


- https://github.com/keras-team/keras/ - on top of other tensorflow, etc libs
- basic explanation of the things offered by Keras: https://machinelearningmastery.com/display-deep-learning-model-training-history-in-keras/

- from: https://machinelearningmastery.com/faq/
-- "Do you have examples of the Restricted Boltzmann Machine (RBM)?
I do not have examples of Restricted Boltzmann Machine (RBM) neural networks.
This is a type of neural network that was popular in the 2000s and was one of the first methods to be referred to as “deep learning”.
These methods are, in general, no longer competitive and their use is not recommended.
In their place I would recommend using deep Multilayer Perceptrons (MLPs) with the rectified linear activation function."

- what is regularization? wenn man eine 3 als Bild dm nezt gibt, dann noch einmal eine geshiftete 3 und dann wieder ... model wuerde sich daran anpassen (eine art von regularisation)
- data augmentation: https://medium.com/nanonets/how-to-use-deep-learning-when-you-have-limited-data-part-2-data-augmentation-c26971dc8ced
- https://neuralocean.de/index.php/de/2018/03/28/data-augmentation-was-ist-das/
- https://neuralocean.de/index.php/de/2018/04/22/programmieren-von-kis-fuer-4x4-tictactoe-part-4-eine-experten-ki/

- model selection:

- how to deploy the AI? https://machinelearningmastery.com/deploy-machine-learning-model-to-production/

# convolutional neural networks
- https://de.wikipedia.org/wiki/Convolutional_Neural_Network
- Yann LeCun

- Götter der ML: https://www.nzz.ch/digital/ehre-fuer-die-deep-learning-mafia-ld.1472761

Lenet5: sehr gut für das MNIST problem mit 5 Layern
https://medium.com/@pechyonkin/key-deep-learning-architectures-lenet-5-6fc3c59e6f4

-> subsampling um aus einem bild mehr bilder zu machen ( also called maxpooling)
"By modern standards, LeNet-5 is a very simple network. It only has 7 layers, among which there are 3 convolutional layers (C1, C3 and C5), 2 sub-sampling (pooling) layers (S2 and S4), and 1 fully connected layer (F6), that are followed by the output layer. Convolutional layers use 5 by 5 convolutions with stride 1. Sub-sampling layers are 2 by 2 average pooling layers. Tanh sigmoid activations are used throughout the network. There are several interesting architectural choices that were made in LeNet-5 that are not very common in the modern era of deep learning."



LSTM model?
https://machinelearningmastery.com/deep-learning-for-time-series-forecasting/

deep learning for coders:
-----> https://course.fast.ai/

run this notebook now:
http://localhost:8888/notebooks/3rdSession/Lenet5.ipynb

good introduction for job preparation:
https://blog.usejournal.com/what-i-learned-from-interviewing-at-multiple-ai-companies-and-start-ups-a9620415e4cc

ensemble model

chapters 2,3,4: http://www.deeplearningbook.org/
for the theorethical foundations of ML

# now: http://localhost:8888/notebooks/3rdSession/CatDog.ipynb
( maybe use this for understanding: https://github.com/girishkuniyal/Cat-Dog-CNN-Classifier )

https://de.wikipedia.org/wiki/Fluch_der_Dimensionalit%C3%A4t

( next: transfer learning )

-------------------
# own notes:

watch this: https://www.youtube.com/watch?v=cAICT4Al5Ow
reading (especially the cat-vs-dg-example for the jupyter notebook)
* https://towardsdatascience.com/cat-or-dog-image-classification-with-convolutional-neural-network-d421a9363c7a
* https://towardsdatascience.com/image-classifier-cats-vs-dogs-with-convolutional-neural-networks-cnns-and-google-colabs-4e9af21ae7a8
* deep learning with Python: https://www.manning.com/books/deep-learning-with-python
* https://pythonprogramming.net/convolutional-neural-network-kats-vs-dogs-machine-learning-tutorial/

* keras opencv and matplotlib and scikit had to be installed
* Cuda from nvidia in version 9 missing, but installing 10 maybe better (let us try and see)
* https://developer.nvidia.com/cuda-10.1-download-archive-base?target_os=Windows&target_arch=x86_64&target_version=10&target_type=exelocal
* what if there is an AMD GPU?!?

-------------------
# fourth day of the course

- stamping function: warum braucht man schwellenwert?
- nonlinear function: https://www.quora.com/Why-does-deep-learning-architectures-only-use-the-non-linear-activation-function-in-the-hidden-layers?source=post_page---------------------------

- LeakyReLU, maxout, tanh, sigmoid, ELU, ReLU

read this: https://medium.com/cracking-the-data-science-interview/the-10-deep-learning-methods-ai-practitioners-need-to-apply-885259f402c1
do this: https://medium.com/abraia/getting-started-with-image-recognition-and-convolutional-neural-networks-in-5-minutes-28c1dfdd401 - also with colab page
do this: https://eu.udacity.com/course/intro-to-tensorflow-for-deep-learning--ud187

multi-label image recognition? https://towardsdatascience.com/fast-ai-season-1-episode-3-a-case-of-multi-label-classification-a4a90672a889

https://towardsdatascience.com/image-detection-from-scratch-in-keras-f314872006c9 - also with hints how to make jupyter

laut meiner graphk mit 500 Epochen:
- anfangs overfitting
- aber halluzination: model lernt dinge, die nicht richtig waren
- model hat gelernt, dass meiste katzen heller waren als Hunde -> aber das ist falsch, das ist Overfitting, keine richtige Erkenntnis
- underfitting (bias): wenn man immernoch nicht zwischen katzen und hudnen unterscheiden kann
- wie kann das model aber so schlecht lernen?
- falsch gelernt und dies dann verstärkt


# ok, really important! #
----> https://medium.com/abraia/first-steps-with-transfer-learning-for-custom-image-classification-with-keras-b941601fcad5

How to detect cats in real life pictures:
https://www.pyimagesearch.com/2016/06/20/detecting-cats-in-images-with-opencv/

# new example: transfer learning

Wie lernt man Traktor fahren, wenn man schon Auto fahren kann?
- vergleichend lernen; vielversprechende idee
- Berühmtes beispiel: AlexNet https://en.wikipedia.org/wiki/AlexNet
- mit vielen Daten und richtigen Parametern schon sehr lange trainiert

- https://qz.com/1307091/the-inside-story-of-how-ai-got-good-enough-to-dominate-silicon-valley/

read: https://www.learnopencv.com/understanding-alexnet/
read: https://towardsdatascience.com/a-comprehensive-hands-on-guide-to-transfer-learning-with-real-world-applications-in-deep-learning-212bf3b2f27a

important terms!
# maxpooling !
# back propagation !


# mobilenet - other network type (already decapitated for our example of transfer learning)

- erster Ansatz immer: transfer learning
