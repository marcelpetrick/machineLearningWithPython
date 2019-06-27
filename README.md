# machineLearningWithPython
notes and snippets from the VHS-course for machine learning with Python

[todo: add the notes taken with pen&paper]
[todo: translate all the german words/phrases]

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

- exmaple in Python:
