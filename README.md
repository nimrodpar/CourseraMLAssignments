# Coursera Machine Learning Course Notes

These are mainly caveats and anacdotes from the Machine Learning course at Coursera. These are *not* detailed lecture notes (the excercises contain most of notes already).

## Week 7 (Unsupervised Learning and Clustering)
* The **Cost Function** (a.k.a *distortion*) for k-means *should never grow*. Every step of the algorithm (either repositioning the centroids of picking a new cluster for each of the points) should only decreas the cost.
* **Random Initialization** determines the how well the classification fits the data. If you have a small number of clusters (K = 2..10) then it's worth running k-means several (100) times to try and find the best classification. However is K is in the hundreds, it is pointless and usually a single run will be as good as the rest.
* When **picking the number of clusters**, it's worth a shot to use the *Elbow Method*, but don't count on it.
* **Dimentionality Reduction**
 * When the dataset can be projected into lower dimentionality (say 2D data points seem to be on a straight line), it's effective to project it onto that lower dimentionality and get better performance\a way to visualze the data.
 * **Principal Component Analysis** is a way to automatically do the former. It basically means that to reduce the dimentionality to K, we simply find a set of K vectors which have minimal distance from *all* data points. These make up a K-dimetional plane which we project the data onto.
   * To choose the best K, use the SVD decomposition of Sigma `[U,S,V] = svd(Sigma)` and find a K s.t. `(S11 * ... * Skk) / (S11 * ... * Snn) >= 0.95` i.e. this K will retain 95% of the variance.
    * Make sure you run PCA (and produce the K vectors a.k.a. `Ureduce`) on the *training set only*, and not on the cross validation or test set.
    * PCA should not be used to prevent over-fitting! (this bad idea stems from the rational that fewer features ==> less overfitting).

## Week 8 (Anomaly Detection and Recommender Systems)
#### Anomaly Detection
* In order to evaluate how well the anomaly prediction performs:
  * Split the good (non-anomaly) exampled of your data set 60-20-20% between the traning, CV and test sets respectively.
  * Split the bad examples (anomalies) 50-50% between the CV and test sets.
  * Use the Precision/Recall or the F1 metric to evaluate the performance of your model on the CV or test set. 
* Pick the value of `Epsilon` which **maximizes** the evaluation metric on the CV set.
* When to use Anomaly Detection vs. Supervised Learning:
  * If you have a fairly large number of positive and negative examples to for the model, and predict that the future examples will look similar to the data set, then use SL.
  * If you have a small number of anomalies, in comparision to the number of good examples, and have no idea whether the future examples will look similar to the anomalies you already have, then use AD, since it's better at prediction "what's not good".
* If you look at your features and they are not distributed on a Gausian, you can benefit from transformaing the feature into a Gausian, for instance by using either `log(x+c)` or `x^c`, where you play with `c` until you get the best looking Gausian. 
* If your AD system reports many anomalies as good examples, try coming up with more features (these can be extrpolated from existing features) that better reflect anomalies (i.e. get a very high or low value).

#### Recommender Systems
* You can **extract features** automatically in this setting. Say you have already learned the users perefences vector Theta. And you have a new movie with rating R(i) from the users, but you don't know how to extract the features (say is it a romance/action movie? etc.). You can just apply Theta * R(i) and you'll get the features according to the users the model is already trained on. Then you can use these features to create recommendations for other users.
   * Interestingly, you can do this without really knowing Theta! you can start by guessing Theta, extract features X and then use that to better you initial Theta and so on iteratively.
   * You can perform this simultaneously, by minimizing X and Theta together (as the main term in the cost function for both is the same, and you only need to have the regularization for both).

## Week 9 (Large Scale Machine Learning)
#### Gradient Descent with Large Datasets
* Motivation: "It's not who has the best algorithm that wins, It's who has the most data."
* Still, before working hard on getting more data and working efficiently with it, make sure it actually achieves better performance. Plot the J_cost and J_CV curves and see that you indeed have high variance (overfiting to the small dataset) before you increase the dataset.
