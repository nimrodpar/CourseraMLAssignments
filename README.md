# Coursera Machine Learning Course Notes

Week 7 (Unsupervised Learning and Clustering)
-
* The **Cost Function** (a.k.a *distortion*) for k-means *should never grow*. Every step of the algorithm (either repositioning the centroids of picking a new cluster for each of the points) should only decreas the cost.
* **Random Initialization** determines the how well the classification fits the data. If you have a small number of clusters (K = 2..10) then it's worth running k-means several (100) times to try and find the best classification. However is K is in the hundreds, it is pointless and usually a single run will be as good as the rest.
* When **picking the number of clusters**, it's worth a shot to use the *Elbow Method*, but don't count on it.
* **Dimentionality Reduction**
 * When the dataset can be projected into lower dimentionality (say 2D data points seem to be on a straight line), it's effective to project it onto that lower dimentionality and get better performance\a way to visualze the data.
 * **Principal Component Analysis** is a way to automatically do the former. It basically means that to reduce the dimentionality to K, we simply find a set of K vectors which have minimal distance from *all* data points. These make up a K-dimetional plane which we project the data onto.
   * To choose the best K, use the SVD decomposition of Sigma `[U,S,V] = svd(Sigma)` and find a K s.t. `(S11 * ... * Skk) / (S11 * ... * Snn) >= 0.95` i.e. this K will retain 95% of the variance.
    * Make sure you run PCA (and produce the K vectors a.k.a. `Ureduce`) on the *training set only*, and not on the cross validation or test set.
    * PCA should not be used to prevent over-fitting! (this bad idea stems from the rational that fewer features ==> less overfitting).

Week 8 (Anomaly Detection)
-
* In order to evaluate how well the anomaly prediction performs:
  * Split the good (non-anomaly) exampled of your data set 60-20-20% between the traning, CV and test sets respectively.
  * Split the bad examples (anomalies) 50-50% between the CV and test sets.
  * Use the Precision/Recall or the F1 metric to evaluate the performance of your model on the CV or test set. 
* Pick the value of `Epsilon` which **maximizes** the evaluation metric on the CV set.
* When to use Anomaly Detection vs. Supervised Learning:
  * If you have a fairly large number of positive and negative examples to for the model, and predict that the future examples will look similar to the data set, then use SL.
  * If you have a small number of anomalies, in comparision to the number of good examples, and have no idea whether the future examples will look similar to the anomalies you already have, then use AD, since it's better at prediction "what's not good".
* If you look at your features and they are not distributed on a Gausian, you can benefit from transformaing the feature into a Gausian, for instance by using either `log(x+c)` or `x^c`, where you play with `c` until you get the best looking Gausian. 
