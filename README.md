# Coursera Machine Learning Course Notes

Week 7 (Unsupervised Learning and Clustering)
-
* The **Cost Function** (a.k.a *distortion*) for k-means *should never grow*. Every step of the algorithm (either repositioning the centroids of picking a new cluster for each of the points) should only decreas the cost.
* **Random Initialization** determines the how well the classification fits the data. If you have a small number of clusters (K = 2..10) then it's worth running k-means several (100) times to try and find the best classification. However is K is in the hundreds, it is pointless and usually a single run will be as good as the rest.
* When **picking the number of clusters**, it's worth a shot to use the *Elbow Method*, but don't count on it.
* Dimentionality Reduction
*   When the dataset can be projected into **lower dimentionality** (say 2D data points seem to be on a straight line), it's effective to project it onto that lower dimentionality and get better performance\a way to visualze the data.
*   Principal Component Analysis is a way to automatically do the former. It basically means that to reduce the dimentionality to K, we simply find a set of K vectors which have minimal distance from *all* data points. These make up a K-dimetional plane which we project the data onto.
