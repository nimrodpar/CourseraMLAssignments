# Coursera Machine Learning Course Notes

Week 7 (Unsupervised Learning and Clustering)
-
* The **Cost Function** (a.k.a *distortion*) for k-means *should never grow*. Every step of the algorithm (either repositioning the centroids of picking a new cluster for each of the points) should only decreas the cost.
* **Random Initialization** determines the how well the classification fits the data. If you have a small number of clusters (K = 2..10) then it's worth running k-means several (100) times to try and find the best classification. However is K is in the hundreds, it is pointless and usually a single run will be as good as the rest.
