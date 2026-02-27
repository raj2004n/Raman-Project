import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

def elbow_plot(data, clusters):
    inertia = []
    for n in range(1, clusters):
        algorithm = KMeans(
            n_clusters=n,
            init='k-means++',
            n_init='auto',
            random_state=125,
        )
        algorithm.fit(data)
        inertia.append(algorithm.inertia_)
    plt.plot(np.arange(1, clusters), inertia, 'o')
    plt.plot(np.arange(1, clusters), inertia, '-', alpha=0.5)
    plt.xlabel('Numbers of Clusters'), plt.ylabel('Inertia')
    plt.show()
