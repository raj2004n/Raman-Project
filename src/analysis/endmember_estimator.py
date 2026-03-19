import kneed
import numpy as np
import ramanspy as rp
from pysptools import material_count
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

def _n_by_pca(hsi_cube):
    # get matrix containing spectral data
    X = hsi_cube.spectral_data

    # prepare data to shape (rows * cols, band_lenght), and scale
    rows, cols, b = X.shape
    X = X.reshape(rows * cols, b).astype(np.float64)
    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    # principal components that capture 80% variance
    pca_80 = PCA(n_components=0.8)
    pca_80.fit_transform(X)
    n_80 = pca_80.n_components_

    # principal components by knee method
    pca_elbow = PCA(n_components=10)
    pca_elbow.fit_transform(X)
    ev = pca_elbow.explained_variance_
    x = np.arange(1, 11)
    kn = kneed.KneeLocator(x, ev, curve='convex', direction='decreasing')
    n_elbow = kn.elbow if kn.elbow else None

    # catch any assumptions that are too high
    if n_80 > 10:
        n_80 = n_80
    
    if n_elbow > 10:
        n_elbow = 1

    """
    # Kaiser's method: Only keep those whose eigenvalues greater than 1.
    # In practice, principal components with eigenvales like 0.95 may still
    # contain significant variance. Hence, this method is not the only one being used.
    pca_kaiser = PCA()
    pca_kaiser.fit_transform(X)
    ev = pca_kaiser.explained_variance_
    # pick out principal components with eigenvalue >= 1
    n_kaiser = np.sum(ev >= 1.0)
    """
    return n_80, n_elbow

def _n_by_hfc(hsi_cube):
    # get spectral matrix
    X = hsi_cube.spectral_data
    # sclae spectral matrix
    X = (X - X.min()) / (X.max() - X.min())

    hfcvd = material_count.HfcVd()
    fars = [1e-2, 1e-3, 1e-4, 1e-5, 1e-6, 1e-7]
    ns_hfc = hfcvd.count(X, far=fars, noise_whitening=True)
    return ns_hfc

def _determine_confidence(ns):
    """
    Simple method to determine the confidence by finding the difference in predictions.

    Possible confidence values:
    - 2 three predictions match
    - 1 two predictions match
    - 0 no prediction match
    """
    p1, p2, p3 = ns
    if p1 == p2 == p3:
        return 'high'
    elif p1 == p2 or p1 == p3 or p2 == p3:
        return 'moderate'
    else:
        return 'low'


def estimate_endmembers(hsi_cube):

    pipeline_pca = rp.preprocessing.Pipeline([
    rp.preprocessing.despike.WhitakerHayes(),
    rp.preprocessing.denoise.Whittaker(),
    rp.preprocessing.baseline.AIRPLS(),
    ])

    pipeline_hfc = rp.preprocessing.Pipeline([
    rp.preprocessing.denoise.Whittaker(),
    rp.preprocessing.baseline.AIRPLS(),
    ])

    hsi_cube_pca = pipeline_pca.apply(hsi_cube)
    hsi_cube_hfc = pipeline_hfc.apply(hsi_cube)
    
    n_80, n_elbow = _n_by_pca(hsi_cube_pca)
    ns_hfc = _n_by_hfc(hsi_cube_hfc)
    
    print(f"Endmember estimates — PCA 80%: {n_80}, PCA Elbow: {n_elbow}, VD: {ns_hfc}")

    # pick ns_hfc which is closest to mean pca
    ns = np.array([n_80, n_elbow])
    mean_pca = ns.mean()
    idx = np.abs(ns_hfc - mean_pca).argmin()
    n_vd = ns_hfc[idx]

    # add n_vd to consideration if it is not too far from max
    if abs(n_vd - np.max(ns)) < 2:
        ns = np.append(ns, n_vd)
    else:
        ns = np.append(ns, 0)
        print("Rejected predictions from HSI Virtual Dimensionality Measure")
        print("Reason: Estimates are too far from PCA predictions")
    
    predicted_n = np.max(ns)
    confidence = _determine_confidence(ns)

    print(f"Final consideration pool: {ns}")
    print(f"Final prediction is {predicted_n} with {confidence} confidence")

    return predicted_n, confidence