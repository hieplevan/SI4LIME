import numpy as np
from sklearn.linear_model import LinearRegression

def compute_quotient(numerator, denominator):
    if denominator == 0:
        return np.inf

    quotient = numerator / denominator

    if quotient <= 0:
        return np.inf

    return quotient

def compute_kernel_weights(X_binary):
    p     = X_binary.shape[1]
    dist  = np.linalg.norm(X_binary - X_binary[0], axis=1)
    width = 0.75 * np.sqrt(p)
    return np.sqrt(np.exp(-(dist ** 2) / (width ** 2)))

def recompute_X_binary(x_test_new, X_inverse_bins, train_stats, j_selected, X_binary_original):
    X_binary = X_binary_original.copy()

    x_bin = np.searchsorted(train_stats['thresholds'][j_selected], x_test_new[j_selected])
    X_binary[:, j_selected] = (X_inverse_bins[:, j_selected] == x_bin).astype(int)

    X_binary[0, :] = 1
    return X_binary

def fit_model(X_train, y_train):
    model = LinearRegression(fit_intercept=True)
    model.fit(X_train, y_train)
    return model, model.coef_, model.intercept_


def construct_test_statistic(X_long, j, p, num_refs):
    etaj = np.zeros((num_refs + 1) * p)
    etaj[j] = 1.0
    for i in range(num_refs):
        etaj[p + i * p + j] = -1.0 / num_refs
    T = np.dot(etaj, X_long)
    return etaj, T

def construct_m_c_from_model(X_long, etaj, beta, bias, X_inverse, p, num_refs, num_samples):
    Cov = np.identity((num_refs + 1) * p)
    denom = np.dot(etaj, np.dot(Cov, etaj))
    if denom == 0:
        return None, None, None, None

    b_full = np.dot(Cov, etaj) / denom
    z = np.dot(etaj, X_long)
    a_full = X_long - b_full * z


    a_test = a_full[:p].reshape(1, p)
    b_test = b_full[:p].reshape(1, p)

    A = np.vstack([a_test, X_inverse[1:]])
    B = np.vstack([b_test, np.zeros((num_samples - 1, p))])

    c = np.dot(A, beta) + bias
    m = np.dot(B, beta)

    return c, m, a_full, b_full

def construct_A_XA_Ac_XAc_bhA(X, bh, p):
    A = []
    Ac = []
    bhA = []
    for j in range(p):
        bhj = bh[j]
        if bhj != 0:
            A.append(j)
            bhA.append(bhj)
        else:
            Ac.append(j)
    XA = X[:, A] if len(A) > 0 else None
    XAc = X[:, Ac] if len(Ac) > 0 else None
    bhA = np.array(bhA).reshape((len(A), 1)) if len(A) > 0 else np.array([])
    return A, XA, Ac, XAc, bhA

def compute_bin_boundaries(a_full, b_vec, train_stats, j, p, num_refs, threshold) :
    thresh_j = train_stats['thresholds'][j]
    crossings = []

    akj = a_full[j]
    bkj = b_vec[j]

    if abs(bkj) > 0:
        for bnd in thresh_j:
            z = (bnd - akj) / bkj
            if -threshold <= z <= threshold:
                crossings.append(z)

    return sorted(set(crossings))