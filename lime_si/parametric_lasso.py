import numpy as np
from sklearn import linear_model
from . import util
from mpmath import mp



def parametric_lasso_step(X, yz, b, lamda, n, p):
    yz_flat = yz.flatten()
    b_flat  = b.flatten()

    clf = linear_model.Lasso(alpha=lamda/n, fit_intercept=False, tol=1e-10, random_state=5)
    clf.fit(X, yz_flat)
    bhz = clf.coef_

    Az, XAz, Acz, XAcz, bhAz = util.construct_A_XA_Ac_XAc_bhA(X, bhz, p)

    etaAz   = np.array([])
    shAz    = np.array([])
    gammaAz = np.array([])

    if XAz is not None and len(Az) > 0:
        inv      = np.linalg.pinv(np.dot(XAz.T, XAz))
        invXAzT  = np.dot(inv, XAz.T)
        etaAz    = np.dot(invXAzT, b_flat)

    if XAcz is not None and len(Acz) > 0:
        if XAz is None or len(Az) == 0:
            e1 = yz_flat
        else:
            e1 = yz_flat - np.dot(XAz, bhAz.flatten())

        e2    = np.dot(XAcz.T, e1)
        shAz  = e2 / lamda 

        if XAz is None or len(Az) == 0:
            gammaAz = np.dot(XAcz.T, b_flat) 
        else:
            gammaAz = (np.dot(XAcz.T, b_flat) - np.dot(np.dot(XAcz.T, XAz), etaAz)) 

    bhAz    = bhAz.flatten()
    etaAz   = etaAz.flatten()
    shAz    = shAz.flatten()
    gammaAz = gammaAz.flatten()

    min1 = np.inf
    min2 = np.inf

    for j in range(len(etaAz)):
        q = util.compute_quotient(-bhAz[j], etaAz[j])
        if q < min1:
            min1 = q

    for j in range(len(gammaAz)):
        numerator = (np.sign(gammaAz[j]) - shAz[j]) * lamda
        q = util.compute_quotient(numerator, gammaAz[j])
        if q < min2:
            min2 = q

    res_step = min(min1, min2)
    return res_step, Az, bhz

def run_parametric( X_inverse_bins,X_binary,
                       c_raw, m_raw,
                      a_full, b_vec,
                      train_stats, j_selected,
                      lamda, num_samples, p,
                      num_refs, threshold):
    boundaries     = util.compute_bin_boundaries(a_full, b_vec, train_stats,
                                            j_selected, p, num_refs, threshold)
    seg_starts     = [-threshold] + boundaries
    seg_ends       = boundaries   + [threshold]

    list_zk_all, list_bhz_all, list_as_all = [-threshold], [], []

    for z_s, z_e in zip(seg_starts, seg_ends):
        if z_s >= z_e:
            continue

        z_mid       = (z_s + z_e) / 2.0
        x_test_mid  = a_full[:p] + b_vec[:p] * z_mid
        Z_seg       = util.recompute_X_binary(x_test_mid, X_inverse_bins, train_stats,j_selected, X_binary_original=X_binary)

        sqrt_W      = util.compute_kernel_weights(Z_seg)
        X_design    = Z_seg   * sqrt_W[:, np.newaxis]
        c_vec       = c_raw   * sqrt_W
        m_vec       = m_raw   * sqrt_W

        zk_seg, bhz_seg, as_seg  = [z_s], [], []
        zk = z_s
        while zk < z_e:
            yz              = m_vec * zk + c_vec
            step, Akz, bhkz = parametric_lasso_step(X_design, yz, m_vec, lamda, num_samples, p)
            step            = max(step, 0)
            zk_next         = min(zk + step + 1e-4, z_e)

            zk_seg.append(zk_next)
            bhz_seg.append(bhkz)
            as_seg.append(Akz)

            if zk_next >= z_e:
                break
            zk = zk_next

        list_zk_all += zk_seg[1:]
        list_bhz_all += bhz_seg
        list_as_all  += as_seg

    return list_zk_all, list_bhz_all, list_as_all

def run_oc(X_binary, weight_seg, c_raw, m_raw, X_long, a_full, b_full,train_stats, j_selected, lamda, num_samples, p, threshold,num_refs):

    thresh_j = train_stats['thresholds'][j_selected]
    extended_thresh = np.sort(np.concatenate([[-np.inf], thresh_j, [np.inf]]))

    val_0 = X_long[j_selected]
    akj   = a_full[j_selected]
    bkj   = b_full[j_selected]

    idx  = np.searchsorted(extended_thresh, val_0, side='right')
    lb_0 = extended_thresh[idx - 1]
    ub_0 = extended_thresh[idx]

    z_min_bin = -threshold
    z_max_bin = threshold

    if abs(bkj) > 0:
        if bkj > 0:
            z_min_bin = (lb_0 - akj) / bkj if not np.isinf(lb_0) else -threshold
            z_max_bin = (ub_0 - akj) / bkj if not np.isinf(ub_0) else  threshold
        else:
            z_min_bin = (ub_0 - akj) / bkj if not np.isinf(ub_0) else -threshold
            z_max_bin = (lb_0 - akj) / bkj if not np.isinf(lb_0) else  threshold

    z_min = max(z_min_bin, -threshold)
    z_max = min(z_max_bin,  threshold)

    if z_min > z_max:
        z_min, z_max = -threshold, threshold
    list_zk, list_bhz, list_active_set = [z_min], [], []

    sqrt_W = np.sqrt(weight_seg)
    m_vec = m_raw * sqrt_W
    c_vec = c_raw * sqrt_W
    X_design_seg = X_binary * sqrt_W[:, np.newaxis]
    zk = z_min
    while zk < z_max:
        yz   = m_vec * zk + c_vec
        step, Akz, bhkz = parametric_lasso_step(X_design_seg, yz, m_vec, lamda, num_samples, p)
        step    = max(step, 0)
        zk_next = min(zk + step + 1e-4, z_max)

        list_zk.append(zk_next)
        list_bhz.append(bhkz)
        list_active_set.append(Akz)
        if zk_next >= z_max: break
        zk = zk_next

    return list_zk, list_bhz, list_active_set

def pivot(A, bh,
          list_active_set, list_zk, list_bhz,
          etaj, etajTy,
          tn_mu, tn_sigma,
          type):

    z_interval = []
    for i in range(len(list_active_set)):
        if type == 'As':
            if np.array_equal(np.sign(bh), np.sign(list_bhz[i])):
                z_interval.append([list_zk[i], list_zk[i + 1] - 1e-10])
        if type == 'A':
            if np.array_equal(A, list_active_set[i]):
                z_interval.append([list_zk[i], list_zk[i + 1] - 1e-10])

    new_z_interval = []
    for each_interval in z_interval:
        if len(new_z_interval) == 0:
            new_z_interval.append(each_interval)
        else:
            sub = each_interval[0] - new_z_interval[-1][1]
            if abs(sub) < 0.01:
                new_z_interval[-1][1] = each_interval[1]
            else:
                new_z_interval.append(each_interval)

    z_interval  = new_z_interval
    numerator   = 0
    denominator = 0
    for each_interval in z_interval:
        al = each_interval[0]
        ar = each_interval[1]

        denominator += mp.ncdf((ar - tn_mu) / tn_sigma) - mp.ncdf((al - tn_mu) / tn_sigma)

        if etajTy >= ar:
            numerator += mp.ncdf((ar - tn_mu) / tn_sigma) - mp.ncdf((al - tn_mu) / tn_sigma)
        elif (etajTy >= al) and (etajTy < ar):
            numerator += mp.ncdf((etajTy - tn_mu) / tn_sigma) - mp.ncdf((al - tn_mu) / tn_sigma)

    if denominator != 0:
        return float(numerator / denominator), z_interval
    else:
        return None , z_interval
