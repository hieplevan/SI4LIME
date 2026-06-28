import numpy as np
from scipy.stats import norm
from . import parametric_lasso

def compute_p_value_oc(X_long, X_binary, weight_seg, train_stats, j_selected ,m_raw, c_raw ,
                             lamda, a_full, b_full, etaj, etajTy, num_samples, p, A, bh, tn_miu, tn_sigma, num_refs):
  threshold = 20*tn_sigma
  list_zk, list_bhz, list_active_set = parametric_lasso.run_oc(
      X_binary, weight_seg,
             c_raw, m_raw,
             X_long, a_full, b_full,
             train_stats, j_selected,
             lamda, num_samples, p, threshold, num_refs
  )
  cdf_val_oc, _ = parametric_lasso.pivot(A, bh, list_active_set, list_zk, list_bhz, etaj, etajTy, tn_miu, tn_sigma,  'As')
  if cdf_val_oc is None:
    return None
  val_oc = 2 * min(cdf_val_oc, 1 - cdf_val_oc)
  return val_oc

def compute_p_value_naive(etajTy, tn_mu, tn_sigma, p):
    cdf = norm.cdf(etajTy, loc=tn_mu, scale=tn_sigma)
    val_naive = 2.0 * min(cdf, 1.0 - cdf)
    val_bonf = min(1.0, val_naive * 2**p)
    return val_naive, val_bonf

def compute_p_value_para( X_inverse_bin, X_binary, train_stats, j_selected ,m_raw, c_raw ,
                             lamda, a_full, b_full, etaj, etajTy, num_samples, p, A, bh, tn_miu, tn_sigma, num_refs,X_long):

    threshold = 20*tn_sigma

    val_0 = X_long[j_selected]
    list_zk, list_bhz, list_active_set = parametric_lasso.run_parametric(X_inverse_bin, X_binary ,c_raw, m_raw,
                      a_full, b_full, train_stats, j_selected,
                      lamda, num_samples, p, num_refs, threshold)
    cdf_val_para, _ = parametric_lasso.pivot(A, bh, list_active_set, list_zk, list_bhz, etaj, etajTy, tn_miu, tn_sigma,  'A')
    if cdf_val_para is None:
      return None
    val_para = 2 * min(cdf_val_para, 1 - cdf_val_para)
    return val_para
