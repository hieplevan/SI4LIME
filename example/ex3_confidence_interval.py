import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
from lime_si import lime,gen_data,util
from lime_si import parametric_lasso
np.random.seed(5)

def run():
    n = 200
    p = 20
    miu = 0
    num_samples = 500
    num_refs = 30
    lamda = 2.5
    X_train, y_train = gen_data.generate_data(n, p)

    train_stats = lime.compute_train_stats(X_train)
    
    X_test= miu + np.random.normal(0, 1, size=p)

    X_refs = miu + np.random.normal(0, 1, size=(num_refs, p))
    X_long = np.concatenate((X_test, X_refs.flatten()))


    model, beta, bias = util.fit_model(X_train, y_train)

    X_binary, X_inverse, X_inv_bins = lime.lime_sample(X_test, train_stats, num_samples)

    bh, X_w, y_w, weights = lime.lime_explain(X_binary, X_inverse, model, lamda)

    A, XA, Ac, XAc, bhA = util.construct_A_XA_Ac_XAc_bhA(X_w, bh, p)
    if len(A) == 0 :
      return None
    
    null_features = [j for j in A ]
    if len(null_features) == 0:
      return None
    
    for j_selected in null_features:
        etaj, etajTy = util.construct_test_statistic(X_long, j_selected , p, num_refs)

        c_raw, m_raw, a_full, b_vec = util.construct_m_c_from_model(
            X_long, etaj, beta, bias, X_inverse, p, num_refs, num_samples
        )
        if c_raw is None or m_raw is None:
          continue

        cov = np.identity((num_refs + 1) * p)
        tn_sigma = np.sqrt(np.dot(np.dot(etaj.T, cov), etaj))
        miuT     = np.full((num_refs + 1) * p, miu)
        tn_mu   = np.dot(etaj, miuT)
        threshold = 20*tn_sigma

        list_zk_para, list_bhz_para, list_active_set_para = parametric_lasso.run_parametric( X_inv_bins,X_binary,
                       c_raw, m_raw,
                      a_full, b_vec,
                      train_stats, j_selected,
                      lamda, num_samples, p,
                      num_refs, threshold)

        cdf_val_para, z_para_interval = parametric_lasso.pivot(A, bh,
          list_active_set_para, list_zk_para, list_bhz_para,
          etaj, etajTy,
          tn_mu, tn_sigma,
          'A')
        
        if cdf_val_para is None:
          continue
        val_para = 2 * min(cdf_val_para, 1 - cdf_val_para)
        list_zk_oc, list_bhz_oc, list_active_set_oc = parametric_lasso.run_oc(
          X_binary, weights,
          c_raw, m_raw,
          X_long, a_full, b_vec,
          train_stats, j_selected,
          lamda, num_samples, p, threshold, num_refs
        )
        cdf_val_oc, z_oc_interval = parametric_lasso.pivot(A, bh,
          list_active_set_oc, list_zk_oc, list_bhz_oc,
          etaj, etajTy,
          tn_mu, tn_sigma,
          'As')
        if cdf_val_oc is None:
          continue
        val_oc = 2 * min(cdf_val_oc, 1 - cdf_val_oc)
        print(f"\n{'='*20} FEATURE SELECTION {j_selected} {'='*20}")
        print(f" Parametric    | P-value: {val_para:.4f} | Interval: {z_para_interval}")
        print(f" Oc       | P-value: {val_oc:.4f} | Interval: {z_oc_interval}")

if __name__ == "__main__":
    run()

 