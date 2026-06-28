import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from lime_si import lime, util,compute_p_value

np.random.seed(5)
def gen_data(n, p):
    X = np.random.normal(0, 1, size=(n, p))
    true_beta = np.zeros(p)
    true_beta[:3] = 2.0
    y = np.dot(X, true_beta)
    noise = np.random.normal(0, 1, size=n)
    y += noise
    return X, y, true_beta
def run():
    n = 200
    p = 15
    miu = 0
    num_samples = 500
    num_refs = 30
    lamda = 2.5
    X_train, y_train, true_beta = gen_data(n, p)

    train_stats = lime.compute_train_stats(X_train)
    
    X_test= miu + np.random.normal(0, 1, size=p)
    X_test = X_test + 2.0 * true_beta
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

      p_val_para = compute_p_value.compute_p_value_para(X_inv_bins, X_binary, train_stats, j_selected ,m_raw, c_raw ,lamda, a_full, b_vec, etaj, etajTy, num_samples, p, A, bh, tn_mu, tn_sigma, num_refs,X_long)
      print(f"\n{'='*20} FEATURE SELECTION {j_selected} {'='*20}")
      print(f"     True Beta: {true_beta[j_selected]:.4f}")
      print(f" Parametric    | P-value: {p_val_para:.4f} ")

if __name__ == "__main__":
  run()


  
