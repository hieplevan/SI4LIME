import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from scipy.stats import kstest
import statsmodels.api as sm
from lime_si import lime, gen_data,util, compute_p_value

np.random.seed(5)

def run(X_train, y_train,
            X_test, X_long,
            miu, num_samples, num_refs,
            p, train_stats):

    model, beta, bias = util.fit_model(X_train, y_train)
    lamda = 2.5

    X_binary, X_inverse, X_inv_bins = lime.lime_sample(X_test, train_stats, num_samples)

    bh, X_w, y_w, weights = lime.lime_explain(X_binary, X_inverse, model, lamda)

    A_set, _, _, _, _ = util.construct_A_XA_Ac_XAc_bhA(X_w, bh, p)
    if len(A_set) == 0:
        return None
    j_selected = np.random.choice(A_set)

    etaj, etajTy = util.construct_test_statistic(X_long, j_selected, p, num_refs)

    c_raw, m_raw, a_full, b_vec = util.construct_m_c_from_model(
        X_long, etaj, beta, bias, X_inverse, p, num_refs, num_samples
    )
    if c_raw is None:
        return None

    cov = np.identity((num_refs + 1) * p)
    tn_sigma = np.sqrt(np.dot(np.dot(etaj.T, cov), etaj))
    miuT     = np.full((num_refs + 1) * p, miu)
    tn_mu   = np.dot(etaj, miuT)

 
    val_oc =  compute_p_value.compute_p_value_oc(X_long, X_binary, weights, train_stats, j_selected ,m_raw, c_raw ,lamda, a_full, b_vec, etaj, etajTy, num_samples, p, A_set, bh, tn_mu, tn_sigma, num_refs)

    val_naive,val_bonf = compute_p_value.compute_p_value_naive(etajTy, tn_mu, tn_sigma,p)


    val_para = compute_p_value.compute_p_value_para(X_inv_bins, X_binary, train_stats, j_selected ,m_raw, c_raw ,lamda, a_full, b_vec, etaj, etajTy, num_samples, p, A_set, bh, tn_mu, tn_sigma, num_refs,X_long)
    return val_oc, val_naive, val_para,val_bonf


if __name__ == "__main__":

    n = 200
    p = 15
    miu = 0
    num_samples = 500
    alpha_level= 0.05
    num_refs = 30
    X_train, y_train  = gen_data.generate_data(n, p)
    train_stats = lime.compute_train_stats(X_train)

    list_p_value_oc = []
    list_p_value_naive = []
    list_p_value_para = []

    for i in range(1000):
        X_test_new = miu + np.random.normal(0, 1, size=p)
        X_refs = miu + np.random.normal(0, 1, size=(num_refs, p))

        X_long_new = np.concatenate((X_test_new, X_refs.flatten()))
        X_ref = X_refs.mean(axis=0)

        result = run(X_train, y_train,X_test_new, X_long_new, miu, num_samples, num_refs, p, train_stats)

        if result is not None:
            val_oc, val_naive, val_para,val_bonf = result
            if val_oc is not None: list_p_value_oc.append(val_oc)
            if val_naive is not None: list_p_value_naive.append(val_naive)
            if val_para is not None: list_p_value_para.append(val_para)

    print(len(list_p_value_para))
    stat, p_ks = kstest(list_p_value_oc, 'uniform')
    stat, p_ks_naive = kstest(list_p_value_naive, 'uniform')
    stat, p_ks_para = kstest(list_p_value_para, 'uniform')
    print(f"KS Test P-value (OC): {p_ks:.4f}")
    print(f"KS Test P-value (Naive): {p_ks_naive:.4f}")
    print(f"KS Test P-value (Parametric): {p_ks_para:.4f}")

    plt.rcParams.update({'font.size': 18})
    grid = np.linspace(0, 1, 101)
    plt.plot(grid, sm.distributions.ECDF(np.array(list_p_value_oc))(grid), 'r-', linewidth=6, label='p-value')
    plt.plot(grid, sm.distributions.ECDF(np.array(list_p_value_naive))(grid), 'b-', linewidth=6, label='p-value naive')
    plt.plot(grid, sm.distributions.ECDF(np.array(list_p_value_para))(grid), 'g-', linewidth=6, label='p-value para')
    plt.plot([0, 1], [0, 1], 'k--')
    plt.legend()
    plt.tight_layout()
    plt.show()

    plt.hist(list_p_value_para)
    plt.show()
    plt.figure(figsize=(8, 6))
