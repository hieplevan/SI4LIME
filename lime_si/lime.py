import numpy as np
from sklearn.linear_model import Lasso
import scipy.stats

def compute_train_stats(X_train, categorical_features=[]):
    n_samples, n_features = X_train.shape
    train_stats = {
        'thresholds': [],
        'bin_frequencies': [],
        'bin_means': [],
        'bin_stds': [],
        'bin_mins': [],
        'bin_maxs': [],
        'is_categorical': [i in categorical_features for i in range(n_features)]
    }

    for i in range(n_features):
        col_data = X_train[:, i]

        if i in categorical_features:
            vals, counts = np.unique(col_data, return_counts=True)
            train_stats['thresholds'].append(None)
            train_stats['bin_frequencies'].append(counts / n_samples)
            train_stats['bin_means'].append(vals)
            train_stats['bin_stds'].append(None)
            train_stats['bin_mins'].append(None)
            train_stats['bin_maxs'].append(None)
        else:
            thresh = np.unique(np.percentile(col_data, [25, 50, 75]))
            train_stats['thresholds'].append(thresh)
            bin_ids = np.searchsorted(thresh, col_data)
            n_bins = len(thresh) + 1
            freqs, means, stds, mins, maxs = [], [], [], [], []
            for b in range(n_bins):
                mask = (bin_ids == b)
                bin_data = col_data[mask]

                freqs.append(len(bin_data) / n_samples)
                if len(bin_data) > 0:
                    means.append(np.mean(bin_data))
                    stds.append(np.std(bin_data))
                    mins.append(np.min(bin_data))
                    maxs.append(np.max(bin_data))
                else:
                    means.append(0)
                    stds.append(1e-10)
                    mins.append(thresh[b-1] if b > 0 else -np.inf)
                    maxs.append(thresh[b] if b < len(thresh) else np.inf)

            train_stats['bin_frequencies'].append(np.array(freqs))
            train_stats['bin_means'].append(np.array(means))
            train_stats['bin_stds'].append(np.array(stds))
            train_stats['bin_mins'].append(np.array(mins))
            train_stats['bin_maxs'].append(np.array(maxs))

    return train_stats

def undiscretize(sampled_bin_ids, train_stats):
    result = sampled_bin_ids.astype(float)
    n_samples, n_features = sampled_bin_ids.shape

    for i in range(n_features):
        if train_stats['is_categorical'][i]:
            continue

        col_bin_ids = sampled_bin_ids[:, i].astype(int)

        for b in range(len(train_stats['bin_frequencies'][i])):
            mask = (col_bin_ids == b)
            if not mask.any(): continue

            mu = train_stats['bin_means'][i][b]
            sigma = train_stats['bin_stds'][i][b]
            low = train_stats['bin_mins'][i][b]
            high = train_stats['bin_maxs'][i][b]

            a = (low - mu) / sigma
            b_param = (high - mu) / sigma

            result[mask, i] = scipy.stats.truncnorm.rvs(
                a, b_param, loc=mu, scale=sigma, size=mask.sum()
            )
    return result


def lime_sample(x_test, train_stats, num_samples=5000):
    n_features = len(x_test)
    X_binary = np.zeros((num_samples, n_features))
    X_inverse_bins = np.zeros((num_samples, n_features))

    x_test_bin_ids = []
    for i in range(n_features):
        if train_stats['is_categorical'][i]:
            x_test_bin_ids.append(x_test[i])
        else:
            x_test_bin_ids.append(np.searchsorted(train_stats['thresholds'][i], x_test[i]))

    for i in range(n_features):
        n_bins = len(train_stats['bin_frequencies'][i])
        bin_indices = np.arange(n_bins)

        sampled_indices = np.random.choice(
            bin_indices, size=num_samples, replace=True, p=train_stats['bin_frequencies'][i]
        )

        if train_stats['is_categorical'][i]:
            X_inverse_bins[:, i] = train_stats['bin_means'][i][sampled_indices]
        else:
            X_inverse_bins[:, i] = sampled_indices

        X_binary[:, i] = (X_inverse_bins[:, i] == x_test_bin_ids[i]).astype(int)

    X_inverse = X_inverse_bins.copy()
    X_inverse[1:] = undiscretize(X_inverse_bins[1:], train_stats)
    X_binary[0, :] = 1
    X_inverse[0, :] = x_test

    return X_binary, X_inverse, X_inverse_bins



def lime_explain(X_binary, X_inverse, model, alpha):
    y_probs = model.predict(X_inverse)

    n, p = X_binary.shape
    dist = np.linalg.norm(X_binary - X_binary[0], axis=1)

    width = 0.75 * np.sqrt(p)
    weights = np.exp(-(dist ** 2) / (width ** 2))

    sqrt_W = np.sqrt(weights)

    X_weighted = X_binary * sqrt_W[:, np.newaxis]
    y_weighted = y_probs * sqrt_W

    lasso = Lasso(alpha=alpha/n, fit_intercept=False, tol=1e-10, random_state=5)
    lasso.fit(X_weighted, y_weighted)

    active_set = lasso.coef_

    return active_set, X_weighted, y_weighted, weights