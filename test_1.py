from numpy.random import normal, uniform
import numpy as np
from scipy.stats import multivariate_normal
import matplotlib.pyplot as plt
import matplotlib.mlab as mlab
from scipy.stats import norm
import time
counter = 0

#
# gmm_pos_c = 0.5
# gmm_pos_sigma = 0.2
# gmm_pos_w = 0.9
#
# gmm_neg_c = 0.9
# gmm_neg_sigma = 0.1
# gmm_neg_w = -0.4
#
# x = np.linspace(0,1,1000)
# plt.subplot(2,1,1)
# plt.plot(x, gmm_pos_w * mlab.normpdf(x, gmm_pos_c, gmm_pos_sigma), linestyle="--")
# plt.plot(x, gmm_neg_w * mlab.normpdf(x, gmm_neg_c, gmm_neg_sigma), linestyle="--")
# plt.plot(x, gmm_pos_w * mlab.normpdf(x, gmm_pos_c, gmm_pos_sigma) + gmm_neg_w * mlab.normpdf(x, gmm_neg_c, gmm_neg_sigma))
# samples = []
# for i in range(1000):
#     #     generate from positive component
#     accepted = False
#     while not accepted:
#         rv = normal(gmm_pos_c, gmm_pos_sigma)
#         pdf_pos = gmm_pos_w * norm.pdf(rv, gmm_pos_c, gmm_pos_sigma)
#         pdf_neg = gmm_neg_w * norm.pdf(rv, gmm_neg_c, gmm_neg_sigma)
#         pdf_res = pdf_pos + pdf_neg
#         if uniform() < pdf_res:
#             accepted = True
#         counter += 1
#     samples.append(rv)
#
# plt.hist(samples, normed=1, bins=100)
#
#
# # uniform
# gmm_pos_c = 0.5
# gmm_pos_sigma = 0.2
# gmm_pos_w = 0.9
#
# gmm_neg_c = 0.5
# gmm_neg_sigma = 0.1
# gmm_neg_w = -0.4
#
# x = np.linspace(0,1,1000)
# plt.subplot(2,1,2)
# plt.plot(x, np.array([gmm_pos_w] * len(x)), linestyle="--")
# plt.plot(x, gmm_neg_w * mlab.normpdf(x, gmm_neg_c, gmm_neg_sigma), linestyle="--")
# plt.plot(x, np.array([gmm_pos_w] * len(x)) + gmm_neg_w * mlab.normpdf(x, gmm_neg_c, gmm_neg_sigma))
# samples = []
#
# for i in range(1000):
#     #     generate from positive component
#     accepted = False
#     while not accepted:
#         rv = uniform()
#         pdf_pos = gmm_pos_w
#         pdf_neg = gmm_neg_w * norm.pdf(rv, gmm_neg_c, gmm_neg_sigma)
#         pdf_res = pdf_pos + pdf_neg
#         if uniform() < pdf_res:
#             accepted = True
#         counter += 1
#     samples.append(rv)
#
# plt.hist(samples, normed=1, bins=100)
# print "number of rejected samples: ", counter
# plt.show()

def test_sampling():
    from numpy.random import multivariate_normal
    np.random.seed(196835)
    # generate k random gausiian components
    k = 5
    counter = 0
    weights = [1.0]
    weights.extend(uniform(-1.0, 1.0, k))
    print "weights: ", weights
    means = uniform(size=k)
    sigmas = uniform(0.01, 0.03, k)

    generator = [lambda: uniform()]
    generator.extend([lambda i=i: normal(means[i], sigmas[i]) for i in range(k)])

    pdfs = [lambda x: weights[0]]
    for i in range(k):
        pdfs.append(lambda x, i=i: weights[i+1]*multivariate_normal.pdf(x, means[i], sigmas[i]))
    # pdfs.extend([lambda x, i=i: norm.pdf(x, means[i], sigmas[i]) for i in range(k)])
    pos_idx = [j for j, e in enumerate(weights) if e > 0]
    cumsum = np.cumsum([weights[j] for j in pos_idx])
    samples = []
    t0 = time.time()
    for i in range(1000):
        #     generate from positive component
        accepted = False
        while not accepted:
            # choose a component to generate from
            rv = uniform(low=0.0, high=cumsum[-1], size=1)
            idx = np.searchsorted(cumsum, rv)
            idx = pos_idx[idx]
            # generate rnd number
            rv = generator[idx]()
            if rv > 1.0 or rv < 0.0:
                counter += 1
                continue
            # rv = weights[0]*uniform() if idx is 0 else normal(means[idx-1], sigmas[idx-1])
            # find probability for given rnd number
            p = sum([pdfs[j](rv) for j in range(len(pdfs))])
            if uniform() < p:
                accepted = True
            counter += 1
        samples.append(rv)
    print "time elapsed: ", time.time()-t0
    # plot
    x = np.linspace(0, 1, 1000)
    res = weights[0] * np.ones(1000)
    plt.plot(x, res, linestyle="--")
    for i in range(0, k):
        pdf = weights[i+1] * mlab.normpdf(x, means[i], sigmas[i])
        res += pdf
        plt.plot(x, pdf, linestyle="--")
    plt.plot(x, res, linewidth="2")
    plt.hist(samples, normed=1, bins=100, color="blue")
    print "number of rejected samples: ", counter
    plt.show()


def test_optimize():
    np.random.seed(196835)
    # generate k random gausiian components
    k = 10
    num_dimensions = 20
    weights = [1.0]
    weights.extend(uniform(0.1, 1.0, k))

    means = np.array([uniform(size=num_dimensions) for i in range(k)])
    sigmas = np.array([ np.identity(num_dimensions)*uniform(0.01, 0.03) for i in range(k) ])

    pdfs = [lambda x: weights[0]]
    for i in range(k):
        pdfs.append(lambda x, i=i: weights[i + 1] * multivariate_normal.pdf(x, means[i], sigmas[i]))

    func = lambda x: sum([pdf(x) for pdf in pdfs])
    bound = [(0.0, 1.0) for i in range(num_dimensions)]
    from scipy.optimize import differential_evolution

    x0 = 0.01

    t0 = time.time()
    optimize_result = differential_evolution(lambda x: -func(x), bounds=bound)
    t1 = time.time()
    print "succeed:", optimize_result.success
    print "global maxima:", optimize_result.x
    print "elapsed time:", t1 - t0


def generate_gm(num_pos, num_neg, d, num_clusters):
    from numpy.random import multivariate_normal
    means   = []
    sigmas  = []
    weights = []

    n = num_pos + num_neg

    np.random.seed(196093)
    sigmas.append(10.0)
    means.append([0.5]*d)
    weights.append(1.0)

    cluster_means = [uniform(0.2, 0.8, size=d) for i in range(num_clusters)]
    print cluster_means
    num_components_per_cluster = num_pos / num_clusters
    for i in range(num_clusters):
        means = np.append(means, [[normal(cluster_means[i][j], 0.08) for j in range(d)] for k in range(num_components_per_cluster)], axis=0)
        sigmas = np.append(sigmas, [uniform(0.01, 0.03) for j in range(num_components_per_cluster)])
        weights = np.append(weights, uniform(0.0, 1.0, size=num_components_per_cluster))
    print "first 2 positive components:"
    print "means:", means[:2]
    print "sigmas:", sigmas[:2]
    print "weights:", weights[:2]

    if num_neg > 1:
        negs_0 = uniform(0.0, 0.3, size=d*num_neg/2)
        negs_1 = uniform(0.7, 1.0, size=d*num_neg/2)
        negs = negs_0.extend(negs_1)
        np.random.shuffle(negs)
        negs = zip(*[iter(negs)] * d)
        means = np.append(means, negs, axis=0)
        sigmas = np.append(sigmas, [uniform(0.01, 0.03) for i in range(num_neg)])
        weights = np.append(weights, uniform(-1.0, 0.0, num_neg))

        print "first 2 negative components:"
        print "means:", means[num_pos:num_pos + 2]
        print "sigmas:", sigmas[num_pos:num_pos + 2]
        print "weights:", weights[num_pos:num_pos + 2]

    assert len(means) == len(sigmas)
    assert len(means) == len(weights)

    return weights, means, sigmas


def plot_gm(weights, means, sigmas):
    x = np.linspace(0, 1, 1000)
    means_1 = [means[i][0] for i in range(len(means))]
    res = np.zeros(len(x))
    for i in range(len(means)):
        pdf = weights[i] * mlab.normpdf(x, means_1[i], sigmas[i])
        res += pdf
        plt.plot(x, pdf, linestyle="--")
    plt.plot(x, res, linewidth="2")
    # plt.plot(x, np.array([gmm_pos_w] * len(x)) + gmm_neg_w * mlab.normpdf(x, gmm_neg_c, gmm_neg_sigma))


def plot_gm_3d(weights, means, sigmas):
    from scipy.stats import multivariate_normal
    from mpl_toolkits.mplot3d import Axes3D
    x = np.linspace(0, 1, 200)
    y = np.linspace(0, 1, 200)
    X, Y = np.meshgrid(x, y)
    pos = np.empty(X.shape + (2,))
    pos[:, :, 0] = X
    pos[:, :, 1] = Y

    k = len(weights)
    rvs = []
    values = np.zeros((len(x), len(y)))
    for i in range(k):
        rvs.append(multivariate_normal(means[i], sigmas[i]))
        values += weights[i] * rvs[-1].pdf(pos)

    fig = plt.figure()
    ax = fig.gca(projection="3d")
    ax.plot_surface(X, Y, values, cmap='viridis', linewidth=0)
    ax.set_xlabel('X axis')
    ax.set_ylabel('Y axis')
    ax.set_zlabel('Z axis')


def pdf(x, w, m, s):
    return sum([w[i] * multivariate_normal.pdf(x, m[i], s[i]) for i in range(len(w))])


# returns smoothed sigmas
def smooth_gm(s, y):
    sigma_smoothed = [np.sqrt(sigma**2 + y**2/2.) for sigma in s]
    return sigma_smoothed


def conditional_gm(w, m, s, given_idx, given_values):
    n = len(w)
    d = len(m[0])
    rest_idx = [range(d)[i] for i in range(d) if i not in given_idx]
    m_reduced = [[m[i][j] for j in rest_idx] for i in range(n)]
    w_new = w
    for i in range(n):
        mean = np.array([m[i][j] for j in given_idx])
        x = np.exp(-(np.linalg.norm(mean-np.array(given_values))**2 / (2. * s[i]**2)))
        w_new[i] *= x
    return w_new, m_reduced, s


def filter_out_gm(w, m, s, given_idx, given_values):
    n = len(w)
    idx_rest = []
    for i in range(n):
        mean = np.array([m[i][j] for j in given_idx])
        x = np.linalg.norm(mean - np.array(given_values)) ** 2
        if x < 3. * s[i]:
            idx_rest.append(i)
    return [w[i] for i in idx_rest], [m[i] for i in idx_rest], [s[i] for i in idx_rest]

#  generate gm
d = 2
num_clusters = 2
w, m, s = generate_gm(500, 1, d, num_clusters)
# plot gm 3d
if d == 2:
    plot_gm_3d(w, m, s)

means_1 = [[m[i][0]] for i in range(len(m))]

# conditional gm conditions
given_idx = [0]
given_values = [0.8]
# filter out components based on its low impact
f_w, f_m, f_s = filter_out_gm(w, m, s, given_idx, given_values)
# calc conditional gm
c_w, c_m, c_s = conditional_gm(f_w, f_m, f_s, given_idx, given_values)
# plot conditional gm
if d == 2:
    plt.figure()
    plot_gm(c_w, c_m, c_s)

# smooth conditional gm
y = 0.05
c_s_smoothed = smooth_gm(c_s, y)
# plot smoothed cond gm
if d == 2:
    plot_gm(c_w, c_m, c_s_smoothed)
# optimize smoothed cond gm
f = lambda x, w, m, s: -1.*pdf(x, w, m, s)
initial_guess = [sum([c_w[i]*c_m[i][j] for i in range(len(c_w))])/sum(c_w[i] for i in range(len(c_w))) for j in range(len(c_m[0]))]
# initial_guess = [0.4]*d
# print initial_guess
min_bound = [initial_guess[i] - 0.05 for i in range(len(c_m[0]))]
max_bound = [initial_guess[i] + 0.05 for i in range(len(c_m[0]))]
bounds = zip(min_bound, max_bound)
from scipy.optimize import minimize
t0 = time.time()
opt_res = minimize(f, initial_guess, (c_w, c_m, c_s_smoothed), method="L-BFGS-B", bounds=bounds)
print opt_res.x
print "elapsed time:", time.time() - t0
# plot_gm_3d(w, m, s)
# plot_gm_3d(w, m, s_smoothed)
#
# plt.figure()
# plot_gm(w, means_1, s)
# plot_gm(w, means_1, s_smoothed)
plt.show()

# x = np.linspace(0, 1, 100)
# smoothed = [smooth_gm(value, w, means_1, s, y) for value in x]
# plt.plot(x, smoothed)
# plt.show()

    # plot
    # x = np.linspace(0, 1, 1000)
    # res = weights[0] * np.ones(1000)
    # plt.plot(x, res, linestyle="--")
    # for i in range(0, k):
    #     pdf = weights[i + 1] * mlab.normpdf(x, means[i], sigmas[i])
    #     res += pdf
    #     plt.plot(x, pdf, linestyle="--")
    # plt.plot(x, res, linewidth="2")
    # plt.show()
    #

# test_optimize()