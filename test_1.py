
from __future__ import division
from numpy.random import normal, uniform
import numpy as np
from scipy.stats import multivariate_normal
import matplotlib.pyplot as plt
import matplotlib.mlab as mlab
import matplotlib
from scipy.stats import norm
import time
import utils
import sys

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
    from scipy.stats import multivariate_normal
    np.random.seed(196835)
    # generate k random gausiian components
    k = 6
    counter = 0
    weights = [1.0]
    weights.extend(uniform(-1., 1., k))
    print "weights: ", weights
    means = uniform(size=k)
    sigmas = uniform(0.03, 0.07, k)
    #
    # means=[0.5, 0.5]
    # sigmas=[0.1, 0.05]
    # weights=[1.0, 1.0, -0.5]

    print "means:", means
    #  normalize weights
    pos_idx = [j for j, e in enumerate(weights) if e > 0]
    neg_idx = [j for j, e in enumerate(weights) if e < 0]
    cumsum = np.cumsum([weights[j] for j in pos_idx])
    pos_sum = cumsum[-1]
    for i in range(len(weights)):
        weights[i] /= pos_sum
    print "weights normed:", weights

    generator = [lambda: uniform()]
    generator.extend([lambda i=i: normal(means[i], sigmas[i]) for i in range(k)])

    pdfs = [lambda x: weights[0]]
    cdfs = [lambda x: weights[0]*x]
    for i in range(k):
        pdfs.append(lambda x, i=i: weights[i+1]*multivariate_normal.pdf(x, mean=means[i], cov=sigmas[i]**2))
        cdfs.append(lambda x, i=i: weights[i+1]*norm.cdf(x, loc=means[i], scale=sigmas[i]))
    # pdfs.extend([lambda x, i=i: norm.pdf(x, means[i], sigmas[i]) for i in range(k)])
    pos_idx = [j for j, e in enumerate(weights) if e > 0]
    cumsum = np.cumsum([weights[j] for j in pos_idx])
    samples = []

    t0 = time.time()
    for i in range(2000):
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
            # calc pdf of positive components
            p_pos = sum([pdfs[j](rv) for j in pos_idx])
            p_neg = sum([pdfs[j](rv) for j in neg_idx])
            p = sum([pdfs[j](rv) for j in range(len(pdfs))])
            if uniform(0.0, p_pos) > -1.*p_neg:
                accepted = True
            counter += 1
        samples.append(rv)
    print "time elapsed: ", time.time()-t0
    # plot
    plt.rc('font', family='serif', size=14)
    plt.rc('xtick')
    plt.rc('ytick')

    fig = plt.figure(figsize=(8, 3))
    ax = fig.add_subplot(1, 1, 1)

    ax.set_xlabel('x')
    # ax.set_ylabel('f(x)')

    x = np.linspace(0, 1, 1000)
    res = weights[0] * np.ones(1000)
    plt.rcParams['lines.color']='gray'
    plt.rcParams['lines.linewidth']=1
    plt.plot(x, res, linestyle="--", color='k', label=r"$w_i \mathcal{N}(x | \mu_i,\Sigma_i)$")
    for i in range(0, k):
        pdf = weights[i+1] * multivariate_normal.pdf(x, means[i], sigmas[i]**2)
        res += pdf
        plt.plot(x, pdf, linestyle="--", color='k')
    # for i in range(len(res)):
        # if res[i] > 1.:
        #     res[i] = 1.0
        # if res[i] < 0.:
        #     res[i] = 0.
    plt.plot(x, res, linewidth="2", color='k', label=r"$h(x)$")
    plt.hist(samples, normed=1, bins=100, color='k', alpha=0.5)
    plt.legend()
    print "number of rejected samples: ", counter
    #  Kolmogorov smirnov test
    import scipy.stats
    cdf = lambda v: [sum(cdf(x) for cdf in cdfs) if 0 < x and x < 1 else 0 for x in v]
    ks_test_res = scipy.stats.kstest(samples, cdf)
    print ks_test_res
    # plt.figure()
    # plt.hist(samples, bins=100, color='k', normed=1, cumulative=True)
    # plt.plot(x, cdf(x), color='k')
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


def fprime(x, w, m, s):
    fprime = []
    for i in range(len(x)):
        fprime_i = sum(
            [(x[i] - m[j][i]) / (s[j]) * w[j] * multivariate_normal.pdf(x, m[j], s[j]) for j in range(len(w))])
        fprime.append(fprime_i)
    return np.array(fprime)


def f(x, weight, mean, sigma):
    return -1. * pdf(x, weight, mean, sigma)


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


def test_gm_optimize():

    #  generate gm
    d = 30
    num_clusters = 5
    w, m, s = generate_gm(100, 1, d, num_clusters)
    # plot gm 3d
    if d == 2:
        plot_gm_3d(w, m, s)

    means_1 = [[m[i][0]] for i in range(len(m))]

    # conditional gm conditions
    given_idx = [0]
    given_values = [0.8]
    # filter out components based on its low impact
    f_w, f_m, f_s = filter_out_gm(w, m, s, given_idx, given_values)
    # print how many components left
    print "{0} components left out of {1}".format(len(f_w), len(w))

    # calc conditional gm
    c_w, c_m, c_s = conditional_gm(f_w, f_m, f_s, given_idx, given_values)
    d_reduced = d - len(given_idx)
    # plot conditional gm
    if d == 2:
        plt.figure()
        plot_gm(c_w, c_m, c_s)

    # smooth conditional gm
    y = 0.08
    c_s_smoothed = smooth_gm(c_s, y)
    # plot smoothed cond gm
    if d == 2:
        plot_gm(c_w, c_m, c_s_smoothed)
    # optimize smoothed cond gm

    print "OPTIMIZATION"




    def hessian(x, w, m, s):
        return

    # f = lambda x, weight, mean, sigma: -1.*pdf(x, weight, mean, sigma)
    initial_guess = [sum([c_w[i]*c_m[i][j] for i in range(len(c_w))])/sum(c_w[i] for i in range(len(c_w))) for j in range(d_reduced)]
    print "initial_guess:", initial_guess

    from scipy.optimize import check_grad
    print "check_grad:", check_grad(f, fprime, initial_guess, c_w, c_m, c_s_smoothed)

    # from scipy.optimize import approx_fprime
    # eps = np.sqrt(np.finfo(float).eps)
    # print "approx_fprime:", approx_fprime(initial_guess, f, eps, c_w, c_m, c_s_smoothed)
    # print "calc_fprime:", fprime(initial_guess, c_w, c_m, c_s_smoothed)

    # initial_guess = [0.4]*d
    # print initial_guess
    min_bound = [initial_guess[i] - 0.2 for i in range(d_reduced)]
    max_bound = [initial_guess[i] + 0.2 for i in range(d_reduced)]
    bounds = zip(min_bound, max_bound)
    from scipy.optimize import minimize
    from scipy.optimize import fmin_l_bfgs_b
    t0 = time.time()
    methods=["L-BFGS-B", "Powell", "Nelder-Mead", "TNC"]
    opt_res = minimize(f, x0=initial_guess, jac=fprime, method=methods[3], tol=1e-1, args=(c_w, c_m, c_s_smoothed))
    print "Success:", opt_res.success
    print "Iterations:", opt_res.nit
    print "func and fprime evals:", opt_res.nfev
    print "Result:", opt_res.x
    print "initial guess:", initial_guess
    print "elapsed time:", time.time() - t0

    # test func comp cost
    t0 = time.time()
    for i in range(opt_res.nfev):
        f(initial_guess, c_w, c_m, c_s_smoothed)
    print "time to run f {0} times: {1}".format(opt_res.nfev, time.time()-t0)
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
# test_sampling()


def plot(func, x):
    plt.rc('font', family='serif', size=14)
    plt.rc('xtick')
    plt.rc('ytick')

    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)

    ax.set_xlabel('d')
    ax.set_ylabel('W(d)')
    y = func(x)
    ax.plot(x, y, color='k', ls='solid')
    plt.show()


def calc_response(w_1, mu_1, cov_1, w_2, mu_2, cov_2):
    m = len(mu_1) - len(mu_2)
    n = len(mu_2)
    mu_3 = mu_1[m:]
    cov_3 = cov_1[m:]

    A = cov_1[:m]
    a = mu_1[:m]

    B = cov_2
    b = mu_2

    AB = np.add(A, B)
    invAB = np.array([1/v for v in AB])
    detAB = np.prod(AB)

    absub = np.subtract(a, b)
    c = sum(absub[i]**2 * invAB[i] for i in range(m))
    z = 1. / np.sqrt(detAB * (2*np.pi)**m) * np.exp(-0.5 * c)
    w_3 = w_1 * w_2 * z
    return w_3, mu_3, cov_3


def calc_response_gm(w_1arr, mu_1arr, cov_1arr, w_2arr, mu_2arr, cov_2arr):
    w_3arr = []
    mu_3arr = []
    cov_3arr = []
    for w_1, mu_1, cov_1 in zip(w_1arr, mu_1arr, cov_1arr):
        for w_2, mu_2, cov_2 in zip( w_2arr, mu_2arr, cov_2arr):
            w_3, mu_3, cov_3 = calc_response(w_1, mu_1, cov_1, w_2, mu_2, cov_2)
            w_3arr.append(w_3)
            mu_3arr.append(mu_3)
            cov_3arr.append(cov_3)
    return w_3arr, mu_3arr, cov_3arr


def test_transfer_function():
    from scipy.stats import multivariate_normal
    from mpl_toolkits.mplot3d import Axes3D
    #  g1
    x = np.linspace(0, 1, 100)
    y = np.linspace(0, 1, 100)
    X, Y = np.meshgrid(x, y)
    pos = np.empty(X.shape + (2,))
    pos[:, :, 0] = X
    pos[:, :, 1] = Y

    weights_1 = [0.1, 0.05]
    means_1 = [[0.7, 0.35], [0.1, 0.75]]
    sigmas_1 = [[0.01, 0.01], [0.01, 0.01]]
    k = len(weights_1)
    rvs = []
    values_g1 = np.zeros((len(x), len(y)))
    for i in range(k):
        rvs.append(multivariate_normal(means_1[i], sigmas_1[i]))
        values_g1 += weights_1[i] * rvs[-1].pdf(pos)
    fig = plt.figure()
    ax1 = plt.subplot(141, projection="3d")
    ax1.plot_surface(X, Y, values_g1, cmap='viridis', linewidth=0)
    ax1.set_xlabel('X axis')
    ax1.set_ylabel('Y axis')
    ax1.set_zlabel('Z axis')
    # g2
    X, Y = np.meshgrid(x, y)
    pos = np.empty(X.shape + (2,))
    pos[:, :, 0] = X
    pos[:, :, 1] = Y

    weights_2 = [0.05, 0.1]
    means_2 = [[0.25], [0.7]]

    sigmas_2 = [[0.005], [0.006]]
    k = len(weights_2)
    rvs = []
    values_g2 = np.zeros((len(x), len(y)))
    for k in range(len(weights_2)):
        for i in range(len(x)):
            for j in range(len(y)):

                rvs = (multivariate_normal(means_2[k], sigmas_2[k]))
                values_g2[i][j] += weights_2[k] * rvs.pdf(pos[i][j][0])
    ax2 = plt.subplot(142, projection="3d", sharez=ax1)
    ax2.plot_surface(X, Y, values_g2, cmap='viridis', linewidth=0)
    ax2.set_xlabel('X axis')
    ax2.set_ylabel('Y axis')
    ax2.set_zlabel('Z axis')
    #  g3 = g1*g2
    X, Y = np.meshgrid(x, y)
    pos = np.empty(X.shape + (2,))
    pos[:, :, 0] = X
    pos[:, :, 1] = Y
    values_g3 = np.zeros((len(x), len(y)))
    for i in range(len(x)):
        for j in range(len(y)):
            values_g3[i][j] += values_g1[i][j] * values_g2[i][j]
    ax3 = plt.subplot(143, projection="3d", sharez=ax1)
    ax3.plot_surface(X, Y, values_g3, cmap='viridis', linewidth=0)
    ax3.set_xlabel('X axis')
    ax3.set_ylabel('Y axis')
    ax3.set_zlabel('Z axis')
    #  g4 = integrate g3 over x
    values_g4 = np.zeros(len(y))
    for j in range(len(y)):
        values_g4[j] += (sum(values_g3[j][i]* (x[1]-x[0]) for i in range(len(x))))
    ax = plt.subplot(144)
    ax.plot(y, values_g4)
    ax.set_xlabel('Y axis')
    ax.set_ylabel('Z axis')
    ax1.set_zlim(0.0, 2.0)
    # plt.axis('equal')

    #  test_calc_response
    w_1 = np.array(weights_1[0])
    mu_1 = np.array(means_1[0])
    cov_1 = np.array(sigmas_1[0])

    w_2 = np.array(weights_2[0])
    mu_2 = np.array(means_2[0])
    cov_2 = np.array(sigmas_2[0])
    w_3, mu_3, cov_3 = calc_response(w_1, mu_1, cov_1, w_2, mu_2, cov_2)

    w3arr, m3arr, c3arr = calc_response_gm(weights_1, means_1, sigmas_1, weights_2, means_2, sigmas_2)

    values_to_test = np.zeros(len(y))
    for w_3, mu_3, cov_3 in zip(w3arr, m3arr, c3arr):
        values_to_test += w_3 * multivariate_normal.pdf(y, mu_3, cov_3)
    ax.plot(y, values_to_test)
    plt.show()

# test_transfer_function()


def test_som():
    from som import GSom
    import utils
    # init som
    num_input_dim = 13
    num_output_dim = 2
    som = GSom(num_input_dim,  num_output_dim)
    num_components = 10
    for i in range(num_components):
        w = 1.
        mu = np.array(uniform(size=num_input_dim+num_output_dim))
        cov = np.array([0.006]*13)
        cov = np.append(cov, [0.005]*2)
        som.add(w, mu, cov)

    print som.weights
    print som.means

    # load sounds mfccs
    mfccs_a = utils.load_obj("Obj/mfcc_a.pkl")
    mfccs_i = utils.load_obj("Obj/mfcc_i.pkl")

    #  prepare data
    train_data = np.array(mfccs_a)
    train_data = np.append(train_data, mfccs_i, axis=0)

    max_val = np.amax(train_data)
    min_val = np.amin(train_data)

    for i in range(len(train_data)):
        for j in range(len(train_data[i])):
            train_data[i][j] = (train_data[i][j] - min_val) / (max_val - min_val)

    # train
    num_trains = 1000
    n_sigmas = np.linspace(0.2, 0.001, num_trains)
    for i in range(num_trains):
        sample = train_data[i % len(train_data)]
        cov = [0.006]*len(sample)
        mu = sample
        neighbour_sigma = n_sigmas[i]
        # som.train([1.], [mu], [cov], neighbour_sigma)

    # test and plot response map

    # test a
    x = np.linspace(0, 1, 20)
    y = np.linspace(0, 1, 20)
    X, Y = np.meshgrid(x, y)
    pos = np.empty(X.shape + (2,))
    pos[:, :, 0] = X
    pos[:, :, 1] = Y
    values = np.zeros((len(x), len(y)))
    k = len(som.weights)
    rvs = []
    for s in range(len(mfccs_a)):
        sample = train_data[s % len(train_data)]
        cov = [0.006] * len(sample)
        mu = sample
        w, m, c = som.propagate_gm_signal([1.], [mu], [cov])
        for i in range(len(w)):
            rvs.append(multivariate_normal(m[i], c[i]))
            values += w[i] * rvs[-1].pdf(pos)

    fig, ax = plt.subplots()
    # ax = fig.gca(projection="3d")
    # ax.plot_surface(X, Y, values, cmap='viridis', linewidth=0)
    # ax.set_xlabel('X axis')
    # ax.set_ylabel('Y axis')
    # ax.set_zlabel('Z axis')
    # plt.show()
    ax.pcolormesh(x, y, values)
    ax.set_title("a")

    # test i
    values = np.zeros((len(x), len(y)))
    k = len(som.weights)
    rvs = []
    for s in range(len(mfccs_a), len(train_data)):
        sample = train_data[s % len(train_data)]
        cov = [0.006] * len(sample)
        mu = sample
        w, m, c = som.propagate_gm_signal([1.], [mu], [cov])
        for i in range(len(w)):
            rvs.append(multivariate_normal(m[i], c[i]))
            values += w[i] * rvs[-1].pdf(pos)

    fig, ax = plt.subplots()
    # ax = fig.gca(projection="3d")
    # ax.plot_surface(X, Y, values, cmap='viridis', linewidth=0)
    # ax.set_xlabel('X axis')
    # ax.set_ylabel('Y axis')
    # ax.set_zlabel('Z axis')
    # plt.show()
    ax.pcolormesh(x, y, values)
    ax.set_title("i")




    plt.show()
    utils.save_obj(som, "som")
    return

# test_som()


def test_color_gsom():
    # prepare training data
    # generate colors
    np.random.seed(1996)
    from som import Som
    num_colors = 2
    raw_data = np.random.randint(0, 255, (3, num_colors))
    # raw_data = np.array([[255, 0, 0], [0, 255, 0], [0, 0, 255]])
    data = raw_data / 255.
    train_colors = data.transpose().reshape((int(data.size / 3), 1, 3))
    plt.subplot(1,3,1)
    plt.imshow(np.array(train_colors))
    print data
    print train_colors
    # generate data
    num_samples = 2000
    train_w = np.array([1.0] * num_samples)
    train_cov = [[0.001, 0.001, 0.001] for _ in range(num_samples)]
    train_m = [data[:, i] for i in np.random.randint(0, num_colors, num_samples)]

    # init som
    from som import GSom
    import utils
    # init som
    num_input_dim = 3
    num_output_dim = 2
    som = GSom(num_input_dim,  num_output_dim)
    num_components = 10
    for i in range(num_components):
        w = 1.
        mu = np.array(uniform(size=num_input_dim+num_output_dim))
        cov = np.array([0.01]*3)
        cov = np.append(cov, [0.005]*2)
        som.add(w, mu, cov)

    # plot initial state of the som
    x = np.linspace(0, 1, 50)
    y = np.linspace(0, 1, 50)
    stride = x[1]-x[0]
    X, Y = np.meshgrid(x, y)
    pos = np.empty(X.shape + (2,))
    pos[:, :, 0] = X
    pos[:, :, 1] = Y
    values = np.zeros((len(x), len(y), num_input_dim))
    values_without_opt = np.zeros((len(x), len(y), num_input_dim))
    k = len(som.weights)

    for i in range(len(x)):
        for j in range(len(y)):
            x_i = x[i]
            y_j = y[j]
            m = [[x_i, y_j]]
            cov = [[0.0001]*2]
            w = [1.]
            w_r, m_r, cov_r = som.propagate_gm_signal(w, m, cov, idx=[3,4])
            m_r = np.array(m_r)
            initial_guess = np.array([np.dot(w_r, m_r[:, k]) / sum(w_r) for k in range(num_input_dim)])

            idx = w_r.index(max(w_r))
            initial_guess = m_r[idx]

            # from scipy.optimize import minimize
            # methods = ["L-BFGS-B", "Powell", "Nelder-Mead", "TNC"]
            # opt_res = minimize(f, x0=initial_guess, args=(w_r, m_r, cov_r))
            #
            # unit_center = opt_res.x
            #
            # values[i, j] = unit_center
            values_without_opt[i, j] = initial_guess
            # print "unit {} : response {}, succes: {}".format([x_i, y_j], unit_center, opt_res.success)

    plt.subplot(1, 3, 2)
    plt.imshow(values)
    plt.subplot(1, 3, 3)
    plt.imshow(values_without_opt)

    # train som

    for w, m, cov, i in zip(train_w, train_m, train_cov, range(len(train_w))):
        print("\r{} out of {}".format(i, len(train_w)))
        print 0.1*(len(train_w)-i)/len(train_w)
        som.train([w], [m], [cov], 0.2*(len(train_w)-i)/len(train_w))

    # show result
    plt.figure()
    for i in range(len(x)):
        for j in range(len(y)):
            x_i = x[i]
            y_j = y[j]
            m = [[x_i, y_j]]
            cov = [[0.0001]*2]
            w = [1.]
            w_r, m_r, cov_r = som.propagate_gm_signal(w, m, cov, idx=[3, 4])
            m_r = np.array(m_r)
            initial_guess = np.array([np.dot(w_r, m_r[:, k]) / sum(w_r) for k in range(num_input_dim)])
            idx = w_r.index(max(w_r))
            initial_guess = m_r[idx]
            # from scipy.optimize import minimize
            # methods = ["L-BFGS-B", "Powell", "Nelder-Mead", "TNC"]
            # opt_res = minimize(f, x0=initial_guess, args=(w_r, m_r, cov_r))
            #
            # unit_center = opt_res.x
            #
            # values[i, j] = unit_center
            values_without_opt[i, j] = initial_guess
            # print "unit {} : response {}, succes: {}".format([x_i, y_j], unit_center, opt_res.success)
    plt.subplot(1,3,1)
    plt.imshow(train_colors)
    plt.subplot(1, 3, 2)
    plt.imshow(values)
    plt.subplot(1, 3, 3)
    plt.imshow(values_without_opt)

    plt.show()
    return

# test_color_gsom()


def generate_color_sequence_training_data(sequence_length, num_unique_colors, num_unique_sequences, num_samples_per_cluster):
    train_data = []
    unique_colors = np.random.randint(0, 255, (num_unique_colors, 3))
    unique_sequences = [unique_colors[np.random.randint(0, num_unique_colors, sequence_length)] for _ in range(num_unique_sequences)]
    for i in range(num_unique_sequences):
        ref_color = [unique_sequences[i] for _ in range(num_samples_per_cluster)]
        delta = [np.random.randint(0, 30, (sequence_length, 3)) for _ in range(num_samples_per_cluster)]
        train_data.extend(np.add(ref_color, delta))
    train_data = np.array(train_data).astype(float)
    # norm to range [0;1]
    for x in np.nditer(train_data, op_flags=[['readwrite']]):
        x[...] = 255 if x > 255 else x
        x[...] = 0 if x < 0 else x
        x[...] = x / 255.

    return train_data, unique_sequences, unique_colors


# train_data = generate_color_sequence_training_data(sequence_length=5, num_unique_colors=10, num_unique_sequences=5, num_samples_per_cluster=100)
# plt.imshow(np.asarray(train_data).astype(float), aspect='auto')
# plt.autoscale()
# plt.show()

def test_color_som():
    np.random.seed(1996)
    from som import Som
    raw_data = np.random.randint(0, 255, (3, 25))
    # raw_data = np.array([[255, 0, 0], [0, 255, 0], [0, 0, 255]])
    train_colors = raw_data.reshape((int(raw_data.size / 3), 1, 3))
    plt.subplot(1,3,1)
    plt.imshow(train_colors.astype(np.uint8))

    network_dimensions = np.array([40, 40])
    n_iterations = 2000
    init_learning_rate = 0.06
    # establish size variables based on data
    m = raw_data.shape[0]
    n = raw_data.shape[1]

    data = raw_data / 255.

    som = Som(network_dimensions, m)
    plt.subplot(1,3,2)
    plt.imshow(som.net)
    som.train(data, n_iterations, init_learning_rate)

    plt.subplot(1,3,3)
    plt.imshow(som.net)
    plt.show()

# test_color_som()


def test_tonotopic_som():
    import utils
    data_a = utils.load_obj("Obj/mfcc_a.pkl")
    data_i = utils.load_obj("Obj/mfcc_i.pkl")
    data_o = utils.load_obj("Obj/mfcc_o.pkl")
    data_u = utils.load_obj("Obj/mfcc_u.pkl")

    raw_data = np.concatenate((data_a, data_i, data_o, data_u))
    labels = np.concatenate((["a"]*len(data_a), ["i"]*len(data_i), ["o"]*len(data_o), ["u"]*len(data_u)))

    # normalize data
    data, maxes, mins = utils.normalize(raw_data, axis=0, norm_by_column=True)

    np.random.seed(1996)
    from som import Som
    network_dimensions = np.array([20, 20])
    n_iterations = 2000
    init_learning_rate = 0.03
    # establish size variables based on data
    data_t = np.transpose(data)
    m = data_t.shape[0]
    n = data_t.shape[1]

    som = Som(network_dimensions, m)

    som.train(data_t, n_iterations, init_learning_rate)

    # show topology

    markers = {"a": "^",
               "i": "o",
               "o": "x",
               "u": "d"}

    for i, label in enumerate(labels):
        if i % 100 == 0:
            _, idx = som.find_bmu(data_t[:, i])
            plt.scatter(idx[0], idx[1], marker=markers[label], c="black")
    plt.show()

    return

# test_tonotopic_som()


def test_sequence_color_som():
    np.random.seed(1991)
    from som import Som
    sequence_length = 10
    num_samples_per_cluster = 100
    train_data, unique_sequences, unique_colors = generate_color_sequence_training_data(sequence_length=sequence_length,
                                                                                        num_unique_colors=10,
                                                                                        num_unique_sequences=10,
                                                                                        num_samples_per_cluster=num_samples_per_cluster)
    plt.subplot(1,3,1)
    plt.imshow(np.asarray(train_data).astype(float), aspect='auto')
    plt.autoscale()
    flattened_data = train_data.reshape((int(train_data.size / 3), 3))
    flattened_data = np.transpose(flattened_data)
    network_dimensions = np.array([20, 20])
    n_iterations = 2000
    init_learning_rate = 0.05
    # establish size variables based on data
    m = flattened_data.shape[0]
    n = flattened_data.shape[1]

    som = Som(network_dimensions, m)
    plt.subplot(1, 3, 2)
    plt.imshow(som.net)
    som.train(flattened_data, n_iterations, init_learning_rate)

    utils.save_obj(som, "color_som")

    plt.subplot(1, 3, 3)
    plt.imshow(som.net)

    seq_train_data = []
    for i in range(train_data.shape[0]):
        sequence = train_data[i]
        responses = np.zeros(network_dimensions[0] * network_dimensions[1] * sequence_length)
        for j in range(len(sequence)):
            bmu, bmu_idx = som.find_bmu(sequence[j])
            responses[bmu_idx[0]*network_dimensions[0] + bmu_idx[1] + network_dimensions[0] * network_dimensions[1] * j] = 1.
        seq_train_data.append(responses)
    seq_train_data = np.array(seq_train_data).transpose()
    seq_som = Som(network_dimensions, seq_train_data.shape[0])
    seq_som.train(seq_train_data, n_iterations, init_learning_rate)

    utils.save_obj(seq_som, "seq_color_som")
    utils.save_obj((train_data, unique_sequences, unique_colors), "seq_color_train_data")

    plt.figure()
    plt.grid()
    colors = matplotlib.cm.rainbow(np.linspace(0, 1, len(unique_sequences)))
    for i in range(seq_train_data.shape[1]):
        sequence = seq_train_data[:, i]
        bmu, bmu_idx = seq_som.find_bmu(sequence)
        print bmu_idx[0], bmu_idx[1]
        print colors[i // num_samples_per_cluster]
        if i % num_samples_per_cluster == 0:
            plt.scatter(bmu_idx[0], bmu_idx[1], c=colors[i // num_samples_per_cluster], label="{}".format(i // num_samples_per_cluster), alpha=1.0)
        else:
            plt.scatter(bmu_idx[0], bmu_idx[1], c=colors[i // num_samples_per_cluster], alpha=0.5)


    plt.legend(loc=2)
    plt.show()

    return

# test_sequence_color_som()


def test_sequence_tonotpoic_som(sequences):
    import utils

    raw_data = []

    num_unique_sequences = len(sequences)
    num_samples_per_cluster = 0
    for s in sequences:
        data = utils.load_obj("Obj/mfcc_{}.pkl".format(s))
        num_samples_per_cluster = np.array(data).shape[0]
        raw_data.extend(data)
    raw_data = np.array(raw_data)
    #
    # data_aa = utils.load_obj("Obj/mfcc_aa.pkl")
    # data_ai = utils.load_obj("Obj/mfcc_ai.pkl")
    # data_ao = utils.load_obj("Obj/mfcc_ao.pkl")
    # data_au = utils.load_obj("Obj/mfcc_au.pkl")
    #
    # raw_data = np.concatenate((data_aa, data_ai))
    #
    # num_samples_per_cluster = np.array(data_aa).shape[0]
    # num_unique_sequences = 2

    # labels = np.concatenate((["aa"] * len(data_aa) * len(data_aa[0]), ["ai"] * len(data_ai) * len(data_aa[0]), ["ao"] * len(data_ao) * len(data_aa[0]), ["au"] * len(data_au) * len(data_aa[0])))

    # normalize data

    # data, maxes, mins = utils.normalize(raw_data, axis=0, norm_by_column=True)
    data = raw_data
    np.random.seed(1996)
    from som import Som
    network_dimensions = np.array([20, 20])
    n_iterations = 2000
    init_learning_rate = 0.04
    # establish size variables based on data
    flattened_data = data.reshape((int(data.size / 13), 13))

    data_t = np.transpose(flattened_data)
    data_t_n = utils.normalize(data_t, axis=1, norm_by_column=True)[0]
    data_train = data_t_n.transpose().reshape(raw_data.shape)
    m = data_t.shape[0]
    n = data_t.shape[1]

    som = Som(network_dimensions, m)

    som.train(data_t_n, n_iterations, init_learning_rate)

    sequence_length = raw_data.shape[1]

    seq_train_data = []
    for i in range(data_train.shape[0]):
        sequence = data_train[i]
        responses = np.zeros(network_dimensions[0] * network_dimensions[1] * sequence_length)
        for j in range(len(sequence)):
            bmu, bmu_idx = som.find_bmu(sequence[j])
            responses[bmu_idx[0]*network_dimensions[0] + bmu_idx[1] + network_dimensions[0] * network_dimensions[1] * j] = 1.
        seq_train_data.append(responses)
    seq_train_data = np.array(seq_train_data).transpose()
    seq_som = Som(network_dimensions, seq_train_data.shape[0])
    seq_som.train(seq_train_data, n_iterations, init_learning_rate)

    utils.save_obj(seq_som, "seq_tonotopic_som")
    # utils.save_obj((data_train, unique_sequences, unique_colors), "seq_color_train_data")

    plt.figure()
    plt.grid()
    colors = matplotlib.cm.rainbow(np.linspace(0, 1, num_unique_sequences))
    for i in range(seq_train_data.shape[1]):
        sequence = seq_train_data[:, i]
        bmu, bmu_idx = seq_som.find_bmu(sequence)
        print bmu_idx[0], bmu_idx[1]
        print colors[i // num_samples_per_cluster]
        if i % num_samples_per_cluster == 0:
            plt.scatter(bmu_idx[0], bmu_idx[1], c=colors[i // num_samples_per_cluster], label="{}".format(i // num_samples_per_cluster), alpha=1.0)
        else:
            plt.scatter(bmu_idx[0], bmu_idx[1], c=colors[i // num_samples_per_cluster], alpha=1.0)


    plt.legend(loc=2)
    plt.show()





    # # show topology
    # colors = matplotlib.cm.rainbow(np.linspace(0, 1, 4))
    # markers = {"aa": 0,
    #            "ai": 1,
    #            "ao": 2,
    #            "au": 3}
    #
    # for i, label in enumerate(labels):
    #     if i % 10 == 0:
    #         _, idx = som.find_bmu(data_t_n[:, i])
    #         plt.scatter(idx[0], idx[1], color=colors[markers[label]], alpha = 0.2)
    # plt.show()

    return

# utils.generate_training_data_VV("a", "a")
# utils.generate_training_data_VV("a", "i")
# utils.generate_training_data_VV("a", "o")
# utils.generate_training_data_VV("a", "u")

sequences = ['aa', 'ai', 'ao' , 'au', 'ii' , 'ia', 'iu', 'io', 'uu' , 'ua', 'ui', 'uo', 'oo','oa', 'oi', 'ou']
# for s in sequences:
#     utils.generate_training_data_VV(s[0], s[1])

# test_sequence_tonotpoic_som(sequences)


def test_tonotopic_som_l1():
    raw_data, maxes, mins = utils.load_obj("Obj/timit_mfccs.pkl")
    data = raw_data[:, :, :2]
    data = data.reshape((data.shape[0], data.shape[1]*2))
    n = data.shape[0]
    m = data.shape[1]

    np.random.seed(1996)
    from som import Som
    network_dimensions = np.array([30, 30])
    n_iterations = 200000
    init_learning_rate = 0.05
    # establish size variables based on data
    data_t = np.transpose(data)
    m = data_t.shape[0]
    n = data_t.shape[1]

    som = Som(network_dimensions, m)

    som.train(data_t, n_iterations, init_learning_rate, shuffle=False)

    # show topology

    utils.save_obj(som, "timit_l1_som")
    return

# test_tonotopic_som_l1()


def test_tonotopic_som_l2():
    print "loading timit"
    raw_data, maxes, mins = utils.load_obj("Obj/timit_mfccs.pkl")
    data = raw_data[:, :, :2]
    data = data.reshape((data.shape[0], data.shape[1] * 2))

    print "loading l1 SOM"

    som_l1 = utils.load_obj("Obj/timit_l1_som.pkl")

    np.random.seed(1996)
    from som import Som
    network_dimensions = np.array([30, 30])
    n_iterations = 20000
    init_learning_rate = 0.05
    sequence_length = 10

    m = som_l1.network_shape[0] * som_l1.network_shape[1] * sequence_length
    som_l2 = Som(network_dimensions, m)

    stride = sequence_length / 2

    l1_output = []
    print "generating training data for l2 SOM:"
    for i in range(data.shape[0]):
        sys.stdout.write('\r' + '{} out of {}'.format(i, data.shape[0]))
        l1_input = data[i]
        bmu, bmu_idx = som_l1.find_bmu(l1_input)
        l1_output.append([bmu_idx[0] / network_dimensions[0], bmu_idx[1] / network_dimensions[1]])

    l1_output = np.array(l1_output)
    utils.save_obj(l1_output, "l1_output")

    n = len(l1_output) // sequence_length
    l1_output_sequence = [np.array(v).reshape((sequence_length*2)) for v in np.array_split(l1_output[:sequence_length * n], n)]

    network_dimensions = np.array([30, 30])
    som_l2 = Som(network_dimensions, sequence_length*2)
    n_iterations = 20000
    learning_rate = 0.05
    print
    print "training:"
    l2_input = np.array(l1_output_sequence)
    utils.save_obj(l2_input, "l2_input")
    l2_input = l2_input.transpose()

    som_l2.train(l2_input, n_iterations, learning_rate)

    utils.save_obj(som_l2, "som_l2")
    return

# test_tonotopic_som_l2()


def test_tonotopic_som_l3():
    # l2_input = utils.load_obj("Obj/l2_input.pkl")
    # l2_som = utils.load_obj("Obj/som_l2.pkl")
    #
    # l2_output = []
    # print "generating training data for l3 SOM:"
    # for i in range(l2_input.shape[0]):
    #     sys.stdout.write('\r' + '{} out of {}'.format(i, l2_input.shape[0]))
    #     input = l2_input[i]
    #     bmu, bmu_idx = l2_som.find_bmu(input)
    #     l2_output.append([bmu_idx[0] / l2_som.network_shape[0], bmu_idx[1] / l2_som.network_shape[1]])
    #
    # l2_output = np.array(l2_output)
    # utils.save_obj(l2_output, "l2_output")
    l2_output = utils.load_obj("Obj/l2_output.pkl")

    sequence_length = 5
    n = len(l2_output) // sequence_length
    network_dimensions = np.array([30, 30])
    l3_input = [np.array(v).reshape((sequence_length * 2)) for v in
                          np.array_split(l2_output[:sequence_length * n], n)]

    np.random.seed(1996)
    from som import Som
    som_l3 = Som(network_dimensions, sequence_length * 2)
    n_iterations = 4000
    learning_rate = 0.05
    print
    print "training:"

    l3_input = np.array(l3_input).transpose()

    som_l3.train(l3_input, n_iterations, learning_rate)
    utils.save_obj(som_l3, "som_l3")
    return

# test_tonotopic_som_l3()


def test_tonotopic_mssom():
    from som import MultilayerSequenceSom as MSSom
    np.random.seed(1996)

    num_layers = 3
    sequences_lengths = [1, 10, 4]
    layers_shapes = [(10, 10)] * num_layers

    print "Loading training data..."
    raw_data, maxes, mins = utils.load_obj("Obj/timit_mfccs.pkl")
    data = raw_data[:2000, :, :2]
    data = data.reshape((data.shape[0], data.shape[1]*2))

    from som import Som
    network_dimensions = np.array([30, 30])
    n_iterations = [2000, 200, 50]
    learning_rates = [0.05] * num_layers
    # establish size variables based on data
    data = np.transpose(data)
    m = data.shape[0] # input_length
    n = data.shape[1] # num of samples

    print "Training data input vector length: {} \n number of samples: {}".format(m, n)

    mssom = MSSom(layers_shapes, sequences_lengths, m)
    mssom.train(data, n_iterations, learning_rates, dump=True)

    utils.save_obj(mssom, "mssom")
    return

test_tonotopic_mssom()


