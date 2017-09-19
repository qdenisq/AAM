from numpy.random import normal, uniform
import numpy as np
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


np.random.seed(196835)
# generate k random gausiian components
k = 5

weights = [1.0]
weights.extend(uniform(-1.0, 1.0, k))
print "weights: ", weights
means = uniform(size=k)
sigmas = uniform(0.01, 0.03, k)

generator = [lambda: uniform()]
generator.extend([lambda i=i: normal(means[i], sigmas[i]) for i in range(k)])

pdfs = [lambda x: weights[0]]
for i in range(k):
    pdfs.append(lambda x, i=i: weights[i+1]*norm.pdf(x, means[i], sigmas[i]))
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




