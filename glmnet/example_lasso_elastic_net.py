import numpy as np
from elastic_net import ElasticNet, Lasso, elastic_net_path, lasso_path

# ======================
# = Run Lasso and Enet =
# ======================
X = np.random.randn(50, 200)
w = 3*np.random.randn(200)
w[10:] = 0
y = np.dot(X, w)
enet = ElasticNet(alpha=0.1)
y_pred = enet.fit(X, y).predict(X)
print enet

# =====================
# = Compute full path =
# =====================

alphas_enet, coefs_enet, intercepts_enet = elastic_net_path(X, y, 0.2)
alphas_lasso, coefs_lasso, intercepts_lasso = lasso_path(X, y)

# ==============
# = show paths =
# ==============
from itertools import cycle
import pylab as pl
pl.close('all')

color_iter = cycle(['b', 'g', 'r', 'c', 'm', 'y', 'k'])
for color, coef_lasso, coef_enet in zip(color_iter,
                            coefs_lasso, coefs_enet):
    pl.plot(-np.log(alphas_lasso[1:]), coef_lasso[1:], color)
    pl.plot(-np.log(alphas_enet[1:]), coef_enet[1:], color+'x')

pl.xlabel('-log(lambda)')
pl.ylabel('coefs')
pl.title('Lasso and Elastic-Net Paths')
pl.legend(['Lasso','Elastic-Net'])
pl.show()

