import numpy as np
import torch
import cvxpy as cp
from cvxpylayer import MyCPLayer
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

torch.manual_seed(42)
np.random.seed(42)
n = 2
eps = 0.01
X , y = make_blobs(50,2,centers=np.array([[3,3],[-3,-3]]),cluster_std=2.0)
trainX, testX, trainy, testy = train_test_split(X,y,test_size=0.5)
trainX, testX, trainy, testy = map(torch.from_numpy,[trainX, testX, trainy, testy])
trainX.requires_grad_(True)
m = trainX.shape[0]
a = cp.Variable((n,1))
b = cp.Variable((1,1))
X = cp.Parameter((m,n))
Y = trainy.numpy()[:,np.newaxis]
MLE = (1./m) * cp.sum(cp.multiply(Y,X@a+b)-cp.logistic(X@a+b))
regu = -0.1 * cp.norm(a,1)- 0.1 * cp.sum_squares(a)
prob = cp.Problem(cp.Maximize(MLE + regu))
cvxpylayer = MyCPLayer(prob,[X],[a,b])
a_tch,b_tch = cvxpylayer(trainX)
loss = 10000 * torch.nn.BCEWithLogitsLoss()((testX@a_tch+b_tch).squeeze(),testy*1.0)
loss.backward()
lr = LogisticRegression(solver='lbfgs')
lr.fit(testX.numpy(),testy.numpy())
beta_train = a_tch.detach().numpy().flatten()
beta_test = lr.coef_.flatten()
b_train = b_tch.squeeze().detach().numpy()
b_test = lr.intercept_[0]
attack_dataX = trainX + eps * trainX.grad
print(((attack_dataX - trainX)**2).sum())
grad = trainX.grad.numpy()
nptrainX = trainX.detach().numpy()
a = cp.Variable((n,1))
b = cp.Variable((1,1))
X = cp.Parameter((m,n))
Y = trainy.numpy()[:,np.newaxis]
MLE = (1./m) * cp.sum(cp.multiply(Y,X@a+b)-cp.logistic(X@a+b))
regu = -0.1 * cp.norm(a,1)- 0.1 * cp.sum_squares(a)
prob = cp.Problem(cp.Maximize(MLE + regu))
cvxpylayer = MyCPLayer(prob,[X],[a,b])
a_dtch,b_dtch = cvxpylayer(attack_dataX)
beta_attack = a_dtch.detach().numpy().flatten()
b_attack = b_dtch.squeeze().detach().numpy()
hyperplane = lambda x, beta, b: - (b + beta[0] * x) / beta[1]
nptrainy = trainy.numpy().astype(np.bool_)
plt.figure()
plt.scatter(nptrainX[nptrainy, 0], nptrainX[nptrainy, 1], s=25, marker='+')
plt.scatter(nptrainX[~nptrainy, 0], nptrainX[~nptrainy, 1], s=25, marker='*')
plt.xlim(-12, 12)
plt.ylim(-12, 12)
plt.plot(np.linspace(-12, 12, 100),
         [hyperplane(x, beta_train, b_train)
          for x in np.linspace(-12, 12, 100)], '--', color='red', label='train')
plt.plot(np.linspace(-12, 12, 100),
         [hyperplane(x, beta_test, b_test)
         for x in np.linspace(-12, 12, 100)], '-', color='blue', label='test')
plt.plot(np.linspace(-12, 12, 100),
         [hyperplane(x, beta_attack, b_attack)
         for x in np.linspace(-12, 12, 100)], '+', color='green', label='attack')
plt.legend()
plt.savefig("./data_poisoning.png")

