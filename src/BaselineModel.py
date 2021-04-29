#!/usr/bin/env python
# coding: utf-8

# In[2]:


from sklearn.base import BaseEstimator
import sklearn.metrics.pairwise as kernels
import cvxpy as cp
import numpy as np


# In[3]:


class BaselineModel(BaseEstimator):
    def __init__(self, l2_beta=0.001, kernel='linear', gamma=0.1, loss_name='hinge', lambda_max=1, max_iter=3000, solver='SCS', verbose=False,reason_points=0.5):

        #   L2 Regularization parameter beta
        self.l2_beta = l2_beta
        
        #   The SVM kernel can be 'linear', 'rbf', or 'poly'. 
        self.kernel = kernel
        
        #   if kernel='rbf', then Gamma = kernel_width. Else if kernel='poly', then Gamma = degree
        self.gamma = gamma
        
        #   Loss function can be 'hinge', 'logistic', 'squared', or 'exponential'.
        self.loss_name = loss_name
        
        #   Initial lambda_max value for beginning of binary search.
        self.lambda_max = lambda_max
        
        #   Maximum number of iterations to be completed.
        self.max_iter = max_iter
        
        #   CVXPY's solver to be used, which can either be 'SCS', or 'ECOS'.
        self.solver = solver
        
        #   If verobe is set to true, default of hiding the solver output will be overriden.
        self.verbose = verbose
        
        #   In the SearchFair's similarity based approach: the percentage of points to be used as 'reasonable points'
        self.reason_points = reason_points

    def fit(self, x_train, y_train,s_train):

        #   Setting of the class variables
        self.x_train = x_train
        self.y_train = y_train
        self.s_train = s_train
        
        #   preprocess() upon receiving data_point size
        self._preprocess()
        
        #   Construction of the CVXPY problem.
        self._construct_problem()
        
        #   Carry out optimization.
        self._optimize()
        
        return self

    def _preprocess(self):
       
        #   Using CVXPY's pre-built loss functions, initialize the coef_ array as a class variable.
        self.coef_ = None

        #   Using the loss_name provided, generate the respective loss function
        if self.loss_name == 'logistic':
            self.loss_func = lambda z: cp.logistic(-z)
        elif self.loss_name == 'hinge':
            self.loss_func = lambda z: cp.pos(1.0 - z)
        elif self.loss_name == 'squared':
            self.loss_func = lambda z: cp.square(-z)
        elif self.loss_name == 'exponential':
            self.loss_func = lambda z: cp.exp(-z)
        else:
            #   If no loss name provided OR different loss name than those defined, then use Hinge-Loss.
            self.loss_func = lambda z: cp.pos(1.0 - z)

        if self.kernel == 'rbf':
            self.kernel_function = lambda X, Y: kernels.rbf_kernel(X, Y, self.gamma)
        elif self.kernel == 'poly':
            self.kernel_function = lambda X, Y: kernels.polynomial_kernel(X, Y, degree=self.gamma)
        elif self.kernel == 'linear':
            self.kernel_function = lambda X, Y: kernels.linear_kernel(X, Y) + 1
        else:
            self.kernel_function = kernel


        #   Random reasonable points chosen:
        self.nmb_pts = len(self.s_train)
        if self.reason_points <= 1:
            self.reason_pts_index = list(range(int(self.nmb_pts * self.reason_points)))
        else:
            self.reason_pts_index = list(range(self.reason_points))
        self.nmb_reason_pts = len(self.reason_pts_index)

    def _construct_problem(self):

        #   Set the params variable to be optimized
        self.params = cp.Variable((len(self.reason_pts_index), 1))
        
        #   The Kernel Matrix to be set 
        self.kernel_matrix = cp.Parameter(shape=(self.x_train.shape[0], len(self.reason_pts_index)))
        
        self.bias = cp.Variable()

        #   The regularized loss function that we will minimize
        self.loss = (1 / self.nmb_pts) * cp.sum(self.loss_func(cp.multiply(self.y_train.reshape(-1, 1), self.kernel_matrix @ self.params))) + self.l2_beta * cp.square(cp.norm(self.params, 2))
        
        #   The final formulation of the problem.
        self.prob = cp.Problem(cp.Minimize(self.loss))

    def _optimize(self):

        self.K_sim = self.kernel_function(self.x_train, self.x_train[self.reason_pts_index])
        self.kernel_matrix.value = self.K_sim

        if self.solver == 'SCS':
            self.prob.solve(solver=cp.SCS, max_iters=self.max_iter, verbose=self.verbose, warm_start=True)
        elif self.solver == 'ECOS':
            try:
                self.prob.solve(solver=cp.ECOS, max_iters=self.max_iter, verbose=self.verbose, warm_start=True)
            except Exception as e:
                self.prob.solve(solver=cp.SCS, max_iters=self.max_iter, verbose=self.verbose, warm_start=True)
    
        self.coef_ = self.params.value.squeeze()
    
    def predict(self, x_test):
        kernel_matrix = self.kernel_function(x_test, self.x_train[self.reason_pts_index])
        
        # Calculation of the estimates
        y_hat = np.dot(self.coef_, np.transpose(kernel_matrix))
        
        return np.sign(y_hat)


# In[ ]:




