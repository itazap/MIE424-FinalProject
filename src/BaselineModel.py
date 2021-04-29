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

        self.l2_beta = l2_beta # Regularization parameter beta for the l2 regularization
        self.kernel = kernel # The SVM kernel to be used.. Options:['linear','rbf','poly']
        self.gamma = gamma # If kernel='rbf', gamma to be kernel width, If kernel='poly', gamma to be degree.
        self.loss_name = loss_name # Loss function to be used. Options:['hinge','logistic','squared','exponential']
        self.lambda_max = lambda_max # The max lambda value for the start of the binary search.
        self.max_iter = max_iter # The number of iterations.
        self.solver = solver # The solver to be used by cvxpy. Options:['SCS','ECOS'].
        self.verbose = verbose # If true, Overrides the default of hiding solver output.
        self.reason_points = reason_points # The ratio of points used as reasonable points for the similarity-based approach of SearchFair.

    def fit(self, x_train, y_train,s_train):

        # Set class variables
        self.x_train = x_train
        self.y_train = y_train
        self.s_train = s_train
        
        # Must call preprocess() after the datapoint size is given
        self._preprocess()
        # Construct the CVXPY problem
        self._construct_problem()
        # Optimize with given settings 
        self._optimize()
        
        return self

    def _preprocess(self):
       
        # Initialize the coef_ array as a class variable using cvxpy's built-in loss functions
        self.coef_ = None

        # Generate loss function based on the given loss_name
        if self.loss_name == 'logistic':
            self.loss_func = lambda z: cp.logistic(-z)
        elif self.loss_name == 'hinge':
            self.loss_func = lambda z: cp.pos(1.0 - z)
        elif self.loss_name == 'squared':
            self.loss_func = lambda z: cp.square(-z)
        elif self.loss_name == 'exponential':
            self.loss_func = lambda z: cp.exp(-z)
        else:
            self.loss_func = lambda z: cp.pos(1.0 - z) # Use Hinge-Loss unless specified otherwise

        if self.kernel == 'rbf':
            self.kernel_function = lambda X, Y: kernels.rbf_kernel(X, Y, self.gamma)
        elif self.kernel == 'poly':
            self.kernel_function = lambda X, Y: kernels.polynomial_kernel(X, Y, degree=self.gamma)
        elif self.kernel == 'linear':
            self.kernel_function = lambda X, Y: kernels.linear_kernel(X, Y) + 1
        else:
            self.kernel_function = kernel


        # Choose random reasonable points
        self.nmb_pts = len(self.s_train)
        if self.reason_points <= 1:
            self.reason_pts_index = list(range(int(self.nmb_pts * self.reason_points)))
        else:
            self.reason_pts_index = list(range(self.reason_points))
        self.nmb_reason_pts = len(self.reason_pts_index)

    def _construct_problem(self):

        # Variable to optimize
        self.params = cp.Variable((len(self.reason_pts_index), 1))
        # Parameter for Kernel Matrix
        self.kernel_matrix = cp.Parameter(shape=(self.x_train.shape[0], len(self.reason_pts_index)))
        
        self.bias = cp.Variable()

        # Loss Function to Minimize (with Regularization)
        self.loss = (1 / self.nmb_pts) * cp.sum(self.loss_func(cp.multiply(self.y_train.reshape(-1, 1), self.kernel_matrix @ self.params))) + self.l2_beta * cp.square(cp.norm(self.params, 2))
        
        # Final Problem Formulization
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
        # Calculate Estimates
        y_hat = np.dot(self.coef_, np.transpose(kernel_matrix))
        
        return np.sign(y_hat)


# In[ ]:




