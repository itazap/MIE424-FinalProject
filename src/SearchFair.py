#!/usr/bin/env python
# coding: utf-8


from sklearn.base import BaseEstimator
import sklearn.metrics.pairwise as kernels
from sklearn.metrics import confusion_matrix
import cvxpy as cp
import numpy as np


class SearchFair(BaseEstimator):
    """SearchFair
    
    Parameters

        fairness_notions: string
            The fairness scoring metric that will be used in the classifier, with options of DDP or DEO.
            
        fairness_regularizer: string
            The fairness relaxation which will be used as a regularization, with options of 'linear', 'wu', or 'wu_bound'.
            
        wu_bound: string
            The function to be used for the bounds defined by Wu, with options of 'hinge', 'logistic', 'squared', or 'exponential'.
            
        reg_beta: float
            The parameter beta used for L2 regularization.
            
        kernel: string
            The kernel to be used, with options of 'linear', 'rbf' or 'poly'.
            
        gamma: float
            The gamma that can be used for kernel='rbf' or kernel='poly'. If 'rbf', then gamma is kernel width. If 'poly', then gamma is degree.
            
        loss_name: string
            The name of the loss used, with options of 'hinge', 'logistic', 'squared', or 'exponential'.
            
        lambda_max: float
            The maximum lambda value used initially at the beginning of the binary search.
            
        max_iter: int
            The maximum number of iterations to be carried out by the solver.
            
        reason_points: float
            SearchFair's similarity-based approach: the percentage of points that are used as reasonable points.
            
        stop_criterion: float
            The stopping criteria for when the solver finds a classifier that is at least as fair as this value.
            
        max_search_iter: int
            The maxmimum number of iterations carried out in the binary search.
            
        solver: string
            The solver that is used by cvxpy, with options of 'SCS' or 'ECOS'.
            
        verbose: boolean
    
    
    Attributes

        coef_: numpy array
            The array that contains each reasonable point's the trained weights 
            
        reason_pts_index: numpy array
        The array that contains each training reasonable point's indices.
    """

    def __init__(self, fairness_notion='DDP', fairness_regularizer='wu', wu_bound='hinge', reg_beta=0.001, kernel='linear', gamma=None, loss_name='hinge', lambda_max=1, max_iter=3000, reason_points=0.5, stop_criterion=0.01, max_search_iter=10, solver='SCS', verbose=False):

        self.reg_beta = reg_beta
        self.fairness_notion = fairness_notion
        self.max_iter = max_iter
        self.max_search_iter = max_search_iter
        self.solver = solver
        self.verbose = verbose
        self.stop_criterion = stop_criterion
        self.reason_points = reason_points
        self.lambda_max = lambda_max
        self.wu_bound = wu_bound
        self.fairness_regularizer = fairness_regularizer
        self.wu_bound = wu_bound
        self.gamma = gamma
        self.loss_name = loss_name
        self.kernel = kernel

    def fit(self, x_train, y_train, s_train=None):
        """ To fit SearchFair on the training dataset
        .
        Parameters
        
            x_train: numpy array
                The shape is (num_points, num_features).
                The training data features.
                
            y_train: numpy array
                The shape is (num_points, ).
                The class labels of the training data.
                
            s_train: numpy array
                The shape is (num_points, ).
                The training data's binary sensitive attributes.
        
        Returns
        
            self: object
        """

        self.x_train = x_train
        self.y_train = y_train
        self.s_train = s_train

        if self.verbose:
            print("Preprocessing...")
        self._preprocess()

        lbda_min, lbda_max = 0, self.lambda_max

        def learn(reg, bound='upper'):
           
            self.fairness_lambda = reg
            if bound is not None:
                self._construct_problem(bound=bound)
            self._optimize()
            DDP, DEO = self.compute_fairness_measures(self.predict(x_train), y_train, s_train)
            if self.fairness_notion == 'DDP':
                fair_value = DDP
            else:
                fair_value = DEO
            if self.verbose: print("Obtained:",self.fairness_notion, "= %0.4f with lambda = %0.4f" % (fair_value, reg))
            return fair_value, self.coef_.copy()

        criterion = False
        bound = 'upper'
        
        if self.verbose: print("Testing lambda_min: %0.2f" % lbda_min)
        min_fair_measure, min_alpha = learn(lbda_min, bound=bound)
        if np.sign(min_fair_measure) < 0: bound = 'lower'
        if self.verbose: print("Testing lambda_max: %0.2f" % lbda_max)
        max_fair_measure, max_alpha = learn(lbda_max, bound)

        if np.abs(min_fair_measure) < np.abs(max_fair_measure):
            best_lbda, best_fair_measure = lbda_min, min_fair_measure
            best_alpha = min_alpha
        else:
            best_lbda, best_fair_measure = lbda_max, max_fair_measure
            best_alpha = max_alpha
        if  np.abs(best_fair_measure) < self.stop_criterion:
            print("Classifier is fair enough with lambda = {:.4f}".format(best_lbda))
        elif np.sign(min_fair_measure) == np.sign(max_fair_measure):
            print('Fairness value has the same sign for lambda_min and lambda_max.')
            
            #   Lambda could be reduced to try to get different results.
            print('Either try a different fairness regularizer or change the values of lambda_min and lambda_max')
        else:
            search_iter = 0
            if self.verbose: print("Starting Binary Search...")
            while not criterion and search_iter < self.max_search_iter:
                lbda_new = (lbda_min + lbda_max) / 2

                if self.verbose:
                    print(10*'-'+"Iteration #%0.0f" % search_iter + 10*'-')
                    print("Testing new Lambda: %0.4f" % lbda_new)

                new_rd, new_alpha = learn(lbda_new, None)
                if np.abs(new_rd) < np.abs(best_fair_measure):
                    best_fair_measure = new_rd
                    best_lbda = lbda_new
                    best_alpha = new_alpha.copy()

                if np.sign(new_rd) == np.sign(min_fair_measure):
                    min_fair_measure = new_rd
                    lbda_min = lbda_new
                else:
                    max_fair_measure = new_rd
                    lbda_max = lbda_new
                if np.abs(new_rd) < self.stop_criterion:
                    criterion = True

                search_iter += 1
            if search_iter==self.max_search_iter and self.verbose:
                print("Hit maximum iterations of Binary Search.")
            elif self.verbose:
                print("Sufficient fairness obtained before maximum iterations were reached.")

        if self.verbose: print(10*'-'+"Found Lambda %0.4f with fairness %0.4f" % (best_lbda, best_fair_measure)+10*'-')
        self.coef_ = best_alpha.copy()

        return self

    def predict(self, x_test):
        """Predict label for data points in the test set.
        
        Parameters
        
            x_test: numpy.array
                The shape is: (num_points, num_Features)
                The test data features.
                
        Returns
        
            y_hat: numpy.array
                The shape is: (num_points, ).
                Class labels that were predicted.
        """
        kernel_matr = self.kernel_function(x_test, self.x_train[self.reason_pts_index])
        y_hat = np.dot(self.coef_, np.transpose(kernel_matr))
        return np.sign(y_hat)

    def _preprocess(self):
        """Setting of loss function, kernel function and weight vectors of the attributes. These depend on the notion of fairness used, and appears
        in the objects related to fairness.
        """
        self.coef_ = None
        self.fairness_lambda = 0
        if self.loss_name == 'logistic':
            self.loss_func = lambda z: cp.logistic(-z)
        elif self.loss_name == 'hinge':
            self.loss_func = lambda z: cp.pos(1.0 - z)
        elif self.loss_name == 'squared':
            self.loss_func = lambda z: cp.square(-z)
        elif self.loss_name == 'exponential':
            self.loss_func = lambda z: cp.exp(-z)
        else:
            print('Using default loss: hinge loss.')
            self.loss_func = lambda z: cp.pos(1.0 - z)

        if self.kernel == 'rbf':
            self.kernel_function = lambda X, Y: kernels.rbf_kernel(X, Y, self.gamma)
        elif self.kernel == 'poly':
            self.kernel_function = lambda X, Y: kernels.polynomial_kernel(X, Y, degree=self.gamma)
        elif self.kernel == 'linear':
            self.kernel_function = lambda X, Y: kernels.linear_kernel(X, Y) + 1
        else:
            self.kernel_function = kernel

        if self.wu_bound == 'logistic':
            self.cvx_kappa = lambda z: cp.logistic(z)
            self.cvx_delta = lambda z: 1 - cp.logistic(-z)
        elif self.wu_bound == 'hinge':
            self.cvx_kappa = lambda z: cp.pos(1 + z)
            self.cvx_delta = lambda z: 1 - cp.pos(1 - z)
        elif self.wu_bound == 'squared':
            self.cvx_kappa = lambda z: cp.square(1 + z)
            self.cvx_delta = lambda z: 1 - cp.square(1 - z)
        elif self.wu_bound == 'exponential':
            self.cvx_kappa = lambda z: cp.exp(z)
            self.cvx_delta = lambda z: 1 - cp.exp(-z)
        else:
            print('Using default bound with hinge.')
            self.cvx_kappa = lambda z: cp.pos(1 + z)
            self.cvx_delta = lambda z: 1 - cp.pos(1 - z)

        self.nmb_pts = len(self.s_train)
        self.nmb_unprotected = np.sum(self.s_train == 1)
        self.prob_unprot = self.nmb_unprotected / self.nmb_pts
        self.prob_prot = 1 - self.prob_unprot

        self.nmb_pos = np.sum(self.y_train == 1)
        self.nmb_prot_pos = np.sum(self.y_train[self.s_train == -1] == 1)
        self.prob_prot_pos = self.nmb_prot_pos / self.nmb_pos
        self.prob_unprot_pos = 1 - self.prob_prot_pos

        #   Creation of the necessary weights for the fairness constraints.
        if self.fairness_notion == 'DDP':
            normalizer = self.nmb_pts
            self.weight_vector = np.array(
                [1.0 / self.prob_prot if self.s_train[i] == -1 else 1.0 / self.prob_unprot for i in range(len(self.s_train))]).reshape(-1,1)
            self.weight_vector = (1 / normalizer) * self.weight_vector
        elif self.fairness_notion == 'DEO':
            normalizer = self.nmb_pos
            self.weight_vector = np.array(
                [1.0 / self.prob_prot_pos if self.s_train[i] == -1 else 1.0 / self.prob_unprot_pos for i in range(len(self.s_train))]).reshape(-1, 1)
            self.weight_vector = 0.5 * (self.y_train.reshape(-1, 1) + 1) * self.weight_vector
            self.weight_vector = (1 / normalizer) * self.weight_vector

        #   Choose reasonable points at random
        if self.reason_points <= 1:
            self.reason_pts_index = list(range(int(self.nmb_pts * self.reason_points)))
        else:
            self.reason_pts_index = list(range(self.reason_points))
        self.nmb_reason_pts = len(self.reason_pts_index)

    def _construct_problem(self, bound='upper'):
        """ Formulize the min problem for cvxpy, depending on the regularizer chosen for fairness.
        """

        #   Optimization variable.
        self.alpha_var = cp.Variable((len(self.reason_pts_index), 1))
        
        #   Kernel matrix parameter.
        self.kernel_matrix = cp.Parameter(shape=(self.x_train.shape[0], len(self.reason_pts_index)))
        self.fair_reg_cparam = cp.Parameter(nonneg=True)


        # L2 regularized SVM formulization
        if self.fairness_lambda == 0:
            self.loss = cp.sum(self.loss_func(cp.multiply(self.y_train.reshape(-1, 1), self.kernel_matrix @ self.alpha_var))) + self.reg_beta * self.nmb_pts * cp.square(
                cp.norm(self.alpha_var, 2))
        else:
            sy_hat = cp.multiply(self.s_train.reshape(-1, 1), self.kernel_matrix @ self.alpha_var)

            if self.fairness_regularizer == 'wu':
                if bound == 'upper':
                    fairness_relaxation = cp.sum(cp.multiply(self.weight_vector, self.cvx_kappa(sy_hat))) - 1
                else:
                    fairness_relaxation = -1 * cp.sum(cp.multiply(self.weight_vector, self.cvx_delta(sy_hat))) - 1


            elif self.fairness_regularizer == 'linear':
                if bound == 'upper':
                    fairness_relaxation = cp.sum(cp.multiply(self.weight_vector, self.kernel_matrix @ self.alpha_var))
                else:
                    fairness_relaxation = -1 * cp.sum(cp.multiply(self.weight_vector, self.kernel_matrix @ self.alpha_var))

            if self.reg_beta == 0:
                self.loss = (1/self.nmb_pts) * cp.sum(self.loss_func(cp.multiply(self.y_train.reshape(-1, 1), self.kernel_matrix @ self.alpha_var))) +                                 self.fair_reg_cparam * fairness_relaxation
            else:
                self.loss = (1 / self.nmb_pts) * cp.sum(self.loss_func(cp.multiply(self.y_train.reshape(-1, 1), self.kernel_matrix @ self.alpha_var))) +                             self.fair_reg_cparam * fairness_relaxation + self.reg_beta * cp.square(cp.norm(self.alpha_var, 2))

        self.prob = cp.Problem(cp.Minimize(self.loss))

    def _optimize(self):
        """Optimize the formed problem using CVXPY's ECOS or SCS.
        """

        #   Initialize the kernel matrix
        self.K_sim = self.kernel_function(self.x_train, self.x_train[self.reason_pts_index])
        self.kernel_matrix.value = self.K_sim
        self.fair_reg_cparam.value = self.fairness_lambda

        if self.verbose == 2:
            verbose = True
        else:
            verbose = False
        if self.solver == 'SCS':
            self.prob.solve(solver=cp.SCS, max_iters=self.max_iter, verbose=verbose, warm_start=True)
        elif self.solver == 'ECOS':
            try:
                self.prob.solve(solver=cp.ECOS, max_iters=self.max_iter, verbose=verbose, warm_start=True)
            except Exception as e:
                self.prob.solve(solver=cp.SCS, max_iters=self.max_iter, verbose=verbose, warm_start=True)
        if verbose:
            print('status %s ' % self.prob.status)
            print('value %s ' % self.prob.value)
        self.coef_ = self.alpha_var.value.squeeze()
    def compute_fairness_measures(self, y_predicted, y_true, sens_attr):
        """Compute DP and EO values for predictions.
        
        Parameters
        
            y_predicted: numpy.array
                The shape is: (num_points, ).
                The predicted class labels.
                
            y_true: numpy.array
                The shape is: (num_points, ).
                The true class labels.
                
            sens_attr: numpy.array
                The shape is: (number_points, ).
                The sensitive labels.
        
        Returns
        
            DDP: float
                The demographic parity difference score.
            DEO: float
                The equality of opportunity difference score.
        """
        
        positive_rate_prot = self.get_positive_rate(y_predicted[sens_attr==-1], y_true[sens_attr==-1])
        positive_rate_unprot = self.get_positive_rate(y_predicted[sens_attr==1], y_true[sens_attr==1])
        true_positive_rate_prot = self.get_true_positive_rate(y_predicted[sens_attr==-1], y_true[sens_attr==-1])
        true_positive_rate_unprot = self.get_true_positive_rate(y_predicted[sens_attr==1], y_true[sens_attr==1])
        DDP = positive_rate_unprot - positive_rate_prot
        DEO = true_positive_rate_unprot - true_positive_rate_prot

        return DDP, DEO

    def get_positive_rate(self, y_predicted, y_true):
        """Compute the positive rate for given predictions of the class label.
        Parameters
        ----------
        y_predicted: numpy array
            The predicted class labels of shape=(number_points,).
        y_true: numpy array
            The true class labels of shape=(number_points,).
        Returns
        ---------
        pr: float
            The positive rate.
        """
        tn, fp, fn, tp = confusion_matrix(y_true, y_predicted).ravel()
        pr = (tp+fp) / (tp+fp+tn+fn)
        return pr

    def get_true_positive_rate(self, y_predicted, y_true):
        """Using class labels' predictions, calculate the true positive rate
        
        Parameters

            y_predicted: numpy.array
                The shape is (number_points, ).
                The predicted class labels.
                
            y_true: numpy.array
                The shape is (number_points, ).
                The true class labels.
                
        Returns
  
            tpr: float
                The true positive rate.
        """
        tn, fp, fn, tp = confusion_matrix(y_true, y_predicted).ravel()
        tpr = tp / (tp+fn)
        return tpr


# In[ ]:




