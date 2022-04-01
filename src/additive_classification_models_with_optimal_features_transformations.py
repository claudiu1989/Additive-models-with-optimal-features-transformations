import time
import copy
import numpy as np
import scipy.optimize as opt
import pandas as pd

import data_utils

from sklearn.metrics import accuracy_score
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from scipy.special import expit

class OptimalFeaturesTransforamtionAdditiveModel:
    def __init__(self, lambda_param, number_of_points=10, balance=1.0, remove_all_zero_columns=True, base='picewise_constant', C=1.0):
        self.limits = np.linspace(0.0, 1.0, num=number_of_points, endpoint=False)
        self.div = 1.0/float(number_of_points)
        self.lambda_param = lambda_param
        self.balance = balance
        self.remove_all_zero_columns = remove_all_zero_columns
        self.base = base
        self.C = C
        self.zero_columns_indices = []

    @staticmethod
    def __objective_function(beta,X_train_work, balanced_Y_train_work, lambda_param):
        n = X_train_work.shape[0]
        data_points_sums = X_train_work.dot(beta)
        exponents = balanced_Y_train_work*data_points_sums
        terms = np.exp(1-exponents)
        obj_value = (1.0/float(n))*np.sum(terms) + lambda_param*np.linalg.norm(beta,ord=2)
        #print(obj_value)
        return obj_value
    
    def __transform(self, X):
        j_max = len(self.limits)
        m = X.shape[1]
        X_transformed = np.zeros((X.shape[0], m*j_max))
        if self.base == 'picewise_constant':
            X[X>=1.0] = 0.999
            for j, limit in enumerate(self.limits):
                    X_j = copy.deepcopy(X)
                    X_j[(X_j>=limit) & (X_j<limit + self.div)] = 1.0
                    X_j[X_j<1.0] = 0.0
                    X_transformed[:,j*m:(j+1)*m] = X_j
        elif self.base == 'poly':
            X_pow = X
            for j, limit in enumerate(self.limits):
                    X_pow = X_pow*X
                    X_transformed[:,j*m:(j+1)*m] = X_pow
        else:
            for j, limit in enumerate(self.limits):
                    X_rad = np.exp(-self.C*np.power(X-limit, 2))
                    X_transformed[:,j*m:(j+1)*m] = X_rad
        return X_transformed

    def train(self, X_train,Y_train):
        Y_train_work = 2.0*Y_train - 1
        n = X_train.shape[0]
        X_train_work = self.__transform(X_train)
        # Remove all-zeros columns
        if self.remove_all_zero_columns: 
            self.zero_columns_indices = np.where(~X_train_work.any(axis=0))[0]
            X_train_work = np.delete(X_train_work, self.zero_columns_indices, axis=1)
            print(f'{len(self.zero_columns_indices)} columns were removed because they had only zeros.')
        bias_col = np.ones((n,1))
        # Add bias
        X_train_work = np.append(X_train_work, bias_col, axis=1)
        # Add original features
        X_train_work = np.append(X_train_work, X_train, axis=1)
        beta = np.zeros(X_train_work.shape[1])
        #beta = np.random.rand(m*j_max)
        if self.balance != 1.0:
            balanced_Y_train_work = np.array([self.balance * y if y > 0.0 else y for y in Y_train_work])
        else:
            balanced_Y_train_work = Y_train_work
        result = opt.minimize(OptimalFeaturesTransforamtionAdditiveModel.__objective_function,  beta, args=(X_train_work, balanced_Y_train_work, self.lambda_param),  tol=1e-3)
        self.beta_optim = result.x
        return self.beta_optim

    def predict(self, X):
        n = X.shape[0]
        X_transformed = self.__transform(X)
        # Remove all-zeros columns
        if self.remove_all_zero_columns:
            X_transformed = np.delete(X_transformed, self.zero_columns_indices, axis=1)
        bias_col = np.ones((n,1))
        X_transformed = np.append(X_transformed, bias_col, axis=1)
        # Add original features
        X_transformed = np.append(X_transformed, X, axis=1)
        predictions = X_transformed.dot(self.beta_optim)
        scores = expit(predictions)
        predictions[predictions <= 0] = 0
        predictions[predictions > 0] = 1
        return predictions, scores

    def evaluate(self, X_test, Y_test):
        Y_predicted, Y_predicted_scores = self.predict(X_test)
        accuracy = accuracy_score(Y_test, Y_predicted)
        auc = roc_auc_score(Y_test, Y_predicted_scores)
        return accuracy, auc
    
    def grid_search(self, X_train_validation, Y_train_validation, n_splits, lambda_values, number_of_points_values, b_values):
        avg_test_auc_min = 0.0
        lambda_param_best = 0.0
        number_of_points_best = 1
        b_best = 1.0
        for lambda_param in lambda_values:
            for number_of_points in number_of_points_values:
                for b in b_values:
                    self.__init__(lambda_param, number_of_points, b, remove_all_zero_columns=True)
                    avg_test_auc = self.evaluate_k_fold(X_train_validation, Y_train_validation, n_splits)
                    if avg_test_auc >= avg_test_auc_min:
                        avg_test_auc_min = avg_test_auc
                        lambda_param_best = lambda_param
                        number_of_points_best = number_of_points
                        b_best = b
        return lambda_param_best, number_of_points_best, b_best

    def evaluate_k_fold(self, X, Y, n_splits):
        # Setup
        test_accuracy_list = list()
        training_accuracy_list = list()
        test_auc_list = list()
        training_auc_list = list()
        training_time_list = list()
        for test_fold_index in range(n_splits):
            print("Process new folds...")
            (X_train, Y_train), (X_test, Y_test) = data_utils.get_train_test_fold(X,Y,test_fold_index+1,n_splits)
            start = time.time()
            beta_optim = self.train(X_train,Y_train)
            end = time.time()
            training_time_list.append(end-start)
            accuracy, auc = self.evaluate(X_train, Y_train)
            training_accuracy_list.append(accuracy)
            training_auc_list.append(auc)
            print('Training accuracy:', accuracy)
            print('Training AUC score:', auc)
            accuracy, auc = self.evaluate(X_test, Y_test)
            test_accuracy_list.append(accuracy)
            test_auc_list.append(auc)
            print('Test accuracy:', accuracy)
            print('Test AUC score:', auc)
        avg_training_acc = sum(training_accuracy_list)/float(n_splits)
        std_training_acc = np.sqrt((np.sum((np.array(training_accuracy_list) - avg_training_acc)**2))/float(n_splits))
        avg_training_auc = sum(training_auc_list)/float(n_splits)
        std_training_auc = np.sqrt((np.sum((np.array(training_auc_list) - avg_training_auc)**2))/float(n_splits))
        avg_test_acc = sum(test_accuracy_list)/float(n_splits)
        std_test_acc = np.sqrt((np.sum((np.array(test_accuracy_list) - avg_test_acc)**2))/float(n_splits))
        avg_test_auc = sum(test_auc_list)/float(n_splits)
        std_test_auc = np.sqrt((np.sum((np.array(test_auc_list) - avg_test_auc)**2))/float(n_splits))
        print('Average training accuracy: ', avg_training_acc)
        print('Standard deviation of training accuracy: ', std_training_acc)
        print('Average training AUC score: ', avg_training_auc)
        print('Standard deviation of training AUC score: ', std_training_auc)
        print('Average test accuracy: ', avg_test_acc)
        print('Standard deviation of test accuracy:', std_test_acc)
        print('Average test AUC score: ', avg_test_auc)
        print('Standard deviation of test AUC score: ', std_test_auc)
        print('Training time (s): ', sum(training_time_list)/float(n_splits))
        return avg_test_auc
    
    def evaluate_k_fold_grid_search(self, X, Y, n_splits, lambda_values, number_of_points_values, b_values):
        # Setup
        test_accuracy_list = list()
        training_accuracy_list = list()
        test_auc_list = list()
        training_auc_list = list()
        training_time_list = list()
        for test_fold_index in range(n_splits):
            print("Process new folds...")
            (X_train_validation, Y_train_validation), (X_test, Y_test) = data_utils.get_train_test_fold(X,Y,test_fold_index+1,n_splits)
            lambda_param_best, number_of_points_best, b_best = self.grid_search(X_train_validation, Y_train_validation, n_splits, lambda_values, number_of_points_values, b_values)
            self.__init__(lambda_param_best, number_of_points_best, b_best, remove_all_zero_columns=True)
            start = time.time()
            beta_optim = self.train(X_train_validation,Y_train_validation)
            end = time.time()
            training_time_list.append(end-start)
            accuracy, auc = self.evaluate(X_train_validation, Y_train_validation)
            training_accuracy_list.append(accuracy)
            training_auc_list.append(auc)
            print('Training accuracy:', accuracy)
            print('Training AUC score:', auc)
            accuracy, auc = self.evaluate(X_test, Y_test)
            test_accuracy_list.append(accuracy)
            test_auc_list.append(auc)
            print('Test accuracy:', accuracy)
            print('Test AUC score:', auc)
        avg_training_acc = sum(training_accuracy_list)/float(n_splits)
        std_training_acc = np.sqrt((np.sum((np.array(training_accuracy_list) - avg_training_acc)**2))/float(n_splits))
        avg_training_auc = sum(training_auc_list)/float(n_splits)
        std_training_auc = np.sqrt((np.sum((np.array(training_auc_list) - avg_training_auc)**2))/float(n_splits))
        avg_test_acc = sum(test_accuracy_list)/float(n_splits)
        std_test_acc = np.sqrt((np.sum((np.array(test_accuracy_list) - avg_test_acc)**2))/float(n_splits))
        avg_test_auc = sum(test_auc_list)/float(n_splits)
        std_test_auc = np.sqrt((np.sum((np.array(test_auc_list) - avg_test_auc)**2))/float(n_splits))
        print('Average training accuracy: ', avg_training_acc)
        print('Standard deviation of training accuracy: ', std_training_acc)
        print('Average training AUC score: ', avg_training_auc)
        print('Standard deviation of training AUC score: ', std_training_auc)
        print('Average test accuracy: ', avg_test_acc)
        print('Standard deviation of test accuracy:', std_test_acc)
        print('Average test AUC score: ', avg_test_auc)
        print('Standard deviation of test AUC score: ', std_test_auc)
        print('Training time (s): ', sum(training_time_list)/float(n_splits))
        return lambda_param_best, number_of_points_best, b_best

def compas_k_fold_test():
    # Data 
    comps_data = pd.read_csv('./data/compas.csv', delimiter=';')
    normalized_comps_data=(comps_data-comps_data.min())/(comps_data.max()-comps_data.min())
    Y = normalized_comps_data['two_year_recid'].to_numpy()
    X = normalized_comps_data.loc[:, normalized_comps_data.columns != 'two_year_recid'].to_numpy()
    # Model
    regularisation_param = 0.03
    number_of_points = 10
    balance = 1.0
    oftam = OptimalFeaturesTransforamtionAdditiveModel(regularisation_param, number_of_points, balance)
    # Evaluate
    n_splits = 5
    oftam.evaluate_k_fold(X, Y, n_splits)

def credit_k_fold_test():
    # Data 
    credit_data = pd.read_csv('./data/credit.csv')
    credit_data.drop(['No'], axis = 1, inplace = True)
    normalized_credit_data=(credit_data-credit_data.min())/(credit_data.max()-credit_data.min())
    Y = normalized_credit_data['Class'].to_numpy()
    X = normalized_credit_data.loc[:,normalized_credit_data.columns != 'Class'].to_numpy()
    # Model
    regularisation_param = 0.009
    number_of_points = 10
    oftam = OptimalFeaturesTransforamtionAdditiveModel(regularisation_param, number_of_points)
    # Evaluate
    n_splits = 5
    oftam.evaluate_k_fold(X, Y, n_splits)

def compas_k_fold_grid_search():
    # Data 
    comps_data = pd.read_csv('./data/compas.csv', delimiter=';')
    normalized_comps_data=(comps_data-comps_data.min())/(comps_data.max()-comps_data.min())
    Y = normalized_comps_data['two_year_recid'].to_numpy()
    X = normalized_comps_data.loc[:, normalized_comps_data.columns != 'two_year_recid'].to_numpy()
    # Model
    regularisation_param = 0.01
    number_of_points = 50
    oftam = OptimalFeaturesTransforamtionAdditiveModel(regularisation_param, number_of_points)
    # Evaluate
    n_splits = 5
    lambda_values = [0.09,0.01, 0.011]
    number_of_points_values = [60,70]
    b_values = [1.0]
    lambda_param_best, number_of_points_best, b_best = oftam.evaluate_k_fold_grid_search(X, Y, n_splits, lambda_values, number_of_points_values, b_values)
    oftam.evaluate_k_fold(X, Y, n_splits)
    print('Best lambda:', lambda_param_best)
    print('Best number of data points:', number_of_points_best)
    print('Best balance:', b_best)

def compas_k_fold_poly_test():
    # Data 
    comps_data = pd.read_csv('./data/compas.csv', delimiter=';')
    normalized_comps_data=(comps_data-comps_data.min())/(comps_data.max()-comps_data.min())
    Y = normalized_comps_data['two_year_recid'].to_numpy()
    X = normalized_comps_data.loc[:, normalized_comps_data.columns != 'two_year_recid'].to_numpy()
    # Model
    regularisation_param = 0.00001
    number_of_points = 122
    balance = 1.0
    C = 20.0
    base = 'poly'
    remove_0_columns = True
    oftam = OptimalFeaturesTransforamtionAdditiveModel(regularisation_param, number_of_points, balance, remove_0_columns, base, C)
    # Evaluate
    n_splits = 5
    oftam.evaluate_k_fold(X, Y, n_splits)

def compas_k_fold_radial_test():
    # Data 
    comps_data = pd.read_csv('./data/compas.csv', delimiter=';')
    normalized_comps_data=(comps_data-comps_data.min())/(comps_data.max()-comps_data.min())
    Y = normalized_comps_data['two_year_recid'].to_numpy()
    X = normalized_comps_data.loc[:, normalized_comps_data.columns != 'two_year_recid'].to_numpy()
    # Model
    regularisation_param = 0.00001
    number_of_points = 122
    balance = 1.0
    C = 20.0
    base = 'radi'
    remove_0_columns = True
    oftam = OptimalFeaturesTransforamtionAdditiveModel(regularisation_param, number_of_points, balance, remove_0_columns, base, C)
    # Evaluate
    n_splits = 5
    oftam.evaluate_k_fold(X, Y, n_splits)

def credit_k_fold_poly_test():
    # Data 
    credit_data = pd.read_csv('./data/credit.csv')
    credit_data.drop(['No'], axis = 1, inplace = True)
    normalized_credit_data=(credit_data-credit_data.min())/(credit_data.max()-credit_data.min())
    Y = normalized_credit_data['Class'].to_numpy()
    X = normalized_credit_data.loc[:,normalized_credit_data.columns != 'Class'].to_numpy()
    # Model
    regularisation_param = 0.009
    number_of_points = 10
    C = 20.0
    base = 'poly'
    remove_0_columns = True
    balance = 1
    oftam = OptimalFeaturesTransforamtionAdditiveModel(regularisation_param, number_of_points, balance, remove_0_columns,  base, C)
    # Evaluate
    n_splits = 5
    oftam.evaluate_k_fold(X, Y, n_splits)

def credit_k_fold_radi_test():
    # Data 
    credit_data = pd.read_csv('./data/credit.csv')
    credit_data.drop(['No'], axis = 1, inplace = True)
    normalized_credit_data=(credit_data-credit_data.min())/(credit_data.max()-credit_data.min())
    Y = normalized_credit_data['Class'].to_numpy()
    X = normalized_credit_data.loc[:,normalized_credit_data.columns != 'Class'].to_numpy()
    # Model
    regularisation_param = 0.01 
    number_of_points = 15
    C = 20.0
    base = 'radi'
    remove_0_columns = True
    balance = 1
    oftam = OptimalFeaturesTransforamtionAdditiveModel(regularisation_param, number_of_points, balance, remove_0_columns,  base, C)
    # Evaluate
    n_splits = 5
    oftam.evaluate_k_fold(X, Y, n_splits)

if __name__ == '__main__':
   #credit_k_fold_test()
   #compas_k_fold_test()
   #compas_k_fold_poly_test()
   #credit_k_fold_radi_test()
   compas_k_fold_test()