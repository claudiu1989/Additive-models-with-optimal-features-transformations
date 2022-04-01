from email.mime import base
import time
import copy
import numpy as np
import scipy.optimize as opt
import pandas as pd

import data_utils

from sklearn.metrics import mean_squared_error
from sklearn.model_selection import KFold

from sklearn.linear_model import LinearRegression

class OptimalFeaturesTransforamtionAdditiveModel:
    def __init__(self, lambda_param, number_of_points=10, remove_all_zero_columns=True, base='picewise_constant', C=1.0):
        self.number_of_points = number_of_points
        self.lambda_param = lambda_param
        self.remove_all_zero_columns = remove_all_zero_columns
        self.base = base
        self.C = C
        self.zero_columns_indices = []

    @staticmethod
    def __objective_function(beta,X_train_work, Y_train_work, lambda_param):
        n = X_train_work.shape[0]
        current_predictions = X_train_work.dot(beta)
        empirical_risk = np.sqrt((np.sum((current_predictions- Y_train_work)**2)/float(n)))
        obj_value = empirical_risk + lambda_param*np.linalg.norm(beta,ord=2)
        return obj_value
    
    def __transform(self, X):
        m = X.shape[1]
        n = X.shape[0]
        X_transformed = np.zeros((X.shape[0], m*self.number_of_points))
        if self.base == 'picewise_constant':
            X[X>=1.0] = 0.999
            limits = np.linspace(0.0, 1.0, num=self.number_of_points, endpoint=False)
            div = 1.0/float(self.number_of_points)
            for j, limit in enumerate(limits):
                    X_j = copy.deepcopy(X)
                    X_j[(X_j>=limit) & (X_j<limit + div)] = 1.0
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
        # Add bias
        bias_col = np.ones((n,1))
        X_transformed = np.append(X_transformed, bias_col, axis=1)
        # Add original features
        X_transformed = np.append(X_transformed, X, axis=1)
        return X_transformed

    def train(self, X_train,Y_train):
        X_train_work = self.__transform(X_train)
        beta = np.zeros(X_train_work.shape[1])
        #beta = np.random.rand(m*j_max)
        result = opt.minimize(OptimalFeaturesTransforamtionAdditiveModel.__objective_function,  beta, args=(X_train_work, Y_train, self.lambda_param),  tol=1e-3)
        self.beta_optim = result.x
        return self.beta_optim

    def predict(self, X):
        X_transformed = self.__transform(X)
        # Remove all-zeros columns
        if self.remove_all_zero_columns:
            X_transformed = np.delete(X_transformed, self.zero_columns_indices, axis=1)
        predictions = X_transformed.dot(self.beta_optim)
        return predictions

    def evaluate(self, X_test, Y_test):
        Y_predicted = self.predict(X_test)
        rmse = mean_squared_error(Y_test, Y_predicted, squared=False)
        #print(Y_test[:20])
        #print(Y_predicted[:20])
        #print(np.sqrt((np.sum((Y_predicted- Y_test)**2))/float(len(Y_predicted))))
        #print(rmse)
        return rmse
    
    def grid_search(self, X_train_validation, Y_train_validation, n_splits, lambda_values, number_of_points_values):
        avg_test_rmse_min = 0.0
        lambda_param_best = 0.0
        number_of_points_best = 1
        for lambda_param in lambda_values:
            for number_of_points in number_of_points_values:
                self.__init__(lambda_param, number_of_points, remove_all_zero_columns=True)
                avg_test_rmse = self.evaluate_k_fold(X_train_validation, Y_train_validation, n_splits)
                if avg_test_rmse >= avg_test_rmse_min:
                    avg_test_rmse_min = avg_test_rmse
                    lambda_param_best = lambda_param
                    number_of_points_best = number_of_points
        return lambda_param_best, number_of_points_best

    def evaluate_k_fold(self, X, Y, n_splits):
        # Setup
        test_rmse_list = list()
        training_rmse_list = list()
        training_time_list = list()
        for test_fold_index in range(n_splits):
            print("Process new folds...")
            (X_train, Y_train), (X_test, Y_test) = data_utils.get_train_test_fold(X,Y,test_fold_index+1,n_splits, stratified=False)
            start = time.time()
            beta_optim = self.train(X_train,Y_train)
            end = time.time()
            training_time_list.append(end-start)
            rmse = self.evaluate(X_train, Y_train)
            training_rmse_list.append(rmse)
            print('Training RMSE:', rmse)
            rmse = self.evaluate(X_test, Y_test)
            test_rmse_list.append(rmse)
            print('Test RMSE:', rmse)
        avg_training_rmse = sum(training_rmse_list)/float(n_splits)
        std_training_rmse = np.sqrt((np.sum((np.array(training_rmse_list) - avg_training_rmse)**2))/float(n_splits))
        avg_test_rmse = sum(test_rmse_list)/float(n_splits)
        std_test_rmse = np.sqrt((np.sum((np.array(test_rmse_list) - avg_test_rmse)**2))/float(n_splits))
       
        print('Average training RMSE: ', avg_training_rmse)
        print('Standard deviation of training RMSE: ', std_training_rmse)
        print('Average test RMSE: ', avg_test_rmse)
        print('Standard deviation of test RMSE:', std_test_rmse)
        print('Training time (s): ', sum(training_time_list)/float(n_splits))
        return avg_test_rmse
    
    def evaluate_k_fold_grid_search(self, X, Y, n_splits, lambda_values, number_of_points_values, b_values):
        # Setup
        test_rmse_list = list()
        training_rmse_list = list()
        training_time_list = list()
        for test_fold_index in range(n_splits):
            print("Process new folds...")
            (X_train_validation, Y_train_validation), (X_test, Y_test) = data_utils.get_train_test_fold(X,Y,test_fold_index+1,n_splits, stratified=False)
            lambda_param_best, number_of_points_best, b_best = self.grid_search(X_train_validation, Y_train_validation, n_splits, lambda_values, number_of_points_values, b_values)
            self.__init__(lambda_param_best, number_of_points_best, b_best, remove_all_zero_columns=True)
            start = time.time()
            beta_optim = self.train(X_train_validation,Y_train_validation)
            end = time.time()
            training_time_list.append(end-start)
            rmse = self.evaluate(X_train_validation, Y_train_validation)
            training_rmse_list.append(rmse)
            print('Training RMSE:', rmse)
            rmse = self.evaluate(X_test, Y_test)
            test_rmse_list.append(rmse)
            print('Test RMSE:', rmse)
        avg_training_rmse = sum(training_rmse_list)/float(n_splits)
        std_training_rmse = np.sqrt((np.sum((np.array(training_rmse_list) - avg_training_rmse)**2))/float(n_splits))
        avg_test_rmse = sum(test_rmse_list)/float(n_splits)
        std_test_rmse = np.sqrt((np.sum((np.array(test_rmse_list) - avg_test_rmse)**2))/float(n_splits))
       
        print('Average training RMSE: ', avg_training_rmse)
        print('Standard deviation of training RMSE: ', std_training_rmse)
        print('Average test RMSE: ', avg_test_rmse)
        print('Standard deviation of test RMSE:', std_test_rmse)
        print('Training time (s): ', sum(training_time_list)/float(n_splits))
        return lambda_param_best, number_of_points_best

def housing_k_fold_test():
    # Data 
    housing_data = pd.read_csv('./data/california_housing.csv')
    housing_data.drop(['No'], axis = 1, inplace = True)
    normalized_housing_data=(housing_data-housing_data.min())/(housing_data.max()-housing_data.min())
    Y = housing_data['Label'].to_numpy()
    X = normalized_housing_data.loc[:, normalized_housing_data.columns != 'Label'].to_numpy()
    # Model
    regularisation_param = 0.001
    number_of_points = 10
    oftam = OptimalFeaturesTransforamtionAdditiveModel(regularisation_param, number_of_points)
    # Evaluate
    n_splits = 5
    oftam.evaluate_k_fold(X, Y, n_splits)

def fico_k_fold_test():
    # Data 
    fico_data = pd.read_csv('./data/fico.csv')
    fico_data.drop(['No'], axis = 1, inplace = True)
    normalized_fico_data=(fico_data-fico_data.min())/(fico_data.max()-fico_data.min())
    Y = fico_data['ExternalRiskEstimate'].to_numpy()
    X = normalized_fico_data.loc[:, normalized_fico_data.columns != 'ExternalRiskEstimate'].to_numpy()
    # Model
    regularisation_param = 0.01
    number_of_points = 10
    oftam = OptimalFeaturesTransforamtionAdditiveModel(regularisation_param, number_of_points)
    # Evaluate
    n_splits = 5
    oftam.evaluate_k_fold(X, Y, n_splits)

def predict_linear_regression_fico():
    fico_data = pd.read_csv('./data/fico.csv')
    fico_data.drop(['No'], axis = 1, inplace = True)
    normalized_fico_data=(fico_data-fico_data.min())/(fico_data.max()-fico_data.min())
    Y = fico_data['ExternalRiskEstimate'].to_numpy()
    X = normalized_fico_data.loc[:, normalized_fico_data.columns != 'ExternalRiskEstimate'].to_numpy()
    test_rmse_list = list()
    training_time_list = list()
    n_splits = 5
    for test_fold_index in range(n_splits):
        print("Process new folds...")
        (X_train_validation, Y_train_validation), (X_test, Y_test) = data_utils.get_train_test_fold(X,Y,test_fold_index+1,n_splits)
        start = time.time()
        reg = LinearRegression().fit(X_train_validation,Y_train_validation)
        end = time.time()
        Y_redicted = reg.predict(X_test)
        rmse = mean_squared_error(Y_test, Y_redicted, squared=False)
        print('Test RMSE:', rmse)
        test_rmse_list.append(rmse)
    avg_test_rmse = sum(test_rmse_list)/float(n_splits)
    std_test_rmse = np.sqrt((np.sum((np.array(test_rmse_list) - avg_test_rmse)**2))/float(n_splits))
    print('Average test RMSE: ', avg_test_rmse)
    print('Standard deviation of test RMSE:', std_test_rmse)
    print('Training time (s): ', sum(training_time_list)/float(n_splits))

def predict_linear_regression_housing():
    housing_data = pd.read_csv('./data/california_housing.csv')
    housing_data.drop(['No'], axis = 1, inplace = True)
    normalized_housing_data=(housing_data-housing_data.min())/(housing_data.max()-housing_data.min())
    Y = housing_data['Label'].to_numpy()
    X = normalized_housing_data.loc[:, normalized_housing_data.columns != 'Label'].to_numpy()
    test_rmse_list = list()
    training_time_list = list()
    n_splits = 5
    for test_fold_index in range(n_splits):
        print("Process new folds...")
        (X_train_validation, Y_train_validation), (X_test, Y_test) = data_utils.get_train_test_fold(X,Y,test_fold_index+1,n_splits, stratified=False)
        start = time.time()
        reg = LinearRegression().fit(X_train_validation,Y_train_validation)
        end = time.time()
        Y_redicted = reg.predict(X_test)
        rmse = mean_squared_error(Y_test, Y_redicted, squared=False)
        print('Test RMSE:', rmse)
        test_rmse_list.append(rmse)
    avg_test_rmse = sum(test_rmse_list)/float(n_splits)
    std_test_rmse = np.sqrt((np.sum((np.array(test_rmse_list) - avg_test_rmse)**2))/float(n_splits))
    print('Average test RMSE: ', avg_test_rmse)
    print('Standard deviation of test RMSE:', std_test_rmse)
    print('Training time (s): ', sum(training_time_list)/float(n_splits))

def housing_k_fold_polynomial_test():
    # Data 
    housing_data = pd.read_csv('./data/california_housing.csv')
    housing_data.drop(['No'], axis = 1, inplace = True)
    normalized_housing_data=(housing_data-housing_data.min())/(housing_data.max()-housing_data.min())
    Y = housing_data['Label'].to_numpy()
    X = normalized_housing_data.loc[:, normalized_housing_data.columns != 'Label'].to_numpy()
    # Model
    regularisation_param = 0.001
    number_of_points = 15
    oftam = OptimalFeaturesTransforamtionAdditiveModel(regularisation_param, number_of_points, base='poly')
    # Evaluate
    n_splits = 5
    oftam.evaluate_k_fold(X, Y, n_splits)

def fico_k_fold_polynomial_test():
    # Data 
    fico_data = pd.read_csv('./data/fico.csv')
    fico_data.drop(['No'], axis = 1, inplace = True)
    normalized_fico_data=(fico_data-fico_data.min())/(fico_data.max()-fico_data.min())
    Y = fico_data['ExternalRiskEstimate'].to_numpy()
    X = normalized_fico_data.loc[:, normalized_fico_data.columns != 'ExternalRiskEstimate'].to_numpy()
    # Model
    regularisation_param = 0.001
    number_of_points = 5
    oftam = OptimalFeaturesTransforamtionAdditiveModel(regularisation_param, number_of_points, base='poly')
    # Evaluate
    n_splits = 5
    oftam.evaluate_k_fold(X, Y, n_splits)

def fico_k_fold_radial_test():
    # Data 
    fico_data = pd.read_csv('./data/fico.csv')
    fico_data.drop(['No'], axis = 1, inplace = True)
    normalized_fico_data=(fico_data-fico_data.min())/(fico_data.max()-fico_data.min())
    Y = fico_data['ExternalRiskEstimate'].to_numpy()
    X = normalized_fico_data.loc[:, normalized_fico_data.columns != 'ExternalRiskEstimate'].to_numpy()
    # Model
    regularisation_param = 0.001
    number_of_points = 60
    oftam = OptimalFeaturesTransforamtionAdditiveModel(regularisation_param, number_of_points, base='radi', C=20.0)
    # Evaluate
    n_splits = 5
    oftam.evaluate_k_fold(X, Y, n_splits)


def housing_k_fold_radial_test():
    # Data 
    housing_data = pd.read_csv('./data/california_housing.csv')
    housing_data.drop(['No'], axis = 1, inplace = True)
    normalized_housing_data=(housing_data-housing_data.min())/(housing_data.max()-housing_data.min())
    Y = housing_data['Label'].to_numpy()
    X = normalized_housing_data.loc[:, normalized_housing_data.columns != 'Label'].to_numpy()
    # Model
    regularisation_param = 0.0001
    number_of_points = 150
    oftam = OptimalFeaturesTransforamtionAdditiveModel(regularisation_param, number_of_points, base='radi', C=20.0)
    # Evaluate
    n_splits = 5
    oftam.evaluate_k_fold(X, Y, n_splits)

if __name__ == '__main__':
   #predict_linear_regression_housing()
   #housing_k_fold_test()
   #fico_k_fold_test()
   #predict_linear_regression_fico()
   #fico_k_fold_radial_test()
   #housing_k_fold_radial_test()
   housing_k_fold_test()