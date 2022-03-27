import time
import copy
import numpy as np
import scipy.optimize as opt
import pandas as pd

import data_utils

from sklearn.metrics import mean_squared_error
from sklearn.model_selection import KFold

class OptimalFeaturesTransforamtionAdditiveModel:
    def __init__(self, lambda_param, number_of_points=10, remove_all_zero_columns=True):
        self.limits = np.linspace(0.0, 1.0, num=number_of_points, endpoint=False)
        self.div = 1.0/float(number_of_points)
        self.lambda_param = lambda_param
        self.remove_all_zero_columns = remove_all_zero_columns
        self.zero_columns_indices = []

    @staticmethod
    def __objective_function(beta,X_train_work, Y_train_work, lambda_param):
        n = X_train_work.shape[0]
        current_predictions = X_train_work.dot(beta)
        empirical_risk = (1.0/float(n))*np.linalg.norm(current_predictions - Y_train_work,ord=2) 
        obj_value = empirical_risk + lambda_param*np.linalg.norm(beta,ord=2)
        #print(obj_value)
        return obj_value

    def train(self, X_train,Y_train):
        n = X_train.shape[0]
        m = X_train.shape[1]
        j_max = len(self.limits)
        X_train_work = np.zeros((X_train.shape[0], m*j_max))
        X_train[X_train>=1.0] = 0.999
        for j, limit in enumerate(self.limits):
                X_j = copy.deepcopy(X_train)
                X_j[(X_j>=limit) & (X_j<limit + self.div)] = 1.0
                X_j[X_j<1.0] = 0.0
                X_train_work[:,j*m:(j+1)*m] = X_j
        # Remove all-zeros columns
        if self.remove_all_zero_columns: 
            self.zero_columns_indices = np.where(~X_train_work.any(axis=0))[0]
            X_train_work = np.delete(X_train_work, self.zero_columns_indices, axis=1)
            print(f'{len(self.zero_columns_indices)} columns were removed because they had only zeros.')
        bias_col = np.ones((n,1))
        # Add bias
        X_train_work = np.append(X_train_work, bias_col, axis=1)
        beta = np.zeros(X_train_work.shape[1])
        #beta = np.random.rand(m*j_max)
        result = opt.minimize(OptimalFeaturesTransforamtionAdditiveModel.__objective_function,  beta, args=(X_train_work, Y_train, self.lambda_param),  tol=1e-3)
        self.beta_optim = result.x
        return self.beta_optim

    def predict(self, X):
        j_max = len(self.limits)
        m = len(X[0])
        X_transformed = np.zeros((len(X),m*j_max))
        for j, limit in enumerate(self.limits):    
            X_j = copy.deepcopy(X)
            X_j[(X_j>=limit) & (X_j<limit + self.div)] = 1.0
            X_j[X_j<1.0] = 0.0
            X_transformed[:,j*m:(j+1)*m] = X_j
        n = X.shape[0]
        # Remove all-zeros columns
        if self.remove_all_zero_columns:
            X_transformed = np.delete(X_transformed, self.zero_columns_indices, axis=1)
        bias_col = np.ones((n,1))
        X_transformed = np.append(X_transformed, bias_col, axis=1)
        predictions = X_transformed.dot(self.beta_optim)
        return predictions

    def evaluate(self, X_test, Y_test):
        Y_predicted = self.predict(X_test)
        rmse = mean_squared_error(Y_test, Y_predicted, squared=False)
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
            (X_train_validation, Y_train_validation), (X_test, Y_test) = data_utils.get_train_test_fold(X,Y,test_fold_index+1,n_splits)
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
    normalized_comps_data=(housing_data-housing_data.min())/(housing_data.max()-housing_data.min())
    Y = normalized_comps_data['Label'].to_numpy()
    X = normalized_comps_data.loc[:, normalized_comps_data.columns != 'Label'].to_numpy()
    # Model
    regularisation_param = 0.01
    number_of_points = 60
    oftam = OptimalFeaturesTransforamtionAdditiveModel(regularisation_param, number_of_points)
    # Evaluate
    n_splits = 5
    oftam.evaluate_k_fold(X, Y, n_splits)
    
if __name__ == '__main__':
   housing_k_fold_test()