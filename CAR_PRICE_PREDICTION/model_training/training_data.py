import sys
sys.path.append("D:\\Assignments_Assesment_UHG\\UHG_Assignment_2\\CAR_PRICE_PREDICTION")
# print("system path :", sys.path)

import os
import pandas as pd
import numpy as np

from data_cleaning import data_preprocessing  #######
 
from sklearn.preprocessing import LabelEncoder
import datetime
import pickle

from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.metrics import r2_score, mean_squared_error



class ModelTraining:     # super class
    def __init__(self):
        self.data = None
        self.preprocess = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None

        # self.self.car_price_df = pd.read_csv("D:\Persistent_Work\Assignment_2\Car details.csv")
        # print("*************init of parent class****************")

    def model_training(self):
        """
            Method Name/Function Name >> model_training
            Description >> This method is a base method that calls different methods for data_preprocessing tasks
            Parameters >> None
            Return >> It returns the final clean and preprocessed dataframe 
        """
        self.car_price_df = pd.read_csv(r"D:\Assignments_Assesment_UHG\UHG_Assignment_2\CAR_PRICE_PREDICTION\Car details.csv")

        preprocessor_instance = data_preprocessing.DataCleaning() # object of DataCleaning class

        self.car_price_df = preprocessor_instance.feature_splitting(self.car_price_df)
        # print(self.car_price_df.columns)
        self.car_price_df = preprocessor_instance.feature_creation_car_age(self.car_price_df)
        self.car_price_df = preprocessor_instance.feature_extraction_mileage(self.car_price_df)
        self.car_price_df = preprocessor_instance.feature_extraction_engine_size(self.car_price_df)
        self.car_price_df = preprocessor_instance.feature_extraction_maximum_power(self.car_price_df)
        self.car_price_df = preprocessor_instance.drop_irrelevent_feature(self.car_price_df, ["name", 'mileage', 'torque','engine', 'max_power','year'])
        self.car_price_df = preprocessor_instance.rename_features(self.car_price_df)
        self.car_price_df = preprocessor_instance.missing_values_imputation(self.car_price_df)
        self.car_price_df = preprocessor_instance.feature_encoding(self.car_price_df)
        # print("!@#111342453254643637345*&^%$$")

        return self.car_price_df

class ModelBuilding(ModelTraining):    # derived class
    def __init__(self):
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None 

        ModelTraining.__init__(self)
        ModelTraining.model_training(self)

        X = self.car_price_df.drop(['selling_price'], axis = 1)
        y = self.car_price_df['selling_price']
        # print("Training Columns :", X.columns)
        # print(X['Tata'].unique())
        # print(X['Tata'].dtypes)
    
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(X, y, test_size = 0.25, random_state = 7)
        
        # self.X_train.to_csv("train.csv")
        # self.X_test.to_csv("test.csv")

        # print("########### init of child ###############")

    def linear_regression_model(self):
        """
            Method Name/Function Name >> linear_regression_model
            Description >> Creates an instance of LinearRegression() and fits it to the training data (self.X_train and self.y_train).
            Uses the trained model to make predictions on both the training and testing data.
            Calculates the R-squared score for the training and testing data using the r2_score() function.
            Prints the R-squared score for the training and testing data.
            Parameters >> None
            Return >> It returns the R-squared score for the testing data.
        """
        print("************************ Linear Regression Model ***********************")
        self.linear_model = LinearRegression()
        self.linear_model.fit(self.X_train, self.y_train)
        train_prediction = self.linear_model.predict(self.X_train)
        test_prediction = self.linear_model.predict(self.X_test)
        
        # self.mse = mean_squared_error(self.y_train , self.result_train)
        # self.rmse = np.sqrt(self.mse)
        # self.mae = mean_absolute_error(self.y_train , self.result_train)
        # r2 value of testing data
        r2_score_linear_reg_train = r2_score(self.y_train,train_prediction)
        r2_score_linear_reg_test = r2_score(self.y_test,test_prediction)
        print("R2 Score Linear Regression Training data:", r2_score_linear_reg_train)
        print("R2 Score Linear Regression Testing data:", r2_score_linear_reg_test)

        return self.linear_model

    def decision_tree_model(self):
        """
            Method Name/Function Name >> decision_tree_model
            Description >> This method Creates an instance of DecisionTreeRegressor() and fits it to the training data (self.X_train and self.y_train).
            Uses the trained model to make predictions on both the training and testing data.
            Calculates the R-squared score for the training and testing data using the r2_score() function.
            Prints the R-squared score for the training and testing data.
            Parameters >> None
            Return >> It Returns the R-squared score for the testing data.
        """
        print("************************ Decision Tree Regressor Model ***********************")
        self.decision_tree_reg = DecisionTreeRegressor()
        self.decision_tree_reg.fit(self.X_train, self.y_train)
        train_prediction = self.decision_tree_reg.predict(self.X_train)
        test_prediction = self.decision_tree_reg.predict(self.X_test)
        
        # self.mse = mean_squared_error(self.y_train , self.result_train)
        # self.rmse = np.sqrt(self.mse)
        # self.mae = mean_absolute_error(self.y_train , self.result_train)
        # r2 value of testing data
        r2_score_decision_tree_reg_train = r2_score(self.y_train,train_prediction)
        r2_score_decision_tree_reg_test = r2_score(self.y_test,test_prediction)
        print("R2 Score Decision Tree Training data:", r2_score_decision_tree_reg_train)
        print("R2 Score Decision Tree Testing data:", r2_score_decision_tree_reg_test)
        
        return self.decision_tree_reg

    # def hypertune_decisiontree_model(self):
    #     print("*********** Decision Tree Regressor Model With Hyperparametr Tuning  ************")
    #     self.decision_tree_reg = DecisionTreeRegressor()
    #     self.decision_tree_reg.fit(self.X_train, self.y_train)
        
    #     hyper_parameters = {"criterion" : ["squared_error", "absolute_error", "poisson"],
    #               "max_depth" : np.arange(3,14),
    #               "min_samples_split" : np.arange(2,20),
    #               "min_samples_leaf" : np.arange(2,5)}

    #     randomized_search_cv = RandomizedSearchCV(decision_tree_reg, hyper_parameters, cv = 5)
    #     randomized_search_cv.fit(self.X_train, self.y_train)
    #     print("Best Estimators   :", randomized_search_cv.best_estimator_)
    #     print("Best Params :: :: :: " ,randomized_search_cv.best_params_ )

    #     ###### DT model with HyperParameter Tuning ##########
    #     self.rscv_dicision_tree_model = randomized_search_cv.best_estimator_
    #     self.rscv_dicision_tree_model.fit(self.X_train,self.y_train)
        
    #     train_prediction = self.rscv_dicision_tree_model.predict(self.X_train)
    #     test_prediction = self.rscv_dicision_tree_model.predict(self.X_test)
        
        
    #     # r2 value of testing data
    #     r2_score_dt_hyper_train = r2_score(self.y_train,train_prediction)
    #     r2_score_dt_hyper_test = r2_score(self.y_test,test_prediction)
    #     print("R2 Score Decision Tree Training data:", r2_score_dt_hyper_train)
    #     print("R2 Score Decision Tree Testing data:", r2_score_dt_hyper_test)
        
    #     return self.rscv_dicision_tree_model
    
    def best_model_finder(self):
        """
            Method Name/Function Name >> best_model_finder
            Description >> The purpose of this function is to find the best model between a linear model and a decision tree regressor model based on their accuracy scores on a test set. 
            The best model is then saved as a pickle file in a specified directory.Assuming that the instance variables self.linear_model, self.decision_tree_reg, self.X_test, and self.y_test have already been defined, 
            the function proceeds to compare the accuracy scores of both models on the test set using the score() method of each model. 
            If the linear model has a higher accuracy score, then it is selected as the best model and its instance is saved in the best_model variable. 
            Otherwise, the decision tree regressor model is selected as the best model and its instance is saved in the best_model variable.
            The os module is then used to check if a specified directory for saving the best model file exists or not. 
            If it does not exist, then a new directory is created. Finally, the pickle module is used to save the best model instance as a pickle file in the specified directory.
            Parameters >> None
            Return >> The function returns the best model instance.
        """
        if self.linear_model.score(self.X_test, self.y_test) > self.decision_tree_reg.score(self.X_test, self.y_test) :
            print("Linear Model Has More Accuracy than DT :",self.linear_model.score(self.X_test, self.y_test))
            self.best_model = self.linear_model
        else:
            print("DT Has More Accuracy than Linear Model :",self.decision_tree_reg.score(self.X_test, self.y_test))
            self.best_model = self.decision_tree_reg
        print("Best Model for training is ::::::", self.best_model)

        if not os.path.exists('D:\\Assignments_Assesment_UHG\\UHG_Assignment_2\\CAR_PRICE_PREDICTION\\saved_model'):
            os.mkdir('D:\\Assignments_Assesment_UHG\\UHG_Assignment_2\\CAR_PRICE_PREDICTION\\saved_model')
        else:
            pass

        path = os.path.join('D:\\Assignments_Assesment_UHG\\UHG_Assignment_2\\CAR_PRICE_PREDICTION\\saved_model')
        with open(path + "\\" "best_model.pkl", "wb") as f:
            pickle.dump(self.best_model, f)

        return self.best_model

    
    def test_prediction(self):
        """
            Method Name/Function Name >> test_prediction
            Description >> The purpose of this method is to test the performance of the best model that has been trained on a training dataset, 
            by making predictions on both the training data (stored in "X_train") and the testing data (stored in "X_test").
            The method then prints the first 5 predictions made on the testing data using the "print" function, 
            and returns the array of predictions made on the testing data.
            Parameters >> None
            Return >> It returns the results of the test set.
        """
        train_prediction = self.best_model.predict(self.X_train)
        test_prediction = self.best_model.predict(self.X_test)
        print("Prediction :: ", test_prediction[:5])
        return test_prediction

    def get_all_methods(self):
        """
            Method Name/Function Name >> get_all_methods
            Description >> This functions calls all the functions listed below like linear_regression_model etc.
            Parameters >> None
            Return >> It returns the results of the below functions.
        """
        self.linear_regression_model()
        self.decision_tree_model()
        # self.hypertune_decisiontree_model()
        self.best_model_finder()
        self.test_prediction()



        




# obj = ModelBuilding()
# # obj.linear_regression_model()
# # obj.decision_tree_model()
# # obj.hypertune_decisiontree_model()
# # obj.best_model_finder()
# # obj.test_prediction()

# obj.get_all_methods()
        