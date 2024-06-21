import os
import ast
import argparse
from abc import abstractmethod
from datetime import datetime
import joblib

import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_absolute_error

import xgboost
import optuna

class BaseTrainer():
    """
    A base class for training machine learning models.
    
    Attributes:
        data (pd.DataFrame): Dataset to be used for training.
        seed (int): Random seed for reproducibility.
        model_name (str): Name of the model.
        processed_data (pd.DataFrame): Data after preprocessing.
        res_path (str): Path to save the results.
    """    

    def __init__(self, data, seed=42):
        
        self.data = data
        self.seed = seed
        self.model_name = "base" 
        self.processed_data = None
        self.res_path = "results"

    def __call__(self, **kwargs):
        """
        Calls the pipeline to preprocess data, train model and save results.

        Args:
            **kwargs: Additional keyword arguments for different models training.
        """
        self.preprocess()
        results = self.train_and_test(**kwargs)
        self.save(results)

        
    @abstractmethod
    def preprocess(self):
        """
            Preprocess the data. The subclasses must override this methods because each model may have different preprocess.
        """
        raise NotImplementedError('You must specify how to preprocess your data')

    def prepare_set(self):
        """
            Split the diamond train, and test set.
            Returns:
                tuple: Split data (x_train, x_test, y_train, y_test).
        """
        x = self.processed_data.drop(columns='price')
        y = self.processed_data.price
        return train_test_split(x, y, test_size=0.2, random_state=self.seed)
    
    @abstractmethod
    def train_and_test(self):
        """
        Train the model and test it. This method must be overridden because each subclass has a different model

        Returns:
            dict: The results of the training and testing.
        """
        raise NotImplementedError('You must specify how to train your data')
    
    def score(self, y_test, preds):
        """
        Calculate performance metrics for the model predictions.

        Args:
            y_test (pd.Series): The true values.
            preds (np.ndarray): The predicted values.

        Returns:
            dict: The performance metrics.
        """
        return {"R2": round(r2_score(y_test, preds), 4), "MAE": round(mean_absolute_error(y_test, preds), 2)}

    def save(self, results):
        """
        Save the results and the model in results directory.

        Args:
            results (dict): The results to be saved.
        """
        
        tmp = datetime.now().strftime("%Y-%m-%d-%H-%M-%S.%f")
        results["timestamp"] = tmp
        print(results)
        if not os.path.exists(self.res_path):
            os.mkdir(self.res_path)
        
        save_path = os.path.join(self.res_path, f"{self.model_name}.csv")
        model_path = os.path.join(self.res_path, f"{self.model_name}-{tmp}.pkl")
        header = not os.path.exists(save_path)
        log_df = pd.DataFrame(results, index=[0])
        log_df.to_csv(save_path, mode='a', header=header, index=False)

        joblib.dump(self.model, model_path)


class LinearTrainer(BaseTrainer):
    """
    A trainer class for training a linear regression model.
    """
    def __init__(self, data, seed=42):
        super().__init__(data, seed)
        self.model_name = "Linear"
        self.model = None

    def preprocess(self):
        self.processed_data = self.data.drop(columns=['depth', 'table', 'y', 'z'])
        self.processed_data = pd.get_dummies(self.processed_data, columns=['cut', 'color', 'clarity'], drop_first=True)
        


    def train_and_test(self, log_trasf= False):
        """
        Train and test the linear regression model.

        Args:
            log_trasf (bool, optional): Whether to apply log transformation to the target variable. Default is False.

        """
        x_train, x_test, y_train, y_test = self.prepare_set()
        if log_trasf:
            y_train = np.log(y_train)
        
        print("Linear model training")
        self.model = LinearRegression()
        self.model.fit(x_train, y_train)
        preds = self.model.predict(x_test)
        if log_trasf:
            preds = np.exp(preds)
        results = self.score(y_test, preds)
        results["log_t"] = log_trasf
        return results
    
class XGBoostTrainer(BaseTrainer):
    """
    A trainer class for training an XGBoost model.
    """
    def __init__(self, data, seed=42):
        super().__init__(data, seed)
        self.model_name = "XGBoost"
        self.model = None

    def preprocess(self):
        self.processed_data = self.data.copy()
        self.processed_data["cut"] = pd.Categorical( self.processed_data['cut'], categories=['Fair', 'Good', 'Very Good', 'Ideal', 'Premium'], ordered=True)
        self.processed_data['color'] = pd.Categorical( self.processed_data['color'], categories=['D', 'E', 'F', 'G', 'H', 'I', 'J'], ordered=True)
        self.processed_data['clarity'] = pd.Categorical( self.processed_data['clarity'], categories=['IF', 'VVS1', 'VVS2', 'VS1', 'VS2', 'SI1', 'SI2', 'I1'], ordered=True)
        print(self.processed_data.keys())

    def xgboost_train(self,x, y, **model_params):
        """
        Train the XGBoost model with given parameters.

        Args:
            x (pd.DataFrame): The feature data.
            y (pd.Series): The target data.
            **model_params: Model initialization params.
        """
        self.model = xgboost.XGBRegressor(**model_params)
        self.model.fit(x, y)


    def objective(self, trial):
        """
        Define the objective function for Optuna to minimize.

        Args:
            trial (optuna.trial.Trial): A trial object for parameter optimization.

        Returns:
            float: The mean absolute error of the model predictions.
        """
        param = {
            'lambda': trial.suggest_float('lambda', 1e-8, 1.0, log=True),
            'alpha': trial.suggest_float('alpha', 1e-8, 1.0, log=True),
            'colsample_bytree': trial.suggest_categorical('colsample_bytree', [0.3, 0.4, 0.5, 0.7]),
            'subsample': trial.suggest_categorical('subsample', [0.5, 0.6, 0.7, 0.8, 0.9, 1.0]),
            'learning_rate': trial.suggest_float('learning_rate', 1e-8, 1.0, log=True),
            'n_estimators': trial.suggest_int('n_estimators', 100, 1000),
            'max_depth': trial.suggest_int('max_depth', 3, 9),
            'random_state': self.seed,
            'min_child_weight': trial.suggest_int('min_child_weight', 1, 10),
            'enable_categorical': True
        }
        x_train, x_test, y_train, y_test = self.prepare_set()
        
        self.xgboost_train(x_train, y_train, **param)
        
        xgb_preds = self.model.predict(x_test)
        
        scores = self.score(y_test, xgb_preds)
        return scores["MAE"]

        


    def train_and_test(self, tuning_trials=False):
        """
        Train and test the XGBoost model, optionally using Optuna for hyperparameter tuning.

        Args:
            tuning_trials (int, optional): The number of trials for Optuna tuning. Default is False.

        """
        model_params = {'random_state': self.seed, 'enable_categorical': True}
        if tuning_trials:
            study = optuna.create_study(direction='minimize',  study_name='Diamonds XGBoost')
            study.optimize(self.objective, n_trials=tuning_trials)
            model_params.update(study.best_params)
            
        print("XGBoost model training")
        x_train, x_test, y_train, y_test = self.prepare_set()
        self.xgboost_train(x_train, y_train, **model_params)
        xgb_preds = self.model.predict(x_test)
        results = self.score(y_test, xgb_preds)
        results["tuning_trials"] = tuning_trials

        return results
            
         

        
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
                    prog='TrainingPipeline',
                    description='Automatic pipeline for models training')

    parser.add_argument('-csv')
    parser.add_argument('-lr', action="store_true")
    parser.add_argument('-xg', action="store_true")
    
    args = parser.parse_args()
    csv_file = args.csv
    
    diamonds = pd.read_csv(csv_file)

    # normalize data for all possible models
    diamonds = diamonds[(diamonds.x * diamonds.y * diamonds.z != 0) & (diamonds.price > 0)]

    # it creates one model for each training session
    if args.lr:
        linear_trainer = LinearTrainer(data=diamonds)
        linear_trainer()
        linear_trainer(log_trasf=True)

    if args.xg:
        xgboost_trainer = XGBoostTrainer(data=diamonds)
        
        xgboost_trainer()
        t_trials = 100
        # hyperparameter tuning
        xgboost_trainer(tuning_trials = t_trials)
