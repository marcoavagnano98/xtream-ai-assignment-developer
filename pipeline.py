# 1 include more datasets
# 2 define preprocessing strategy
import pandas as pd
from abc import abstractmethod
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_absolute_error
import numpy as np
from datetime import datetime
import os
import joblib
import xgboost
import optuna

class BaseTrainer():
    def __init__(self, data, seed=42):
        
        self.data = data
        self.seed = seed
        self.model_name = "base" 
        self.processed_data = None
        self.res_path = "results"

    def __call__(self, **kwargs):
        self.preprocess()
        results = self.train_and_test(**kwargs)
        self.save(results)

        
    @abstractmethod
    def preprocess(self):
        raise NotImplementedError('You must specify how to preprocess your data')

    def prepare_set(self):
        x = self.processed_data.drop(columns='price')
        y = self.processed_data.price

        return train_test_split(x, y, test_size=0.2, random_state=self.seed)
    
    @abstractmethod
    def train_and_test(self):
        raise NotImplementedError('You must specify how to train your data')
    
    def score(self, y_test, preds):
        return {"R2": round(r2_score(y_test, preds), 4), "MAE": round(mean_absolute_error(y_test, preds), 2)}

    def save(self, results):
        tmp = datetime.now().strftime("%Y-%m-%d-%H-%M-%S.%f")
        results["timestamp"] = tmp
        if not os.path.exists(self.res_path):
            os.mkdir(self.res_path)
        
        save_path = os.path.join(self.res_path, f"{self.model_name}.csv")
        model_path = os.path.join(self.res_path, f"{self.model_name}-{tmp}.pkl")
        if not os.path.exists(save_path):
            header = True
        else:
            header = False
       
        log_df = pd.DataFrame(results, index=[0])
        log_df.to_csv(save_path, mode='a', header=header, index=False)

        joblib.dump(self.model, model_path)


class LinearTrainer(BaseTrainer):
    def __init__(self, data, seed=42):
        super().__init__(data, seed)
        self.model_name = "Linear"
        self.model = None

    def preprocess(self):
        self.processed_data = self.data.drop(columns=['depth', 'table', 'y', 'z'])
        self.processed_data = pd.get_dummies(self.data, columns=['cut', 'color', 'clarity'], drop_first=True)
       


    def train_and_test(self, log_process= False):
        x_train, x_test, y_train, y_test = self.prepare_set()
        if log_process:
            y_train = np.log(y_train)

        self.model = LinearRegression()
        self.model.fit(x_train, y_train)
        preds = self.model.predict(x_test)
        if log_process:
            preds = np.exp(preds)
        results = self.score(y_test, preds)
        results["log_p"] = log_process
        
        return results
    
class XGBoostTrainer(BaseTrainer):
    def __init__(self, data, seed=42):
        super().__init__(data, seed)
        self.model_name = "XGBoost"
        self.model = None

    def preprocess(self):
        self.processed_data = self.data.copy()
        self.processed_data["cut"] = pd.Categorical( self.processed_data['cut'], categories=['Fair', 'Good', 'Very Good', 'Ideal', 'Premium'], ordered=True)
        self.processed_data['color'] = pd.Categorical( self.processed_data['color'], categories=['D', 'E', 'F', 'G', 'H', 'I', 'J'], ordered=True)
        self.processed_data['clarity'] = pd.Categorical( self.processed_data['clarity'], categories=['IF', 'VVS1', 'VVS2', 'VS1', 'VS2', 'SI1', 'SI2', 'I1'], ordered=True)
    

    def xgboost_train(self,x, y, **model_params):
        self.model = xgboost.XGBRegressor(**model_params)
        self.model.fit(x, y)


    def objective(self, trial=None):
        
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
        model_params = {'random_state': self.seed, 'enable_categorical': True}
        if tuning_trials:
            study = optuna.create_study(direction='minimize', study_name='Diamonds XGBoost')
            study.optimize(self.objective, n_trials=tuning_trials)
            model_params.update(study.best_params)
        
        x_train, x_test, y_train, y_test = self.prepare_set()
        self.xgboost_train(x_train, y_train, **model_params)
        xgb_preds = self.model.predict(x_test)
        results = self.score(y_test, xgb_preds)
        results["tuning_trials"] = tuning_trials

        return results
            
         

        
if __name__ == "__main__":
    
    diamonds = pd.read_csv("data/diamonds.csv")
    
    # normalize data for all possible models
    diamonds = diamonds[(diamonds.x * diamonds.y * diamonds.z != 0) & (diamonds.price > 0)]

    linear_trainer = LinearTrainer(data=diamonds)
    xgboost_trainer = XGBoostTrainer(data=diamonds)

    linear_trainer()
    
    linear_trainer(log_process=True)

    xgboost_trainer()

    # hyperparameter tuning
    xgboost_trainer(tuning_trials = 100)