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

class BaseTrainer():
    def __init__(self, data, seed=42):
        
        self.data = data
        self.seed = seed
        self.model_name = "base" 
        self.processed = False
        self.res_path = "results"
    
    @abstractmethod
    def preprocess(self):
        raise NotImplementedError('You must specify how to preprocess your data')

    @abstractmethod
    def train_and_test(self):
        raise NotImplementedError('You must specify how to train your data')
    
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
        

    def __call__(self, log_process = False, new_data = False):
        if new_data:
            self.data = new_data
        self.preprocess()
        results = self.train_and_test(log_process=log_process)
        self.save(results)


    def preprocess(self):
        self.processed_data = self.data.drop(columns=['depth', 'table', 'y', 'z'])
        self.processed_data = pd.get_dummies(self.data, columns=['cut', 'color', 'clarity'], drop_first=True)
       
    def prepare_set(self):
        x = self.processed_data.drop(columns='price')
        y = self.processed_data.price

        return train_test_split(x, y, test_size=0.2, random_state=self.seed)

    def train_and_test(self, log_process):
        x_train, x_test, y_train, y_test = self.prepare_set()
        if log_process:
            y_train = np.log(y_train)

        self.model = LinearRegression()
        self.model.fit(x_train, y_train)
        preds = self.model.predict(x_test)
        if log_process:
            preds = np.exp(preds)
        
        return {"R2": round(r2_score(y_test, preds), 4), "MAE": round(mean_absolute_error(y_test, preds), 2), "log_p": log_process}
    
if __name__ == "__main__":
    diamonds = pd.read_csv("data/diamonds.csv")
    
    # normalize data for all possible models
    diamonds = diamonds[(diamonds.x * diamonds.y * diamonds.z != 0) & (diamonds.price > 0)]

    linear_trainer = LinearTrainer(data=diamonds)

    # train and save models changing some parameters when object is called

    linear_trainer(log_process=False)
    
    linear_trainer(log_process=True)