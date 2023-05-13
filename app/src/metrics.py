from typing import Any
from sklearn import metrics
import numpy as np

class RegressionMetrics:
    def __init__(self):
        self.metrics = {
            "mse"   : self._mse,
            "mae"   : self._mae,
            "rmse"  : self._rmse,
            "msle"  : self._msle,
            "rmsle" : self._rmsle,
            "r2"    : self._r2
        }


    def __call__(self, metric, y_true, y_pred):
        if metric not in self.metrics:
            raise Exception("Metric not found!")
        if metric == "mse":
            return self._mse(y_true, y_pred)
        if metric == "mae":
            return self._mae(y_true, y_pred)
        if metric == "rmse":
            return self._rmse(y_true, y_pred)
        if metric == "msle":
            return self._msle(y_true, y_pred)
        if metric == "rmsle":
            return self._rmsle(y_true, y_pred)
        if metric == "r2":
            return self._r2(y_true, y_pred)
        
    @staticmethod
    def _mse(y_true, y_pred):
        return metrics.mean_squared_error(y_true=y_true, y_pred=y_pred)

    @staticmethod
    def _mae(y_true, y_pred):
        return metrics.mean_absolute_error(y_true=y_true, y_pred=y_pred)
    
    def _rmse(self, y_true, y_pred):
        return np.sqrt(self._mse(y_true, y_pred))
    
    @staticmethod
    def _msle(y_true, y_pred):
        return metrics.mean_squared_log_error(y_true=y_true, y_pred=y_pred)
    
    def _rmsle(self, y_true, y_pred):
        return np.sqrt(self._msle(y_true, y_pred))
    
    @staticmethod
    def _r2(y_true, y_pred):
        return metrics.r2_score(y_true, y_pred)

    
class ClassificationMetrics:
    def __init__(self):
        self.metrics = {
            "accuracy": self._accuracy,
            "f1": self._f1,
            "recall": self._recall,
            "precision": self._precision,
            "auc": self._auc,
            "logloss": self._logloss
        }
        
        
    def __call__(self, metric, y_true, y_pred, y_proba=None):
        if metric not in self.metrics:
            raise Exception("Metrics out of scope!")
        
        if metric =="auc":
            if y_proba is not None:
                return self._auc(y_true=y_true, y_pred=y_proba)
            else:
                raise Exception("y_proba cannot be None")
            
        elif metric =="logloss":
            if y_proba is not None:
                return self._logloss(y_true=y_true, y_pred=y_proba)
            else:
                raise Exception("y_proba cannot be None")
            
        return self.metrics[metric](y_true=y_true, y_pred=y_pred)
    
    #we define as static method since it does not use the class itself
    @staticmethod 
    def _accuracy(y_true, y_pred):
        return metrics.accuracy_score(y_true=y_true, y_pred=y_pred)
    
    @staticmethod
    def _auc(y_true, y_pred):
        return metrics.roc_auc_score(y_true=y_true, y_score=y_pred)
        
    @staticmethod
    def _f1(y_true, y_pred):
        return metrics.f1_score(y_true=y_true, y_pred=y_pred)
    
    @staticmethod
    def _logloss(y_true, y_pred):
        return metrics.log_loss(y_true=y_true, y_pred=y_pred)
    
    @staticmethod
    def _precision(y_true, y_pred):
        return metrics.precision_score(y_true=y_true, y_pred=y_pred)
    
    @staticmethod
    def _recall(y_true, y_pred):
        return metrics.recall_score(y_true=y_true, y_pred=y_pred)
    
    
    
if __name__ == "__main__":
    from metrics import ClassificationMetrics
    from metrics import RegressionMetrics
    t = [0, 0, 1, 0, 1, 1]
    p = [0, 1, 0, 0, 1, 1]
    y1 = [1.5, 2.5, 0.5, 0.8]
    y2 = [1.2, 2.0, 0.7, 0.9]
    print("Accuracy score:",ClassificationMetrics()("accuracy", t, p))
    print("Recall score",ClassificationMetrics()("recall", t, p))
    print("AUC score:",ClassificationMetrics()("auc", t, p, [0.5, 0.5, 0.5, 0.6, 0.7, 0.8]))
    print("logloss metric",ClassificationMetrics()("logloss", t, p, [0.5, 0.5, 0.5, 0.6, 0.7, 0.8]))
    print("MSE",RegressionMetrics()("mse", y1, y2))
        
