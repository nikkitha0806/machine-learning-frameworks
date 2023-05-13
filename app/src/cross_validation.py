import pandas as pd
from sklearn import model_selection
"""
Various problems that we can have:

- binary classification
- multi class classification
- multi label classification = single column having different labels like car,bike,lorry
- single column regression 
- multi column regression 
- holdout == these are mainly in case of time series data

"""
class CrossValidation:
    def __init__(
            self, 
            df, 
            target_cols, 
            shuffle,
            probelm_type="binary_classification",
            multilabel_delimiter = ",",
            num_folds=5,
            random_state=42
        ):
        self.dataframe = df
        self.target_cols = target_cols
        self.num_targets = len(target_cols)
        self.problem_type = probelm_type
        self.multilabel_delimiter = multilabel_delimiter
        self.num_folds = num_folds
        self.shuffle = shuffle
        self.random_state = random_state
        
        if self.shuffle is True:
            self.dataframe = self.dataframe.sample(frac=1).reset_index(drop=True)
        
        self.dataframe["kfold"] = -1
    
    def split(self):
        if self.problem_type in ("binary_classification","multiclass_classification"):
            if self.num_targets !=1:
                raise Exception("Invalid number of targets for this problem type")
            target = self.target_cols[0]
            unique_values = self.dataframe[target].nunique()
            if unique_values == 1:
                raise Exception("Only one unique value found!")
            elif unique_values > 1:
                kf = model_selection.StratifiedKFold(n_splits=self.num_folds,
                                                     shuffle=False)
                
                for fold,(train_index,val_index) in enumerate(kf.split(X=self.dataframe,y = self.dataframe[target].values)):
                    self.dataframe.loc[val_index,"kfold"] = fold   
            
        elif self.problem_type in ("single_col_regression","multi_col_regression"):
            if self.num_targets != 1 and self.problem_type=="single_col_regression":
                raise Exception("Invalid number of targets for this problem type")
            if self.num_targets < 2 and self.problem_type=="multi_col_regression":
                raise Exception("Invalid number of targets for this problem type")
            kf = model_selection.KFold(n_splits=self.num_folds,shuffle=False)
            for fold,(train_index,val_index) in enumerate(kf.split(X=self.dataframe)):
                self.dataframe.loc[val_index,"kfold"] = fold
        
        elif self.problem_type.startswith("holdout_"):
            holdout_percentage = int(self.problem_type.split("_")[1])
            num_holdout_samples = int(len(self.dataframe) * holdout_percentage / 100)
            self.dataframe.loc[:len(self.dataframe) - num_holdout_samples, "kfold"] = 0 
            self.dataframe.loc[len(self.dataframe) - num_holdout_samples:, "kfold"] = 1
        
        elif self.problem_type == "multi_label_classification":
            if self.num_targets != 1:
                raise Exception("Invalid number of targets for this problem type")
            targets = self.dataframe[self.target_cols[0]].apply(lambda x: len(str(x).split(self.multilabel_delimiter)))
            kf = model_selection.StratifiedKFold(n_splits=self.num_folds)
                
            for fold,(train_index,val_index) in enumerate(kf.split(X=self.dataframe,y = targets)):
                    self.dataframe.loc[val_index,"kfold"] = fold      
        
        else:
            raise Exception("Problem type not constructed yet")
                
        return self.dataframe
        
if __name__ == "__main__":
    df = pd.read_csv("../../data/multilabel_regression/train.csv")
    cv = CrossValidation(df,
                         shuffle=True,
                         target_cols=["attribute_ids"],
                         probelm_type="multi_label_classification",
                         multilabel_delimiter=" ")
    df_split = cv.split()
    print(df_split.head())
    print(df_split.kfold.value_counts())