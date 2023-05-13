import pandas as pd
import os
from sklearn import model_selection

if __name__ == "__main__":
    df = pd.read_csv("../../data/train.csv")
    df["kfold"] = -1
    #shuffle the data:
    df.sample(frac=1).reset_index(drop=True)

    kf = model_selection.StratifiedKFold(n_splits=5,shuffle=True,random_state=42)
    
    for fold,(train_index,val_index) in enumerate(kf.split(X=df,y = df.target.values)):
        print(len(train_index),len(val_index))
        df.loc[val_index,"kfold"] = fold
        
    df.to_csv("../../data/train_folds.csv",index=False)
        
        