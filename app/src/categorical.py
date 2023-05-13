import pandas as pd
from sklearn import preprocessing
from sklearn import linear_model

""" 
- label encoding
- one hot encoding 
- binarization 
"""

class CategoricalFeatures:
    def __init__(self, df, categorical_features, encoding_type, handle_na=False):
        """_Handling categorical variables_

        Args:
            df: Pandas dataframe
            categorical_features: List of all column names, e.g.["ord_1", "nom_1", ...]
            encoding_type: label, binary, onehot
            handle_na: True or False
        """
        self.df = df
        self.output_df = self.df.copy(deep=True)
        self.cat_feats = categorical_features
        self.enc_type = encoding_type
        self.label_encoders = dict()
        self.binary_encoders = dict()
        self.onehot_encoders = None
        self.handle_na = handle_na
        for c in self.cat_feats:
            self.df.loc[:,c] = self.df.loc[:,c].astype(str).fillna("-9999999")
            
    def _label_encoding(self):
        for c in self.cat_feats:
            lbl = preprocessing.LabelEncoder()
            #.values returns an array:
            lbl.fit(self.df[c].values)
            self.output_df.loc[:,c] = lbl.transform(self.df[c].values)
            self.label_encoders[c] = lbl
        return self.output_df
    
    def _label_binarization(self):
        for c in self.cat_feats:
            lbl = preprocessing.LabelBinarizer()
            lbl.fit(self.df[c].values)
            val = lbl.transform(self.df[c].values) #array
            self.output_df = self.output_df.drop(c, axis = 1)
            for j in range(val.shape[1]):
                new_col_name = c + f"__bin_{j}"
                self.output_df[new_col_name] = val[:, j]
            self.binary_encoders[c] = lbl 
        return self.output_df      
    
    def _onehot_encoder(self):
        ohe = preprocessing.OneHotEncoder()
        ohe.fit(self.df[self.cat_feats].values)
        return ohe.transform(self.df[self.cat_feats].values)
                        
    def fit_transform(self):
        if self.enc_type == "label":
            return self._label_encoding()
        
        elif self.enc_type == "binary":
            return self._label_binarization()
        
        elif self.enc_type == "ohe":
            return self._onehot_encoder()
        else:
            raise Exception("Encoding type not constructed!")
    
    def transform(self,dataframe):
        if self.handle_na:
            for c in self.cat_feats:
                dataframe.loc[:, c] = dataframe.loc[:, c].astype(str).fillna("-999999")
                
        self.output_df = self.df.copy(deep=True)
        if self.enc_type == "label":
            for c,lbl in self.label_encoders.items():
                dataframe.loc[:, c] = lbl.transform(dataframe[c].values)
            return dataframe
        
        elif self.enc_type == "binary":
            for c,lbl in self.binary_encoders.items():
                #binary encoders have results as array:
                val = lbl.transform(dataframe[c].values)
                dataframe = dataframe.drop(c, axis=1)
                for j in range(val.shape[1]):
                    new_col_name = c + f"__bin_{j}"
                    dataframe[new_col_name] = val[:, j]
            return dataframe

        elif self.enc_type == "ohe":
            return self.onehot_encoders(dataframe[self.cat_feats].values)
        
        else:
            raise Exception("Encoding type not understood !")
            

if __name__ == "__main__":
    df = pd.read_csv("../../data/categorical_variables/train.csv")
    df_test = pd.read_csv("../../data/categorical_variables/test.csv")
    sample = pd.read_csv("../../data/categorical_variables/sample_submission.csv")
    
    train_len = len(df)
    #create this fake column as test set doesnt have target column
    df_test["target"] = -1 
   # train_idx = df["id"].values
   # test_idx = df_test["id"].values
    #to avoid the error of unseen labels:
    full_data = pd.concat([df,df_test])
    columns = [c for c in full_data.columns if c not in ["id","target"]]
    print(columns)
    cat_feats = CategoricalFeatures(full_data,
                                    categorical_features=columns,
                                    encoding_type="ohe",
                                    handle_na=True,
                                    )
    full_data_transformed = cat_feats.fit_transform()
    #train_df = full_data_transformed[full_data_transformed["id"].isin(train_idx)].reset_index(drop=True)
    #test_df  = full_data_transformed[full_data_transformed["id"].isin(test_idx)].reset_index(drop=True)
    X = full_data_transformed[:train_len, :]
    X_test  = full_data_transformed[train_len:, :]
    
    #print(test_df.shape)
    #print(train_df.shape)

    clf = linear_model.LogisticRegression()
    clf.fit(X, df.target.values)
    preds = clf.predict_proba(X_test)[:, 1]
    
    sample.loc[:, "target"] = preds
    sample.to_csv("../../data/categorical_variables/submission.csv",index=False)
    
        
        