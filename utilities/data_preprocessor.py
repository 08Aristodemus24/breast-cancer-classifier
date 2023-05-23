from sklearn.preprocessing import OrdinalEncoder, StandardScaler
import pandas as pd
from pandas.core.indexes.base import Index

def preprocess(df, feat_idxs: str | Index='all'):
    """# preprocess data
    - normalize
    - encode to numerical values Y column
    """

    # extract diagnosis as Y
    Y = df['diagnosis']

    # transform Y to 2-dim 1 x m matrix
    Y = Y.to_numpy().reshape(Y.shape[0], -1)

    # note that 1 is now the malignant class 
    # and 0 is the benign class/category
    oe = OrdinalEncoder()
    Y = oe.fit_transform(Y)

    # drop unnecessary columns
    X = df.drop(['id', 'Unnamed: 32', 'diagnosis'], axis=1, inplace=False)
    X =  X.loc[:, feat_idxs] if type(feat_idxs) == Index else X
    df_columns = X.columns
    
    # normalize X
    scaler = StandardScaler()
    X_normed = pd.DataFrame(scaler.fit_transform(X), columns=df_columns)

    return X_normed, Y