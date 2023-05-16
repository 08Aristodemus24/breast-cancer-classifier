from sklearn.preprocessing import OrdinalEncoder
import pandas as pd

def preprocess(df):
    """# preprocess data
    - normalize
    - encode to numerical values Y column
    """

    # delete id diagnosis
    Y = df['diagnosis']

    # transform Y to 2-dim 1 x m matrix
    Y = Y.to_numpy().reshape(Y.shape[0], -1)

    X = df.drop(['id', 'Unnamed: 32', 'diagnosis'], axis=1, inplace=False)

    # note that 1 is now the malignant class 
    # and 0 is the benign class/category
    oe = OrdinalEncoder()
    Y = oe.fit_transform(Y)
    return X, Y