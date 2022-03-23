import pandas as pd
import pyreadstat
import numpy as np
import pickle
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score
from scipy.stats import pearsonr
import warnings

warnings.filterwarnings("ignore")

if __name__ == '__main__':
    df, meta = pyreadstat.read_sas7bdat('churn_data/looking_glass_v5.sas7bdat')
    np.sum(df['churn']) / len(df.index)

    xcols = ["count_of_suspensions_6m", "tot_drpd_pr1", "nbr_contacts", "calls_care_acct", "price_mention"]
    ycol = ['churn']

    # normalize data
    df['lifetime_value'] = np.log(df['lifetime_value'] + 1)
    df["Est_HH_Income"] = np.log(df["Est_HH_Income"] + 1)
    # df.to_csv('looking_glass_v5.csv')

    # df = pd.concat([df[df['churn'] == 1], df[df['churn'] == 0].sample(n=2354)], axis=0)
    df_notna = df[xcols + ycol].fillna(0)
    X = df_notna[xcols].values
    y = df_notna['churn'].values

    # impute na values and standardize
    # imputer = SimpleImputer(missing_values=np.nan, strategy='mean')
    # imputer.fit(X)
    # X = imputer.transform(X)
    # X = MinMaxScaler().fit_transform(X)
    # df_notna[xcols] = X

    # feature selection
    # X = SelectKBest(chi2, k=min(5, len(xcols))).fit_transform(X, y)
    # for i in range(np.shape(X)[1]):
    #     corr, _ = pearsonr(X[:, i], y)
    #     print(corr)

    # split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)

    # define the model
    model = LogisticRegression()
    # model = RandomForestClassifier()

    # fit the model on the train dataset
    model.fit(X_train, y_train)

    # make prediction for test
    y_pred = model.predict(X_test)

    cm = confusion_matrix(y_test, y_pred)
    print(cm)
    print(accuracy_score(y_test, y_pred))

    pickle.dump(model, open('churn_model.sav', 'wb'))
