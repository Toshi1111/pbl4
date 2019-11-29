import pandas as pd
import numpy as np
import warnings
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder
warnings.simplefilter('ignore')
train_columns = None
test_columns  = None

def reduce_mem_usage(props):
    start_mem_usg = props.memory_usage().sum() / 1024**2
    #print("Memory usage of properties dataframe is :",start_mem_usg," MB")
    NAlist = [] # Keeps track of columns that have missing values filled in.
    for col in props.columns:
        if props[col].dtype != object:  # Exclude strings

            # Print current column type
            #print("******************************")
            #print("Column: ",col)
            #print("dtype before: ",props[col].dtype)

            # make variables for Int, max and min
            IsInt = False
            mx = props[col].max()
            mn = props[col].min()

            # Integer does not support NA, therefore, NA needs to be filled
            if not np.isfinite(props[col]).all():
                NAlist.append(col)
                props[col].fillna(mn-1,inplace=True)

            # test if column can be converted to an integer
            asint = props[col].fillna(0).astype(np.int64)
            result = (props[col] - asint)
            result = result.sum()
            if result > -0.01 and result < 0.01:
                IsInt = True


            # Make Integer/unsigned Integer datatypes
            if IsInt:
                if mn >= 0:
                    if mx < 255:
                        props[col] = props[col].astype(np.uint8)
                    elif mx < 65535:
                        props[col] = props[col].astype(np.uint16)
                    elif mx < 4294967295:
                        props[col] = props[col].astype(np.uint32)
                    else:
                        props[col] = props[col].astype(np.uint64)
                else:
                    if mn > np.iinfo(np.int8).min and mx < np.iinfo(np.int8).max:
                        props[col] = props[col].astype(np.int8)
                    elif mn > np.iinfo(np.int16).min and mx < np.iinfo(np.int16).max:
                        props[col] = props[col].astype(np.int16)
                    elif mn > np.iinfo(np.int32).min and mx < np.iinfo(np.int32).max:
                        props[col] = props[col].astype(np.int32)
                    elif mn > np.iinfo(np.int64).min and mx < np.iinfo(np.int64).max:
                        props[col] = props[col].astype(np.int64)

            # Make float datatypes 32 bit
            else:
                props[col] = props[col].astype(np.float32)

            # Print new column type
            #print("dtype after: ",props[col].dtype)
            #print("******************************")

    # Print final result
    #print("___MEMORY USAGE AFTER COMPLETION:___")
    mem_usg = props.memory_usage().sum() / 1024**2
   #print("Memory usage is: ",mem_usg," MB")
    #print("This is ",100*mem_usg/start_mem_usg,"% of the initial size")
    return props, NAlist

#カラム数の調整
def fill_missing_columns(df_a, df_b):
    columns_for_b = set(df_a.columns) - set(df_b.columns)
    for column in columns_for_b:
        df_b[column] = 0
    columns_for_a = set(df_b.columns) - set(df_a.columns)
    for column in columns_for_a:
        df_a[column] = 0

purchase_df = pd.read_csv("./purchase_record.csv")
user_df = pd.read_csv("./user_info.csv")
purchase_df.fillna(0,inplace = True)
user_df.fillna(0,inplace = True)

train_df = pd.merge(purchase_df, user_df, how='left', on='user_id')
train_df = train_df.drop('user_id',axis=1)
train_df = train_df.drop('purchase_id',axis=1)
train_df = train_df.drop('date_x',axis=1)
train_df = train_df.drop('date_y',axis=1)
train_df = train_df*1
train_df, NAlist = reduce_mem_usage(train_df)
train = pd.get_dummies(train_df,drop_first=True)
train = train.drop('purchase',axis =1)


test_df = pd.read_csv("./purchase_record_test.csv")
test_df.fillna(0,inplace = True)
test_df = pd.merge(test_df, user_df, how='left', on='user_id')

test_df = test_df.drop('user_id',axis=1)
test_df = test_df.drop('date_x',axis=1)
test_df = test_df.drop('date_y',axis=1)
test_df = test_df*1
sample_test_df = test_df
test_df = test_df.drop('purchase_id',axis=1)
test_df, NAlist = reduce_mem_usage(test_df)
test = pd.get_dummies(test_df,drop_first=True)

fill_missing_columns(train,test)
test_df, NAlist = reduce_mem_usage(train)
test_df, NAlist = reduce_mem_usage(test)
y = train_df['purchase']
X = train
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33,random_state = 42)
logreg = LogisticRegression()
clf = AdaBoostClassifier(n_estimators=100, random_state=0)
clf.fit(X_train, y_train)
proba = clf.predict_proba(test)[:, 1]
sample_test_df['Probability'] = proba
print(clf.score(X_test,y_test))
submit_df = sample_test_df[['purchase_id', 'Probability']]
submit_df.to_csv('./submit_191128.csv', header=False, index=False, )
