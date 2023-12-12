import warnings; warnings.simplefilter('ignore')

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from sklearn import metrics
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler, LabelEncoder, MinMaxScaler, RobustScaler, Normalizer
from sklearn.model_selection import KFold, StratifiedKFold
from tqdm import tqdm_notebook as tqdm
import time
train = pd.read_csv('C:\\Users\\Saharat Srisawang\\Desktop\\Project\\test_ML_PJ\\UNSW_NB15_training-set.csv')
test = pd.read_csv('C:\\Users\\Saharat Srisawang\\Desktop\\Project\\test_ML_PJ\\UNSW_NB15_testing-set.csv')

if train.shape[0]<100000:
    print("Fixing train test")
    train, test = test, train

drop_columns = ['attack_cat','rate', 'id']
for df in [train, test]:
    for col in drop_columns:
        if col in df.columns:
            print('Dropping '+col)
            df.drop([col], axis=1, inplace=True)
print(df)
def detection_rate(y_true, y_pred):
    CM = metrics.confusion_matrix(y_true, y_pred)
    TN = CM[0][0]
    FN = CM[1][0]
    TP = CM[1][1]
    FP = CM[0][1]
    return TP/(TP+FN)

def false_positive_rate(y_true, y_pred):
    CM = metrics.confusion_matrix(y_true, y_pred)
    TN = CM[0][0]
    FN = CM[1][0]
    TP = CM[1][1]
    FP = CM[0][1]
    return FP/(FP+TN)

def false_alarm_rate(y_true, y_pred):
    CM = metrics.confusion_matrix(y_true, y_pred)
    TN = CM[0][0]
    FN = CM[1][0]
    TP = CM[1][1]
    FP = CM[0][1]
    return (FP+FN)/(TP+TN+FP+FN)

def get_xy(df):
    return pd.get_dummies(df.drop(['label'], axis=1)), df['label']

def get_cat_columns(train):
    categorical = []
    for col in train.columns:
        if train[col].dtype == 'object':
            categorical.append(col)
    return categorical

def label_encode(train, test):
    for col in get_cat_columns(train):
        le = LabelEncoder()
        le.fit(list(train[col].astype(str).values) + list(test[col].astype(str).values))
        train[col] = le.transform(list(train[col].astype(str).values))
        test[col] = le.transform(list(test[col].astype(str).values))
    return train, test

def process_feature(df):
    df.loc[~df['state'].isin(['FIN', 'INT', 'CON', 'REQ', 'RST']), 'state'] = 'others'
    df.loc[~df['service'].isin(['-', 'dns', 'http', 'smtp', 'ftp-data', 'ftp', 'ssh', 'pop3']), 'service'] = 'others'
    df.loc[df['proto'].isin(['igmp', 'icmp', 'rtp']), 'proto'] = 'igmp_icmp_rtp'
    df.loc[~df['proto'].isin(['tcp', 'udp', 'arp', 'ospf', 'igmp_icmp_rtp']), 'proto'] = 'others'
    return df

def get_train_test(train, test, feature_engineer=True, label_encoding=False, scaler=StandardScaler()):
    x_train, y_train = train.drop(['label'], axis=1), train['label']
    x_test, y_test = test.drop(['label'], axis=1), test['label']
    
    if feature_engineer:
        x_train, x_test = process_feature(x_train), process_feature(x_test)
    
    categorical_columns = get_cat_columns(x_train)
    non_categorical_columns = [x for x in x_train.columns if x not in categorical_columns]
    if scaler is not None:
        x_train[non_categorical_columns] = scaler.fit_transform(x_train[non_categorical_columns])
        x_test[non_categorical_columns] = scaler.transform(x_test[non_categorical_columns])

    if label_encoding:
        x_train, x_test = label_encode(x_train, x_test)
        features = x_train.columns
    else:
        x_train = pd.get_dummies(x_train)
        x_test = pd.get_dummies(x_test)
        print("Column mismatch {0}, {1}".format(set(x_train.columns)- set(x_test.columns),  set(x_test.columns)- set(x_train.columns)))
        features = list(set(x_train.columns) & set(x_test.columns))
    print(f"Number of features {len(features)}")
    x_train = x_train[features]
    x_test = x_test[features]

    return x_train, y_train, x_test, y_test


def detection_rate(y_true, y_pred):
    CM = metrics.confusion_matrix(y_true, y_pred)
    TN = CM[0][0]
    FN = CM[1][0]
    TP = CM[1][1]
    FP = CM[0][1]
    return TP / (TP + FN)

def false_positive_rate(y_true, y_pred):
    CM = metrics.confusion_matrix(y_true, y_pred)
    TN = CM[0][0]
    FN = CM[1][0]
    TP = CM[1][1]
    FP = CM[0][1]
    return FP / (FP + TN)

def false_alarm_rate(y_true, y_pred):
    CM = metrics.confusion_matrix(y_true, y_pred)
    TN = CM[0][0]
    FN = CM[1][0]
    TP = CM[1][1]
    FP = CM[0][1]
    return (FP + FN) / (TP + TN + FP + FN)

def get_xy(df):
    return pd.get_dummies(df.drop(['label'], axis=1)), df['label']

def get_cat_columns(train):
    categorical = []
    for col in train.columns:
        if train[col].dtype == 'object':
            categorical.append(col)
    return categorical

def label_encode(train, test):
    for col in get_cat_columns(train):
        le = LabelEncoder()
        le.fit(list(train[col].astype(str).values) + list(test[col].astype(str).values))
        train[col] = le.transform(list(train[col].astype(str).values))
        test[col] = le.transform(list(test[col].astype(str).values))
    return train, test

def process_feature(df):
    df.loc[~df['state'].isin(['FIN', 'INT', 'CON', 'REQ', 'RST']), 'state'] = 'others'
    df.loc[~df['service'].isin(['-', 'dns', 'http', 'smtp', 'ftp-data', 'ftp', 'ssh', 'pop3']), 'service'] = 'others'
    df.loc[df['proto'].isin(['igmp', 'icmp', 'rtp']), 'proto'] = 'igmp_icmp_rtp'
    df.loc[~df['proto'].isin(['tcp', 'udp', 'arp', 'ospf', 'igmp_icmp_rtp']), 'proto'] = 'others'
    return df

def get_train_test(train, test, feature_engineer=True, label_encoding=False, scaler=StandardScaler()):
    x_train, y_train = train.drop(['label'], axis=1), train['label']
    x_test, y_test = test.drop(['label'], axis=1), test['label']
    
    if feature_engineer:
        x_train, x_test = process_feature(x_train), process_feature(x_test)
    
    categorical_columns = get_cat_columns(x_train)
    non_categorical_columns = [x for x in x_train.columns if x not in categorical_columns]
    if scaler is not None:
        x_train[non_categorical_columns] = scaler.fit_transform(x_train[non_categorical_columns])
        x_test[non_categorical_columns] = scaler.transform(x_test[non_categorical_columns])

    if label_encoding:
        x_train, x_test = label_encode(x_train, x_test)
        features = x_train.columns
    else:
        x_train = pd.get_dummies(x_train)
        x_test = pd.get_dummies(x_test)
        print("Column mismatch {0}, {1}".format(set(x_train.columns)- set(x_test.columns),  set(x_test.columns)- set(x_train.columns)))
        features = list(set(x_train.columns) & set(x_test.columns))
    print(f"Number of features {len(features)}")
    x_train = x_train[features]
    x_test = x_test[features]

    return x_train, y_train, x_test, y_test

def results(y_test, y_pred):
    acc = metrics.accuracy_score(y_test, y_pred)
    pre = metrics.precision_score(y_test, y_pred)
    rec = metrics.recall_score(y_test, y_pred)
    f1 = metrics.f1_score(y_test, y_pred)
    print(f"Acc {acc}, Precision {pre}, Recall {rec}, F1-score {f1}")
    
    CM = metrics.confusion_matrix(y_test, y_pred)
    TN = CM[0][0]
    FN = CM[1][0]
    TP = CM[1][1]
    FP = CM[0][1]
    
    # detection rate or true positive rate
    DR = TP*100/(TP+FN)
    # false positive rate
    FPR = FP*100/(FP+TN)
    # false alarm rate 
    FAR = (FP+FN)*100/(TP+TN+FP+FN)
    
    print("DR {0}, FPR {1}, FAR {2}".format(DR, FPR, FAR))
    print(metrics.classification_report(y_test, y_pred))
   # ... (previous code)

def cross_validation(params, X, Y):
    y_probs = []
    y_vals = []

    # for tr_idx, val_idx in tqdm(kf.split(X, Y), total=folds):
    for tr_idx, val_idx in kf.split(X, Y):
        x_train, y_train = X.iloc[tr_idx], Y[tr_idx]
        x_val, y_val = X.iloc[val_idx], Y[val_idx]
        clf = RandomForestClassifier(**params)
        clf.fit(x_train, y_train)
        y_prob = clf.predict_proba(x_val)[:, 1]
        
        y_probs.append(y_prob)
        y_vals.append(y_val)
        
    acc, pre, rec, f1, far, fpr, dr, auc = 0, 0, 0, 0, 0, 0, 0, 0
    folds = len(y_probs)
    for i in range(folds):
        y_prob, y_val = y_probs[i], y_vals[i]
        y_pred = np.where(y_prob >= 0.5, 1, 0)

        acc += metrics.accuracy_score(y_val, y_pred) / folds
        f1 += metrics.f1_score(y_val, y_pred) / folds
        pre += metrics.precision_score(y_val, y_pred) / folds
        rec += metrics.recall_score(y_val, y_pred) / folds
        dr += detection_rate(y_val, y_pred) / folds
        fpr += false_positive_rate(y_val, y_pred) / folds
        far += false_alarm_rate(y_val, y_pred) / folds
        auc += metrics.roc_auc_score(y_val, y_prob) / folds
    
    print(f"Acc {acc}, Precision {pre}, Recall {rec}, F1-score {f1} \nFAR {far}, FPR {fpr}, DR {dr} , AUC {auc}")

# ... (remaining code)

    
def test_run(params, X, Y):
    clf = RandomForestClassifier(**params)
    clf.fit(X, Y)
    y_pred = clf.predict(x_test)
    results(y_test, y_pred)
    
    y_prob = clf.predict_proba(x_test)[:, 1]
    print("Auc {0}".format(metrics.roc_auc_score(y_test, y_prob)))
folds = 10
seed = 1
kf = StratifiedKFold(n_splits=folds, shuffle=True, random_state=seed)
params = {
    'n_estimators': 100,
    'random_state':1,
    'class_weight': {0:2, 1:1}
}
X, Y, x_test, y_test = get_train_test(train, test, feature_engineer=False, label_encoding=True, scaler=None)

clf = RandomForestClassifier()
clf.fit(X,Y)
y_pred = clf.predict(X)
results(Y, y_pred)
cross_validation(params, X, Y)
test_run(params, X, Y)
# Drop Features with low importance
drop_columns = ['response_body_len', 'is_sm_ips_ports', 'ct_flw_http_mthd', 'trans_depth', 'dwin', 'ct_ftp_cmd', 'is_ftp_login']
for df in [train, test]:
    df.drop(drop_columns, axis=1, inplace=True)
X, Y, x_test, y_test = get_train_test(train, test, feature_engineer=True, label_encoding=False, scaler=RobustScaler())
params = {
    'random_state':1,
    'class_weight': {0:2, 1:1}
}
start_time = time.clock()
cross_validation(params, X, Y)
print("Time spent in 10-fold cross validation of train data ", time.clock()-start_time)

start_time = time.clock()
test_run(params, X, Y)
print("Time spent in test run ", time.clock()-start_time)
X, Y, x_test, y_test = get_train_test(train, test, feature_engineer=True, label_encoding=False, scaler=RobustScaler())
for n_estimators in [20, 50]:
    for max_features in [10, 30]:
        print("n_estimators {0} max_features {1}".format(n_estimators, max_features))
        params = {
           'n_estimators': n_estimators,
            'random_state':1,
            'max_depth':10,
            'max_features': max_features,
            'class_weight': {0:2, 1:1}
        }
        cross_validation(params, X, Y)
        test_run(params, X, Y)
        print()