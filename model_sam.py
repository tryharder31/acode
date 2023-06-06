"""
1. feature analysis
2. 60+
3. questionnaires

MLP Crossvalidation results:
f1_score 0.703
jaccard_score 0.567

hidden_layer_sizes: 500
alpha: 0.0001

"""
import argparse
import os
import pickle
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split, KFold
from sklearn.cluster import KMeans
from sklearn.inspection import permutation_importance
from sklearn.metrics import classification_report

from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from scipy.spatial.distance import correlation
from sklearn.decomposition import PCA,KernelPCA
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.feature_selection import VarianceThreshold, RFE, SequentialFeatureSelector
from sklearn.linear_model import LinearRegression
from sklearn.metrics import f1_score, jaccard_score,r2_score
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.neural_network import MLPRegressor, MLPClassifier

from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

import matplotlib.pyplot as plt

def qrint(*args, **kwargs):
    print(*args, **kwargs)
    quit()

def drop_if_in(df,cols):
    for col in cols:
        if col in df.columns:
            df = df.drop(col, axis=1)
    return df

def train_dev_test_split(df):

    df=drop_if_in(df,['frameTime','id','key_0'])
    ages = df['age'].to_numpy()
    ages_num = []
    for age in ages:
        if '-' in age:
            age_range = age.split('-')
            ages_num.append(1+int(np.mean(list(map(int,age_range)))))
        else:
            ages_num.append(int(age))
    def collapse_classes(age):
        if age<20:
            return 0
        elif age<60:
            return 1
        else:
            return 2
    ages_num = [collapse_classes(age) for age in ages_num]
    df['age'] = ages_num
    string_to_int = {age: i for i, age in enumerate(set(df['age']))}
    string_to_int.update({i: word for word, i in string_to_int.items()})
    #print(len(string_to_int))

    # save string_to_int to a file:
    #with open('output/string_to_int.pkl','wb') as f:
    #    pickle.dump(string_to_int,f)

    def prepare(df):
        y = df['age'].to_numpy()
        X = df.drop(['age', 'dset'], axis=1)
        return X, y

    #df['age'] = df['age'].apply(lambda x: string_to_int[x])

    train = df[df['dset'] == 'train']
    dev = df[df['dset'] == 'dev']
    test = df[df['dset'] == 'test']


    test_X, test_y = prepare(test)

    train_X, train_y = prepare(train);dev_X, dev_y = prepare(dev)

    return train_X, train_y, dev_X, dev_y#, test_X, test_y


def grid_search(X, y,dev_X,dev_y, out_dir):
    # Define the models
    models = [
        # ('RandomForest', RandomForestClassifier(), {
        #     'n_estimators': [100,300, 400],
        #     'max_depth': [None, 10, 20, 30],
        #     'min_samples_split': [2, 5, 10],
        #     'n_jobs': [-1]
        # }),
        ('GradientBoosting', GradientBoostingClassifier(), {
            'n_estimators': [50, 100, 200],
            'learning_rate': [0.01, 0.1, 1.0],
            'max_depth': [5, 10, 15],
        }),
        ('SVM', SVC(), {
            'C': [0.1, 1, 5, 10],
            'gamma': ['scale', 'auto'],
            'kernel': ['linear', 'rbf'],
        })
    ]

    for name, model, params in models:
        print('Training model: {}'.format(name))
        gs = GridSearchCV(model, params, cv=5,scoring='f1_weighted',verbose=True)
        gs.fit(X, y)

        y_pred = gs.predict(dev_X)
        report = classification_report(dev_y, y_pred)

        with open(out_dir+f"/{name}_report.txt", "w") as f:
            cv_res=gs.cv_results_
            for mean_score, params in sorted(zip(cv_res["mean_test_score"], cv_res["params"]),key=lambda sp:sp[0]):
                print(params, "Mean validation score:", round(mean_score,3),file=f)
            print(file=f)
            f.write(report)



def eval_model(model, train_X, train_y, dev_X, dev_y,metric='f1_score'):
    model.fit(train_X, train_y)
    pred = model.predict(dev_X)
    if metric == 'r2_score':
        return(round(eval(metric)(pred, dev_y),3))
    else:
        return round(eval(metric)(pred, dev_y, average='weighted'),3)

def eval_with_sfs(model_func,train_X,train_y,dev_X,dev_y):

    sfs_support = get_sfs_support(model_func,train_X,train_y)
    # model = model_func().fit(train_X[:,sfs_support],train_y)
    # print(f1_score(model.predict(dev_X[:,sfs_support]), dev_y, average='weighted'))
    return sfs_support

def fit_transform_df(model,X_df):
    model.fit_transform(X_df)
    mask = model.get_support()
    return X_df.loc[:,mask]

def transform_df(model,X_df):
    model.transform(X_df)
    mask = model.get_support()
    return X_df.loc[:,mask]


def scale_df(df):
    return pd.DataFrame(StandardScaler().fit_transform(df), columns=df.columns)

def find_optimal_clusters(data, max_k=5):
    iters = range(2, max_k + 1, 2)

    sse = []
    for k in iters:
        sse.append(KMeans(n_clusters=k, random_state=20).fit(data).inertia_)
        print('Fit {} clusters'.format(k))

    f, ax = plt.subplots(1, 1)
    ax.plot(iters, sse, marker='o')
    ax.set_xlabel('Cluster Centers')
    ax.set_xticks(iters)
    ax.set_xticklabels(iters)
    ax.set_ylabel('SSE')
    ax.set_title('SSE by Cluster Center Plot')
    plt.show()


def classification_model_breakdown(model_name):
    model = eval(model_name)
    print(model_name+':\n--------------')
    print('scaled:')
    eval_model(model, scaled_var_tr_X, train_y, scaled_var_dv_X, dev_y)
    model = eval(model_name)
    print('var scaled:')
    eval_model(model, scaled_var_tr_X, train_y, scaled_var_dv_X, dev_y)
    model = eval(model_name)
    print()
    print('pca1k scaled:')
    model = eval(model_name)
    eval_model(model,pca1k_tr_X,train_y,pca1k_dv_X,dev_y)
    print()
    model = eval(model_name)
    print('pca3k scaled:')
    eval_model(model,scaled_pca3k_tr_X,train_y,scaled_pca3k_dv_X,dev_y)
    print()

def get_top_k_features(clf, K=500):
    importances = clf.feature_importances_
    indices = np.argsort(importances)[::-1]
    return list(indices[:K])


def get_sfs_support(estimator_func, X, y, n_features_to_select='auto'):

    estimator = estimator_func()
    selector = SequentialFeatureSelector(estimator_func(),
                                         scoring='f1_weighted',
                                         n_features_to_select=n_features_to_select,cv=2,
                                         direction='backward')
    selector = selector.fit(X, y)

    # now select the features


    return selector.support_


def do(model_func,X,y, out_dir,n_splits=5,):
    split_indss = list(KFold(n_splits=n_splits, shuffle=True).split(X,y))
    scores = []
    models = []
    for i in range(n_splits):
        split_inds = (split_indss)[i]
        X_train = X[split_inds[0]]
        y_train = y[split_inds[0]]
        X_val = X[split_inds[1]]
        y_val = y[split_inds[1]]
        model = model_func().fit(X_train, y)
        models.append(model)
        pred = model.predict(X_val)
        score = f1_score(pred, y_val, average='weighted')
        scores.append(score)
    i = scores.index(max(scores))
    winner = models[i]
    # is it really nec?

path = 'data/samromur_queries_21.12_featureized_.1.csv'
path = 'data/sample.csv'
path='data/samromur_queries_21.12_featureized_processed.csv'
K=50 # number of top pefrorming features to be selected

# now load into "path" the value of the -p runtime keyword:
parser = argparse.ArgumentParser()
parser.add_argument("-p", nargs=1, type=str, default=path, help="Which path to use for the data.")
parser.add_argument("-k", nargs=1, type=int, default=K, help="Which path to use for the data.")
path = parser.parse_args().p
K = parser.parse_args().k
var_thresh = .3

out_dir = 'output/'+path.split('/')[1]
print(path)
print(out_dir)

if not os.path.exists(out_dir):
    os.makedirs(out_dir)

km10 = 'KNeighborsClassifier(n_neighbors=10)'
mlparameters = {'hidden_layer_sizes': ((100,),(100,100),(500),(500,500)),
                'alpha' : (0.0001,0.01,0.1,.5)}
mlp_classifiers = 'GridSearchCV(MLPClassifier(),mlparameters,verbose=1)'


var_selector = VarianceThreshold(threshold=.3)

if path == 'data/sample.csv':
    pca3k = PCA(n_components=30)
    pca1k = PCA(n_components=10)
else:
    pca3k = PCA(n_components=3000)
    pca1k = PCA(n_components=100)

if 1:
    df = pd.read_csv(path,index_col=0)
    feats = drop_if_in(df,['frameTime','id','key_0','age','dset'])
    train_X,train_y,dev_X,dev_y = train_dev_test_split(df)
else:
    from sklearn.datasets import load_diabetes
    # load the boston dataset
    X,y = load_diabetes(return_X_y=True)
    train_X, dev_X, train_y, dev_y = train_test_split(X, y, test_size=0.2, random_state=42)

var_selector = VarianceThreshold(threshold=var_thresh)
var_tr_X = fit_transform_df(var_selector,train_X)
var_dv_X = transform_df(var_selector,dev_X)

scaled_var_tr_X = scale_df(var_tr_X)
scaled_var_dv_X = scale_df(var_dv_X)

pca3k_tr_X = pca3k.fit_transform(train_X)
pca3k_dv_X = pca3k.transform(dev_X)
pca1k_tr_X = pca1k.fit_transform(train_X)
pca1k_dv_X = pca1k.transform(dev_X)


model_func = lambda: RandomForestClassifier(n_jobs=-1, n_estimators=100, min_samples_split=2)

model1 = model_func()
score1 = eval_model(model1,scaled_var_tr_X,train_y,scaled_var_dv_X,dev_y)


top_k_features_inds = get_top_k_features(model1, K)
top_k_features = scaled_var_tr_X.columns[top_k_features_inds]
topp_scaled_var_tr_X = scaled_var_tr_X.iloc[:, top_k_features_inds]
topp_scaled_var_dv_X = scaled_var_dv_X.iloc[:, top_k_features_inds]
print(topp_scaled_var_tr_X.shape)
score2 = eval_model(model_func(), topp_scaled_var_tr_X, train_y, topp_scaled_var_dv_X, dev_y)
print(score1)
print(score2)

support = (eval_with_sfs(model_func, topp_scaled_var_tr_X, train_y, topp_scaled_var_dv_X, dev_y))
sfs_topp_scaled_var_tr_X = topp_scaled_var_tr_X.loc[:,support]
sfs_topp_scaled_var_dv_X = topp_scaled_var_dv_X.loc[:,support]
score3 = eval_model(model_func(), sfs_topp_scaled_var_tr_X, train_y, sfs_topp_scaled_var_dv_X, dev_y)
print(score3)

with open(out_dir+'/top_features.txt','w') as f:
    for feat in sfs_topp_scaled_var_tr_X.columns:
        f.write(str(feat)+'\n')
