import pandas as pd
import numpy as np

import os, glob
import argparse
import time
from datetime import datetime

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import f1_score

from pprint import pprint
from utils.misc import label_gen_np, save_pred

parser = argparse.ArgumentParser(description='Atlas Protein')
parser.add_argument('-n', '--name', default='./configs/config.json', required=True, 
                    help="name for the files")
parser.add_argument('--individual', action='store_true', default=False,
                    help='Train classifiers seperately for each label')
args = parser.parse_args()

train_labels_path = f"./data/train.csv"
external_labels_path = f"./data/external/HPAv18RBGY_wodpl.csv"

if not os.path.exists('./stacks'):
    os.makedirs('./stacks')

def read_preds(models):
    train_dfs = []
    test_dfs = []
    for model in models:
        mod_df = pd.concat((pd.read_csv(model), pd.read_csv(model.replace('train', 'external')))).reset_index(drop=True)
        # mod_df.columns = [models[0].split('_')[2] + "_" + cname for cname in mod_df.columns]
        train_dfs.append(mod_df)

        test_dfs.append(pd.read_csv(model.replace('train', 'test')))
    return train_dfs, test_dfs

def fit_features(train_features, train_labels, scoring="f1_macro", n_iter=10, cv=3):
    # Number of trees in random forest
    n_estimators = [int(x) for x in np.linspace(start = 200, stop = 2000, num = 10)]
    # Number of features to consider at every split
    max_features = ['auto', 'sqrt']
    criterion = ['gini', 'entropy']
    # Maximum number of levels in tree
    max_depth = [int(x) for x in np.linspace(10, 100, num = 10)]
    max_depth.append(None)
    # Minimum number of samples required to split a node
    min_samples_split = [2, 5, 10]
    # Minimum number of samples required at each leaf node
    min_samples_leaf = [1, 2, 4]
    # Method of selecting samples for training each tree
    bootstrap = [True, False]

    # Create the random grid
    random_grid = {'n_estimators': n_estimators,
                   'criterion': criterion,
                   'max_features': max_features,
                   'max_depth': max_depth,
                   'min_samples_split': min_samples_split,
                   'min_samples_leaf': min_samples_leaf,
                   'bootstrap': bootstrap}

    # Use the random grid to search for best hyperparameters
    # First create the base model to tune
    rf = RandomForestClassifier()
    # Random search of parameters, using 3 fold cross validation, 
    # search across 100 different combinations, and use all available cores
    rf_random = RandomizedSearchCV(estimator=rf, param_distributions=random_grid,
                                  n_iter = n_iter, scoring=scoring, 
                                  cv = cv, verbose=5, random_state=42, n_jobs=-1)

    # Fit the random search model
    rf_random.fit(train_features, train_labels)

    return rf_random

if __name__ == '__main__':
    models = glob.glob('./preds/*train.csv')
    print("Found {} models".format(str(len(models))))
    train_dfs, test_dfs = read_preds(models)    

    # Extract features and labels
    labels = pd.concat((pd.read_csv(train_labels_path), 
                pd.read_csv(external_labels_path))).reset_index(drop=True)
    labels = labels['Target'].apply(label_gen_np)

    if args.individual:
        print("Doing label-wise stacking")
        pred = np.zeros((len(test_dfs[0]), 28))
        for label_ind in range(28):
            t1 = time.time()
            print("Fitting label ", label_ind)
            features = pd.concat((train_dfs[i][str(label_ind)] \
                            for i in range(len(train_dfs))), axis=1)
            labels = np.stack(labels.values)[:, label_ind]
            test_features = pd.concat((test_dfs[i][str(label_ind)] \
                                for i in range(len(test_dfs))), axis=1)

            features = np.array(features)
            labels = np.array(labels)
            test_features = np.array(test_features)

            rf_random = fit_features(features, labels, "f1", n_iter=10, cv=3)
            pred[:, label_ind] = rf_random.predict(test_features)
            np.save('./stacks/{}_label_{}.npy'.format(args.name, str(label_ind)), pred)
            t2 = time.time()
            print("Fitted. Best score: ", rf_random.best_score_, ". Time taken = ", t2-t1)

    else:
        t1 = time.time()
        print("Doing all labels stacking")
        features = pd.concat((train_dfs[i] for i in range(len(train_dfs))), axis=1)
        labels = np.stack(labels.values)
        test_features = pd.concat((test_dfs[i] for i in range(len(test_dfs))), axis=1)

        features = np.array(features)
        labels = np.array(labels)
        test_features = np.array(test_features)

        rf_random = fit_features(features, labels, "f1_macro", n_iter=2, cv=3)
        pred = rf_random.predict(test_features)
        np.save('./stacks/{}.npy'.format(args.name), pred)
        t2 = time.time()
        print("Fitted. Best score: ", rf_random.best_score_, ". Time taken = ", t2-t1)

    save_pred(pred, th=0.5, SUBM_OUT='./subm/{}.csv'.format(args.name), fill_empty=False)

