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
from utils.misc import label_gen_np, save_pred, f1_loss_keras, f1_keras

parser = argparse.ArgumentParser(description='Atlas Protein')
parser.add_argument('-n', '--name', default='stacking', required=True, 
                    help="name for the files")
parser.add_argument('-i', '--individual', action='store_true', default=False,
                    help='Train classifiers seperately for each label')
parser.add_argument('-c', '--classifier', default="randomforest",
                    help='which classifier to use')
args = parser.parse_args()

train_labels_path = f"./data/train.csv"
external_labels_path = f"./data/external/HPAv18RBGY_wodpl.csv"

if not os.path.exists('./stacks'):
    os.makedirs('./stacks')

def read_preds(pred_files):
    train_dfs = []
    test_dfs = []
    for model in pred_files:
        mod_df = pd.concat((pd.read_csv(model), pd.read_csv(model.replace('train', 'external')))).reset_index(drop=True)
        # mod_df.columns = [pred_files[0].split('_')[2] + "_" + cname for cname in mod_df.columns]
        train_dfs.append(mod_df)

        test_dfs.append(pd.read_csv(model.replace('train', 'test')))
    return train_dfs, test_dfs

def fit_random_forest(train_features, train_labels, scoring="f1_macro", n_iter=10, cv=3):
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

def fit_neural_network(featuers, labels):
    from keras import models
    from keras import layers
    from keras.optimizers import Adam
    from keras.callbacks import ModelCheckpoint, LearningRateScheduler, EarlyStopping, ReduceLROnPlateau

    model = models.Sequential()
    # Input - Layer
    model.add(layers.Dense(featuers.shape[1], activation = "relu", 
                    input_shape=(featuers.shape[1], )))
    # Hidden - Layers
    model.add(layers.Dropout(0.3, noise_shape=None, seed=None))
    model.add(layers.Dense(50, activation = "relu"))
    model.add(layers.Dropout(0.2, noise_shape=None, seed=None))
    model.add(layers.Dense(50, activation = "relu"))
    # Output- Layer
    model.add(layers.Dense(labels.shape[1], activation = "sigmoid"))
    print(model.summary())

    model.compile(
     optimizer = Adam(1e-03),
     loss = "binary_crossentropy",
     metrics = ["accuracy", f1_keras]
    )

    checkpoint = ModelCheckpoint('./stacks/nn.model', monitor='val_loss', verbose=1, 
            save_best_only=True, save_weights_only=False, mode='min', period=1)
    reduceLROnPlato = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, 
                verbose=1, mode='min')

    X_train, X_test, y_train, y_test = train_test_split(featuers, labels, 
                                            test_size=0.2, random_state=42)

    hist = model.fit(x=featuers, y=labels, batch_size=16, epochs=100, verbose=1, 
        callbacks=[checkpoint, reduceLROnPlato], validation_split=0.2, 
        shuffle=True, class_weight=None, sample_weight=None, initial_epoch=0, 
        validation_steps=None)

    return model

if __name__ == '__main__':
    pred_files = glob.glob('./preds/*train.csv')
    print("Found {} prediction files".format(str(len(pred_files))))
    train_dfs, test_dfs = read_preds(pred_files)    

    # Extract features and labels
    labels = pd.concat((pd.read_csv(train_labels_path), 
                pd.read_csv(external_labels_path))).reset_index(drop=True)
    labels = labels['Target'].apply(label_gen_np)
    labels = np.stack(labels.values)

    if args.individual:
        print("Doing label-wise stacking")
        pred = np.zeros((len(test_dfs[0]), 28))
        for label_ind in range(28):
            lsave = './stacks/{}_label_{}.npy'.format(args.name, str(label_ind))
            if os.path.isfile(lsave):
                print("Found this label already trained")
                pred = np.load(lsave)
                continue
            t1 = time.time()
            print("Fitting label ", label_ind)
            features = pd.concat((train_dfs[i][str(label_ind)] \
                            for i in range(len(train_dfs))), axis=1)
            test_features = pd.concat((test_dfs[i][str(label_ind)] \
                                for i in range(len(test_dfs))), axis=1)

            features = np.array(features)
            test_features = np.array(test_features)

            if args.classifier in ["rf", "randomforest"]:
                rf_random = fit_random_forest(features, labels[:, label_ind], "f1", 
                                            n_iter=12, cv=3)
                pred[:, label_ind] = rf_random.predict(test_features)
                bs = rf_random.best_score_

            if args.classifier in ["nn", "neuralnetwork"]:
                model = fit_neural_network(features, labels)
                pred[:, label_ind] = model.predict(test_features, batch_size=16)
            
            np.save(lsave, pred)
            t2 = time.time()
            print("Fitted. Best score: ", bs, ". Time taken = ", t2-t1)

    else:
        t1 = time.time()
        print("Doing all labels stacking")
        features = pd.concat((train_dfs[i] for i in range(len(train_dfs))), axis=1)
        test_features = pd.concat((test_dfs[i] for i in range(len(test_dfs))), axis=1)

        features = np.array(features)
        test_features = np.array(test_features)

        if args.classifier == "randomforest":
            rf_random = fit_features(features, labels, "f1_macro", n_iter=2, cv=3)
            pred = rf_random.predict(test_features)
            bs = rf_random.best_score_
        
        if args.classifier in ["nn", "neuralnetwork"]:
                model = fit_neural_network(features, labels)
                pred = model.predict(test_features, batch_size=16)

        np.save('./stacks/{}.npy'.format(args.name), pred)
        t2 = time.time()
        print("Fitted. Best score: ", bs, ". Time taken = ", t2-t1)

    save_pred(pred, th=0.5, SUBM_OUT='./subm/{}.csv'.format(args.name), fill_empty=False)

