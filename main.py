import sys
import pandas as pd
import numpy as np
import multimodal_models
import argparse
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, f1_score
from sklearn.svm import SVC
from sklearn.pipeline import make_pipeline
from sklearn.impute import SimpleImputer
from sklearn.ensemble import VotingClassifier, GradientBoostingClassifier, AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
from xgboost import XGBClassifier
from imblearn.ensemble import RUSBoostClassifier


def load_data(dataset_path, task):

    dataframe = pd.read_csv(dataset_path)
    dataframe = dataframe[dataframe["tweet_id"] != 1321141824300306433]
    if task == 'AS':
        y_true = np.array(dataframe.stance)
    else:
        y_true = np.array(dataframe.persuasiveness)
    
    return dataframe, y_true


def handle_AS_approach_1 (GC_train_dataset_path, A_train_dataset_path, GC_dev_dataset_path, A_dev_dataset_path, GC_test_dataset_path, A_test_dataset_path, train_dev_images, test_images):
    
    GC_df_train, GC_y_train = load_data (GC_train_dataset_path, 'AS')
    A_df_train, A_y_train = load_data (A_train_dataset_path, 'AS')

    GC_df_dev, GC_y_dev = load_data (GC_dev_dataset_path, 'AS')
    A_df_dev, A_y_dev = load_data (A_dev_dataset_path, 'AS')
    
    GC_df_test, GC_y_test = load_data (GC_test_dataset_path, 'AS')
    A_df_test, A_y_test = load_data (A_test_dataset_path, 'AS')

    GC_X_train = multimodal_models.clip32(GC_df_train, train_dev_images)
    A_X_train = multimodal_models.clip32(A_df_train, train_dev_images)
    GC_X_dev = multimodal_models.clip32(GC_df_dev, train_dev_images)
    A_X_dev = multimodal_models.clip32(A_df_dev, train_dev_images)
    GC_X_test = multimodal_models.clip32(GC_df_test, test_images)
    A_X_test = multimodal_models.clip32(A_df_test, test_images)
    
    base_estimator = DecisionTreeClassifier(max_depth=2)
    GC_classifier = AdaBoostClassifier( base_estimator=base_estimator,
    n_estimators=150,
    learning_rate=0.3,
    algorithm='SAMME')
    GC_classifier.fit(GC_X_train, GC_y_train)
    GC_y_dev_pred = GC_classifier.predict(GC_X_dev)
    GC_y_test_pred = GC_classifier.predict(GC_X_test)
    
    A_classifier = AdaBoostClassifier(base_estimator=base_estimator,
    n_estimators=150,
    learning_rate=0.2,
    algorithm='SAMME')
    A_classifier.fit(A_X_train, A_y_train)
    A_y_dev_pred = A_classifier.predict(A_X_dev)
    A_y_test_pred = A_classifier.predict(A_X_test)
    y_dev_true = np.concatenate((A_y_dev, GC_y_dev))
    y_dev_pred = np.concatenate((A_y_dev_pred, GC_y_dev_pred))
    y_test_true = np.concatenate((A_y_test, GC_y_test))
    y_test_pred = np.concatenate((A_y_test_pred, GC_y_test_pred))

    pos_label = 'support'
    dev_f1 = round(f1_score(y_dev_true, y_dev_pred, pos_label=pos_label),4)
    test_f1 = round(f1_score(y_test_true, y_test_pred, pos_label=pos_label),4)
    print ("dev_score: ", dev_f1)
    print ("test_score: ", test_f1)
    
    
def handle_AS_approach_2 (GC_train_dataset_path, A_train_dataset_path, GC_dev_dataset_path, A_dev_dataset_path, GC_test_dataset_path, A_test_dataset_path, train_dev_images, test_images):
    
    GC_df_train, GC_y_train = load_data (GC_train_dataset_path, 'AS')
    A_df_train, A_y_train = load_data (A_train_dataset_path, 'AS')

    GC_df_dev, GC_y_dev = load_data (GC_dev_dataset_path, 'AS')
    A_df_dev, A_y_dev = load_data (A_dev_dataset_path, 'AS')
    
    GC_df_test, GC_y_test = load_data (GC_test_dataset_path, 'AS')
    A_df_test, A_y_test = load_data (A_test_dataset_path, 'AS')

    GC_X_train = multimodal_models.clip32(GC_df_train, train_dev_images)
    A_X_train = multimodal_models.clip32(A_df_train, train_dev_images)
    GC_X_dev = multimodal_models.clip32(GC_df_dev, train_dev_images)
    A_X_dev = multimodal_models.clip32(A_df_dev, train_dev_images)
    GC_X_test = multimodal_models.clip32(GC_df_test, test_images)
    A_X_test = multimodal_models.clip32(A_df_test, test_images)
   
    label_encoder = LabelEncoder()
    GC_y_train = label_encoder.fit_transform(GC_y_train)
    classifier1 = XGBClassifier(
    max_depth=3,
    learning_rate=0.1,
    subsample=0.8,
    colsample_bytree=0.8,
    reg_alpha=0.1,
    reg_lambda=0.1
    )
    classifier2 = GradientBoostingClassifier(
    learning_rate=0.2,
    n_estimators=80,
    max_depth=3,
    subsample=1.0,
    random_state=42
    )
    GC_classifier = VotingClassifier(estimators=[('xgb', classifier1), ('gb', classifier2)], voting='soft', weights=[3, 1])
    GC_classifier.fit(GC_X_train, GC_y_train)
    GC_y_dev_pred = GC_classifier.predict(GC_X_dev)
    GC_y_dev_pred = label_encoder.inverse_transform(GC_y_dev_pred)
    GC_y_test_pred = GC_classifier.predict(GC_X_test)
    GC_y_test_pred = label_encoder.inverse_transform(GC_y_test_pred)
    
    base_estimator = DecisionTreeClassifier(max_depth=2)
    A_classifier = AdaBoostClassifier( base_estimator=base_estimator,
    n_estimators=150,
    learning_rate=0.2,
    algorithm='SAMME')
    A_classifier.fit(A_X_train, A_y_train)
    A_y_dev_pred = A_classifier.predict(A_X_dev)
    A_y_test_pred = A_classifier.predict(A_X_test)
    
    y_dev_true = np.concatenate((A_y_dev, GC_y_dev))
    y_dev_pred = np.concatenate((A_y_dev_pred, GC_y_dev_pred))
    y_test_true = np.concatenate((A_y_test, GC_y_test))
    y_test_pred = np.concatenate((A_y_test_pred, GC_y_test_pred))

    pos_label = 'support'
    dev_f1 = round(f1_score(y_dev_true, y_dev_pred, pos_label=pos_label),4)
    test_f1 = round(f1_score(y_test_true, y_test_pred, pos_label=pos_label),4)
    print ("dev_score: ", dev_f1)
    print ("test_score: ", test_f1)


def handle_AS_approach_3(GC_train_dataset_path, A_train_dataset_path, GC_dev_dataset_path, A_dev_dataset_path, GC_test_dataset_path, A_test_dataset_path, train_dev_images, test_images):

    GC_df_train, GC_y_train = load_data (GC_train_dataset_path, 'AS')
    A_df_train, A_y_train = load_data (A_train_dataset_path, 'AS')

    GC_df_dev, GC_y_dev = load_data (GC_dev_dataset_path, 'AS')
    A_df_dev, A_y_dev = load_data (A_dev_dataset_path, 'AS')
    
    GC_df_test, GC_y_test = load_data (GC_test_dataset_path, 'AS')
    A_df_test, A_y_test = load_data (A_test_dataset_path, 'AS')

    GC_X_train = multimodal_models.clip32(GC_df_train, train_dev_images)
    A_X_train = multimodal_models.clip32(A_df_train, train_dev_images)
    GC_X_dev = multimodal_models.clip32(GC_df_dev, train_dev_images)
    A_X_dev = multimodal_models.clip32(A_df_dev, train_dev_images)
    GC_X_test = multimodal_models.clip32(GC_df_test, test_images)
    A_X_test = multimodal_models.clip32(A_df_test, test_images)
    
    GC_classifier = RUSBoostClassifier(n_estimators=150,random_state=42, learning_rate=0.18,sampling_strategy='not majority')
    GC_classifier.fit(GC_X_train, GC_y_train)
    GC_y_dev_pred = GC_classifier.predict(GC_X_dev)
    GC_y_test_pred = GC_classifier.predict(GC_X_test)
    
    base_estimator = DecisionTreeClassifier(max_depth=2)
    A_classifier = AdaBoostClassifier( base_estimator=base_estimator,
    n_estimators=150,
    learning_rate=0.2,
    algorithm='SAMME')
    A_classifier.fit(A_X_train, A_y_train)
    A_y_dev_pred = A_classifier.predict(A_X_dev)
    A_y_test_pred = A_classifier.predict(A_X_test)

    y_dev_true = np.concatenate((A_y_dev, GC_y_dev))
    y_dev_pred = np.concatenate((A_y_dev_pred, GC_y_dev_pred))
    y_test_true = np.concatenate((A_y_test, GC_y_test))
    y_test_pred = np.concatenate((A_y_test_pred, GC_y_test_pred))

    pos_label = 'support'
    dev_f1 = round(f1_score(y_dev_true, y_dev_pred, pos_label=pos_label),4)
    test_f1 = round(f1_score(y_test_true, y_test_pred, pos_label=pos_label),4) 
    print ("dev_score: ", dev_f1)
    print ("test_score: ", test_f1)
    

def handle_AS_approach_4(GC_train_dataset_path, A_train_dataset_path, GC_dev_dataset_path, A_dev_dataset_path, GC_test_dataset_path, A_test_dataset_path, train_dev_images, test_images):
    
    GC_df_train, GC_y_train = load_data (GC_train_dataset_path, 'AS')
    A_df_train, A_y_train = load_data (A_train_dataset_path, 'AS')
    df_train = pd.concat([A_df_train, GC_df_train], axis=0)
    y_train = np.concatenate((A_y_train, GC_y_train))

    GC_df_dev, GC_y_dev = load_data (GC_dev_dataset_path, 'AS')
    A_df_dev, A_y_dev = load_data (A_dev_dataset_path, 'AS')
    df_dev = pd.concat([A_df_dev, GC_df_dev], axis=0)
    y_dev_true = np.concatenate((A_y_dev, GC_y_dev))
    
    GC_df_test, GC_y_test = load_data (GC_test_dataset_path, 'AS')
    A_df_test, A_y_test = load_data (A_test_dataset_path, 'AS')
    df_test = pd.concat([A_df_test, GC_df_test], axis=0)
    y_test_true = np.concatenate((A_y_test, GC_y_test))
    
    X_train = multimodal_models.clip32(df_train, train_dev_images)
    X_dev = multimodal_models.clip32(df_dev, train_dev_images)
    X_test = multimodal_models.clip32(df_test, test_images)
    
    label_encoder = LabelEncoder()
    y_train = label_encoder.fit_transform(y_train)
    classifier1 = XGBClassifier(
    max_depth=2,
    learning_rate=0.3,
    subsample=0.8,
    colsample_bytree=0.8,
    reg_alpha=0.1,
    reg_lambda=0.12
    )
    classifier2 = GradientBoostingClassifier(
    learning_rate=0.4,
    n_estimators=80,
    max_depth=3,
    subsample=1.0,
    random_state=42
    )
    classifier = VotingClassifier(estimators=[('xgb', classifier1), ('gb', classifier2)], voting='soft', weights =[5,2])
    classifier.fit(X_train, y_train)
 
    y_dev_pred = classifier.predict(X_dev)
    y_dev_pred = label_encoder.inverse_transform(y_dev_pred)
    y_test_pred = classifier.predict(X_test)
    y_test_pred = label_encoder.inverse_transform(y_test_pred)

    pos_label = 'support'
    dev_f1 = round(f1_score(y_dev_true, y_dev_pred, pos_label=pos_label),4)
    test_f1 = round(f1_score(y_test_true, y_test_pred, pos_label=pos_label),4) 
    print ("dev_score: ", dev_f1)
    print ("test_score: ", test_f1)


def handle_AS_approach_5(GC_train_dataset_path, A_train_dataset_path, GC_dev_dataset_path, A_dev_dataset_path, GC_test_dataset_path, A_test_dataset_path, train_dev_images, test_images):
    
    GC_df_train, GC_y_train = load_data (GC_train_dataset_path, 'AS')
    A_df_train, A_y_train = load_data (A_train_dataset_path, 'AS')
    df_train = pd.concat([A_df_train, GC_df_train], axis=0)
    y_train = np.concatenate((A_y_train, GC_y_train))

    GC_df_dev, GC_y_dev = load_data (GC_dev_dataset_path, 'AS')
    A_df_dev, A_y_dev = load_data (A_dev_dataset_path, 'AS')
    df_dev = pd.concat([A_df_dev, GC_df_dev], axis=0)
    y_dev_true = np.concatenate((A_y_dev, GC_y_dev))
    
    GC_df_test, GC_y_test = load_data (GC_test_dataset_path, 'AS')
    A_df_test, A_y_test = load_data (A_test_dataset_path, 'AS')
    df_test = pd.concat([A_df_test, GC_df_test], axis=0)
    y_test_true = np.concatenate((A_y_test, GC_y_test))
    
    X_train = multimodal_models.clip32(df_train, train_dev_images)
    X_dev = multimodal_models.clip32(df_dev, train_dev_images)
    X_test = multimodal_models.clip32(df_test, test_images)
 
    classifier = SVC(kernel='poly', degree=2, C=1.0, coef0=0.6)
    classifier.fit(X_train, y_train)
 
    y_dev_pred = classifier.predict(X_dev)
    y_test_pred = classifier.predict(X_test)
    
    pos_label = 'support'
    dev_f1 = round(f1_score(y_dev_true, y_dev_pred, pos_label=pos_label),4)
    test_f1 = round(f1_score(y_test_true, y_test_pred, pos_label=pos_label),4) 
    print ("dev_score: ", dev_f1)
    print ("test_score: ", test_f1)


def handle_AS_approach_6(GC_train_dataset_path, A_train_dataset_path, GC_dev_dataset_path, A_dev_dataset_path, GC_test_dataset_path, A_test_dataset_path, train_dev_images, test_images):

    GC_df_train, GC_y_train = load_data (GC_train_dataset_path, 'AS')
    A_df_train, A_y_train = load_data (A_train_dataset_path, 'AS')

    GC_df_dev, GC_y_dev = load_data (GC_dev_dataset_path, 'AS')
    A_df_dev, A_y_dev = load_data (A_dev_dataset_path, 'AS')
    
    GC_df_test, GC_y_test = load_data (GC_test_dataset_path, 'AS')
    A_df_test, A_y_test = load_data (A_test_dataset_path, 'AS')

    GC_X_train = multimodal_models.clip32(GC_df_train, train_dev_images)
    A_X_train = multimodal_models.clip32(A_df_train, train_dev_images)
    GC_X_dev = multimodal_models.clip32(GC_df_dev, train_dev_images)
    A_X_dev = multimodal_models.clip32(A_df_dev, train_dev_images)
    GC_X_test = multimodal_models.clip32(GC_df_test, test_images)
    A_X_test = multimodal_models.clip32(A_df_test, test_images)
    
    GC_classifier = MLPClassifier( hidden_layer_sizes=(1000,100),learning_rate_init=0.0005, random_state=42)
    GC_classifier.fit(GC_X_train, GC_y_train)
    GC_y_dev_pred = GC_classifier.predict(GC_X_dev)
    GC_y_test_pred = GC_classifier.predict(GC_X_test)
    
    base_estimator = DecisionTreeClassifier(max_depth=2)
    A_classifier = AdaBoostClassifier( base_estimator=base_estimator,
    n_estimators=150,
    learning_rate=0.2,
    algorithm='SAMME')
    A_classifier.fit(A_X_train, A_y_train)
    A_y_dev_pred = A_classifier.predict(A_X_dev)
    A_y_test_pred = A_classifier.predict(A_X_test)

    y_dev_true = np.concatenate((A_y_dev, GC_y_dev))
    y_dev_pred = np.concatenate((A_y_dev_pred, GC_y_dev_pred))
    y_test_true = np.concatenate((A_y_test, GC_y_test))
    y_test_pred = np.concatenate((A_y_test_pred, GC_y_test_pred))

    pos_label = 'support'
    dev_f1 = round(f1_score(y_dev_true, y_dev_pred, pos_label=pos_label),4)
    test_f1 = round(f1_score(y_test_true, y_test_pred, pos_label=pos_label),4) 
    print ("dev_score: ", dev_f1)
    print ("test_score: ", test_f1)


def handle_IP_approach_1(GC_train_dataset_path, A_train_dataset_path, GC_dev_dataset_path, A_dev_dataset_path, GC_test_dataset_path, A_test_dataset_path, train_dev_images, test_images):

    GC_df_train, GC_y_train = load_data (GC_train_dataset_path, 'IP')
    A_df_train, A_y_train = load_data (A_train_dataset_path, 'IP')
    df_train = pd.concat([A_df_train, GC_df_train], axis=0)
    y_train = np.concatenate((A_y_train, GC_y_train))

    GC_df_dev, GC_y_dev = load_data (GC_dev_dataset_path, 'IP')
    A_df_dev, A_y_dev = load_data (A_dev_dataset_path, 'IP')
    df_dev = pd.concat([A_df_dev, GC_df_dev], axis=0)
    y_dev_true = np.concatenate((A_y_dev, GC_y_dev))
    
    GC_df_test, GC_y_test = load_data (GC_test_dataset_path, 'IP')
    A_df_test, A_y_test = load_data (A_test_dataset_path, 'IP')
    df_test = pd.concat([A_df_test, GC_df_test], axis=0)
    y_test_true = np.concatenate((A_y_test, GC_y_test))
    
    X_train = multimodal_models.clip32(df_train, train_dev_images)
    X_dev = multimodal_models.clip32(df_dev, train_dev_images)
    X_test = multimodal_models.clip32(df_test, test_images)
 
    classifier = SVC(kernel='poly', degree=2, C=1.0, coef0=0.02,  shrinking=False, probability=True)
    classifier.fit(X_train, y_train)
 
    y_dev_pred = classifier.predict(X_dev)
    y_test_pred = classifier.predict(X_test)
    
    pos_label = 'yes'
    dev_f1 = round(f1_score(y_dev_true, y_dev_pred, pos_label=pos_label),4)
    test_f1 = round(f1_score(y_test_true, y_test_pred, pos_label=pos_label),4) 
    print ("dev_score: ", dev_f1)
    print ("test_score: ", test_f1)


def handle_IP_approach_2(GC_train_dataset_path, A_train_dataset_path, GC_dev_dataset_path, A_dev_dataset_path, GC_test_dataset_path, A_test_dataset_path, train_dev_images, test_images):

    GC_df_train, GC_y_train = load_data (GC_train_dataset_path, 'IP')
    A_df_train, A_y_train = load_data (A_train_dataset_path, 'IP')

    GC_df_dev, GC_y_dev = load_data (GC_dev_dataset_path, 'IP')
    A_df_dev, A_y_dev = load_data (A_dev_dataset_path, 'IP')
    
    GC_df_test, GC_y_test = load_data (GC_test_dataset_path, 'IP')
    A_df_test, A_y_test = load_data (A_test_dataset_path, 'IP')

    GC_X_train = multimodal_models.clip32(GC_df_train, train_dev_images)
    A_X_train = multimodal_models.clip32(A_df_train, train_dev_images)
    GC_X_dev = multimodal_models.clip32(GC_df_dev, train_dev_images)
    A_X_dev = multimodal_models.clip32(A_df_dev, train_dev_images)
    GC_X_test = multimodal_models.clip32(GC_df_test, test_images)
    A_X_test = multimodal_models.clip32(A_df_test, test_images)
    
    GC_classifier = SGDClassifier(
    alpha=0.05,
    learning_rate="optimal",
    random_state=42
    )
    GC_classifier.fit(GC_X_train, GC_y_train)
    GC_y_dev_pred = GC_classifier.predict(GC_X_dev)
    GC_y_test_pred = GC_classifier.predict(GC_X_test)
    
    A_classifier = SGDClassifier(
    alpha=0.0344,
    learning_rate="optimal",
    random_state=42
    )
    A_classifier.fit(A_X_train, A_y_train)
    A_y_dev_pred = A_classifier.predict(A_X_dev)
    A_y_test_pred = A_classifier.predict(A_X_test)

    y_dev_true = np.concatenate((A_y_dev, GC_y_dev))
    y_dev_pred = np.concatenate((A_y_dev_pred, GC_y_dev_pred))
    y_test_true = np.concatenate((A_y_test, GC_y_test))
    y_test_pred = np.concatenate((A_y_test_pred, GC_y_test_pred))

    pos_label = 'yes'
    dev_f1 = round(f1_score(y_dev_true, y_dev_pred, pos_label=pos_label),4)
    test_f1 = round(f1_score(y_test_true, y_test_pred, pos_label=pos_label),4)
    print ("dev_score: ", dev_f1)
    print ("test_score: ", test_f1)


def handle_IP_approach_3(GC_train_dataset_path, A_train_dataset_path, GC_dev_dataset_path, A_dev_dataset_path, GC_test_dataset_path, A_test_dataset_path, train_dev_images, test_images):

    GC_df_train, GC_y_train = load_data (GC_train_dataset_path, 'IP')
    A_df_train, A_y_train = load_data (A_train_dataset_path, 'IP')
    df_train = pd.concat([A_df_train, GC_df_train], axis=0)
    y_train = np.concatenate((A_y_train, GC_y_train))

    GC_df_dev, GC_y_dev = load_data (GC_dev_dataset_path, 'IP')
    A_df_dev, A_y_dev = load_data (A_dev_dataset_path, 'IP')
    df_dev = pd.concat([A_df_dev, GC_df_dev], axis=0)
    y_dev_true = np.concatenate((A_y_dev, GC_y_dev))
    
    GC_df_test, GC_y_test = load_data (GC_test_dataset_path, 'IP')
    A_df_test, A_y_test = load_data (A_test_dataset_path, 'IP')
    df_test = pd.concat([A_df_test, GC_df_test], axis=0)
    y_test_true = np.concatenate((A_y_test, GC_y_test))
    
    X_train = multimodal_models.clip14(df_train, train_dev_images)
    X_dev = multimodal_models.clip14(df_dev, train_dev_images)
    X_test = multimodal_models.clip14(df_test, test_images)
 
    classifier = SVC(kernel='poly', random_state=42, coef0=0.17)
    classifier.fit(X_train, y_train)
 
    y_dev_pred = classifier.predict(X_dev)
    y_test_pred = classifier.predict(X_test)
    
    pos_label = 'yes'
    dev_f1 = round(f1_score(y_dev_true, y_dev_pred, pos_label=pos_label),4)
    test_f1 = round(f1_score(y_test_true, y_test_pred, pos_label=pos_label),4) 
    print ("dev_score: ", dev_f1)
    print ("test_score: ", test_f1)


def handle_IP_approach_4(GC_train_dataset_path, A_train_dataset_path, GC_dev_dataset_path, A_dev_dataset_path, GC_test_dataset_path, A_test_dataset_path, train_dev_images, test_images):

    GC_df_train, GC_y_train = load_data (GC_train_dataset_path, 'IP')
    A_df_train, A_y_train = load_data (A_train_dataset_path, 'IP')

    GC_df_dev, GC_y_dev = load_data (GC_dev_dataset_path, 'IP')
    A_df_dev, A_y_dev = load_data (A_dev_dataset_path, 'IP')
    
    GC_df_test, GC_y_test = load_data (GC_test_dataset_path, 'IP')
    A_df_test, A_y_test = load_data (A_test_dataset_path, 'IP')

    GC_X_train = multimodal_models.convnext_small_rel(GC_df_train, train_dev_images)
    A_X_train = multimodal_models.clip32(A_df_train, train_dev_images)
    GC_X_dev = multimodal_models.convnext_small_rel(GC_df_dev, train_dev_images)
    A_X_dev = multimodal_models.clip32(A_df_dev, train_dev_images)
    GC_X_test = multimodal_models.convnext_small_rel(GC_df_test, test_images)
    A_X_test = multimodal_models.clip32(A_df_test, test_images)
    
    GC_classifier = LogisticRegression ()
    GC_classifier.fit(GC_X_train, GC_y_train)
    GC_y_dev_pred = GC_classifier.predict(GC_X_dev)
    GC_y_test_pred = GC_classifier.predict(GC_X_test)
    
    A_classifier = SGDClassifier(
    alpha=0.0344,
    learning_rate="optimal",
    random_state=42
    )
    A_classifier.fit(A_X_train, A_y_train)
    A_y_dev_pred = A_classifier.predict(A_X_dev)
    A_y_test_pred = A_classifier.predict(A_X_test)

    y_dev_true = np.concatenate((A_y_dev, GC_y_dev))
    y_dev_pred = np.concatenate((A_y_dev_pred, GC_y_dev_pred))
    y_test_true = np.concatenate((A_y_test, GC_y_test))
    y_test_pred = np.concatenate((A_y_test_pred, GC_y_test_pred))

    pos_label = 'yes'
    dev_f1 = round(f1_score(y_dev_true, y_dev_pred, pos_label=pos_label),4)
    test_f1 = round(f1_score(y_test_true, y_test_pred, pos_label=pos_label),4) 
    print ("dev_score: ", dev_f1)
    print ("test_score: ", test_f1)


def handle_IP_approach_5(GC_train_dataset_path, A_train_dataset_path, GC_dev_dataset_path, A_dev_dataset_path, GC_test_dataset_path, A_test_dataset_path, train_dev_images, test_images):

    GC_df_train, GC_y_train = load_data (GC_train_dataset_path, 'IP')
    A_df_train, A_y_train = load_data (A_train_dataset_path, 'IP')

    GC_df_dev, GC_y_dev = load_data (GC_dev_dataset_path, 'IP')
    A_df_dev, A_y_dev = load_data (A_dev_dataset_path, 'IP')
    
    GC_df_test, GC_y_test = load_data (GC_test_dataset_path, 'IP')
    A_df_test, A_y_test = load_data (A_test_dataset_path, 'IP')

    GC_X_train = multimodal_models.convnext_small_rel(GC_df_train, train_dev_images)
    A_X_train = multimodal_models.clip32(A_df_train, train_dev_images)
    GC_X_dev = multimodal_models.convnext_small_rel(GC_df_dev, train_dev_images)
    A_X_dev = multimodal_models.clip32(A_df_dev, train_dev_images)
    GC_X_test = multimodal_models.convnext_small_rel(GC_df_test, test_images)
    A_X_test = multimodal_models.clip32(A_df_test, test_images)

    GC_classifier = LogisticRegression ()
    GC_classifier.fit(GC_X_train, GC_y_train)
    GC_y_dev_pred = GC_classifier.predict(GC_X_dev)
    GC_y_test_pred = GC_classifier.predict(GC_X_test)
    
    A_classifier = SVC(kernel='poly', degree=2, C=1.0, coef0=0.25)
    A_classifier.fit(A_X_train, A_y_train)
    A_y_dev_pred = A_classifier.predict(A_X_dev)
    A_y_test_pred = A_classifier.predict(A_X_test)

    y_dev_true = np.concatenate((A_y_dev, GC_y_dev))
    y_dev_pred = np.concatenate((A_y_dev_pred, GC_y_dev_pred))
    y_test_true = np.concatenate((A_y_test, GC_y_test))
    y_test_pred = np.concatenate((A_y_test_pred, GC_y_test_pred))

    pos_label = 'yes'
    dev_f1 = round(f1_score(y_dev_true, y_dev_pred, pos_label=pos_label),4)
    test_f1 = round(f1_score(y_test_true, y_test_pred, pos_label=pos_label),4) 
    print ("dev_score: ", dev_f1)
    print ("test_score: ", test_f1)


def handle_IP_approach_6(GC_train_dataset_path, A_train_dataset_path, GC_dev_dataset_path, A_dev_dataset_path, GC_test_dataset_path, A_test_dataset_path, train_dev_images, test_images):

    GC_df_train, GC_y_train = load_data (GC_train_dataset_path, 'IP')
    A_df_train, A_y_train = load_data (A_train_dataset_path, 'IP')

    GC_df_dev, GC_y_dev = load_data (GC_dev_dataset_path, 'IP')
    A_df_dev, A_y_dev = load_data (A_dev_dataset_path, 'IP')
    
    GC_df_test, GC_y_test = load_data (GC_test_dataset_path, 'IP')
    A_df_test, A_y_test = load_data (A_test_dataset_path, 'IP')

    GC_X_train = multimodal_models.convnext_small_rel(GC_df_train, train_dev_images)
    A_X_train = multimodal_models.swin_v2_s_camembert_base(A_df_train, train_dev_images)
    GC_X_dev = multimodal_models.convnext_small_rel(GC_df_dev, train_dev_images)
    A_X_dev = multimodal_models.swin_v2_s_camembert_base(A_df_dev, train_dev_images)
    GC_X_test = multimodal_models.convnext_small_rel(GC_df_test, test_images)
    A_X_test = multimodal_models.swin_v2_s_camembert_base(A_df_test, test_images)
    
    GC_classifier = LogisticRegression ()
    GC_classifier.fit(GC_X_train, GC_y_train)
    GC_y_dev_pred = GC_classifier.predict(GC_X_dev)
    GC_y_test_pred = GC_classifier.predict(GC_X_test)
    
    A_classifier = LogisticRegression ()
    A_classifier.fit(A_X_train, A_y_train)
    A_y_dev_pred = A_classifier.predict(A_X_dev)
    A_y_test_pred = A_classifier.predict(A_X_test)

    y_dev_true = np.concatenate((A_y_dev, GC_y_dev))
    y_dev_pred = np.concatenate((A_y_dev_pred, GC_y_dev_pred))
    y_test_true = np.concatenate((A_y_test, GC_y_test))
    y_test_pred = np.concatenate((A_y_test_pred, GC_y_test_pred))

    pos_label = 'yes'
    dev_f1 = round(f1_score(y_dev_true, y_dev_pred, pos_label=pos_label),4)
    test_f1 = round(f1_score(y_test_true, y_test_pred, pos_label=pos_label),4) 
    print ("dev_score: ", dev_f1)
    print ("test_score: ", test_f1)


def handle_IP_approach_7(GC_train_dataset_path, A_train_dataset_path, GC_dev_dataset_path, A_dev_dataset_path, GC_test_dataset_path, A_test_dataset_path, train_dev_images, test_images):

    GC_df_train, GC_y_train = load_data (GC_train_dataset_path, 'IP')
    A_df_train, A_y_train = load_data (A_train_dataset_path, 'IP')

    GC_df_dev, GC_y_dev = load_data (GC_dev_dataset_path, 'IP')
    A_df_dev, A_y_dev = load_data (A_dev_dataset_path, 'IP')
    
    GC_df_test, GC_y_test = load_data (GC_test_dataset_path, 'IP')
    A_df_test, A_y_test = load_data (A_test_dataset_path, 'IP')

    GC_X_train = multimodal_models.convnext_small_re_OD(GC_df_train, train_dev_images)
    A_X_train = multimodal_models.swin_v2_s_camembert_base_OD(A_df_train, train_dev_images)
    GC_X_dev = multimodal_models.convnext_small_re_OD(GC_df_dev, train_dev_images)
    A_X_dev = multimodal_models.swin_v2_s_camembert_base_OD(A_df_dev, train_dev_images)
    GC_X_test = multimodal_models.convnext_small_re_OD(GC_df_test, test_images)
    A_X_test = multimodal_models.swin_v2_s_camembert_base_OD(A_df_test, test_images)
    
    GC_classifier = LogisticRegression ()
    GC_classifier.fit(GC_X_train, GC_y_train)
    GC_y_dev_pred = GC_classifier.predict(GC_X_dev)
    GC_y_test_pred = GC_classifier.predict(GC_X_test)
    
    A_classifier = LogisticRegression ()
    A_classifier.fit(A_X_train, A_y_train)
    A_y_dev_pred = A_classifier.predict(A_X_dev)
    A_y_test_pred = A_classifier.predict(A_X_test)
    
    y_dev_true = np.concatenate((A_y_dev, GC_y_dev))
    y_dev_pred = np.concatenate((A_y_dev_pred, GC_y_dev_pred))
    y_test_true = np.concatenate((A_y_test, GC_y_test))
    y_test_pred = np.concatenate((A_y_test_pred, GC_y_test_pred))

    pos_label = 'yes'
    dev_f1 = round(f1_score(y_dev_true, y_dev_pred, pos_label=pos_label), 4)
    test_f1 = round(f1_score(y_test_true, y_test_pred, pos_label=pos_label), 4)
    print ("dev_score: ", dev_f1)
    print ("test_score: ", test_f1)


if __name__ == "__main__":
    
    GC_train_dataset_path = "data/gun_control_train.csv"
    A_train_dataset_path = "data/abortion_train.csv"
    GC_test_dataset_path = "test/data/gun_control_test.csv"
    A_test_dataset_path = "test/data/abortion_test.csv"
    GC_dev_dataset_path = "data/gun_control_dev.csv"
    A_dev_dataset_path = "data/abortion_dev.csv"
        
    train_dev_images = "data/images/image/"
    test_images = "test/data/images/image/"

    parser = argparse.ArgumentParser(description='Task and Approach')
    parser.add_argument('--t', type=str, help='Kind of Task')
    parser.add_argument('--a', type=int, help='Number of Approach')

    args = parser.parse_args()
    task, approach = args.t, args.a
    
    approaches_for_AS = {
    1: handle_AS_approach_1,
    2: handle_AS_approach_2,
    3: handle_AS_approach_3,
    4: handle_AS_approach_4,
    5: handle_AS_approach_5,
    6: handle_AS_approach_6
    }
    
    approaches_for_IP = {
    1: handle_IP_approach_1,
    2: handle_IP_approach_2,
    3: handle_IP_approach_3,
    4: handle_IP_approach_4,
    5: handle_IP_approach_5,
    6: handle_IP_approach_6,
    7: handle_IP_approach_7
    }

    approaches = (approaches_for_AS if task == 'AS' else
                  approaches_for_IP if task == 'IP' else
                 (print('Task not found') or None))

    if approaches:
        if approach in approaches:
            selected_approach = approaches[approach]
            if selected_approach:
                
                selected_approach(GC_train_dataset_path, A_train_dataset_path, GC_dev_dataset_path, A_dev_dataset_path, GC_test_dataset_path, A_test_dataset_path, train_dev_images, test_images)
            
            else:
                print("Approach not found or not defined for the given task and approach.")
        else:
            print("Approach not found for the given task.")
    else:
        print("Approaches not defined for the given task.")