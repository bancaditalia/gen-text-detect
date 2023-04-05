# Results

## gboost_cv_results.csv

import pickle
import pandas as pd
import numpy as np

Save path for evaluation reports

results_path = "./results"

Load training data
Load pickled embedded training data

# path_prefix = "./data/embed"
path_prefix = "./../../get_text_detect_space/datasets/embed/"
dataset_path = "webtext.train.jsonl.clean100k.csv+xl-1542M.train.jsonl.clean100k.csv_embed.pickle"
with open(os.path.join(path_prefix, dataset_path), "rb") as f:
    training_data = pickle.load(f)

​

X = training_data["X"]
y = training_data["y"]

X

array([[ 1.56034485e-01, -1.03202422e-03,  9.29313446e-02, ...,
        -6.20653304e-03,  1.78792963e-02, -1.16704679e-02],
       [ 1.19502099e-01,  1.64194560e-02, -1.96907246e-02, ...,
        -1.13290602e-02,  5.91688235e-03,  1.52714059e-03],
       [ 1.67067327e-01,  5.72166997e-02, -9.31411220e-03, ...,
        -6.56531715e-03, -1.25897509e-02,  2.12441978e-03],
       ...,
       [ 1.45809149e-01, -3.29254096e-02, -2.40912571e-02, ...,
         6.88818067e-03, -4.50552379e-03, -3.19229779e-03],
       [ 1.07385538e-01, -5.34839545e-02,  2.93454104e-02, ...,
        -1.70121954e-02, -6.23178832e-03,  8.76551081e-03],
       [ 1.83371955e-01,  9.32380781e-02, -1.46499623e-03, ...,
         1.69348226e-05, -6.23653770e-03,  8.20869394e-03]])

Load test data

Load pickled embedded test data

#path_prefix = "./data/embed"
path_prefix = "./../../get_text_detect_space/datasets/embed/"
dataset_names = ["webtext.test.human_embed.pickle", "gpt2.xl-1542M.test.machine_embed.pickle"]
with open(os.path.join(path_prefix, dataset_names[0]), "rb") as f:
    test_data = pickle.load(f)
    X_test = test_data["X"]
    y_test = test_data["y"]
with open(os.path.join(path_prefix, dataset_names[1]), "rb") as f:
    test_data = pickle.load(f)
    X_test = np.concatenate((X_test, test_data["X"]))
    y_test += test_data["y"]

Train classifiers

from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report

GradientBoosting

from sklearn.ensemble import GradientBoostingClassifier
gboost = GradientBoostingClassifier(random_state=42)

# parameters = {'n_estimators':[100, 200, 300], 'max_depth':[3, 4, 5], 'learning_rate':[0.2, 0.5, 1.0]}

parameters = {'n_estimators':[100, 200, 300], 'max_depth':[3, 4, 5]}

​

clf_gboost = GridSearchCV(gboost, parameters, n_jobs = -1)

clf_gboost.fit(X, y)

GridSearchCV

GridSearchCV(estimator=GradientBoostingClassifier(random_state=42), n_jobs=-1,
             param_grid={'max_depth': [3, 4, 5],
                         'n_estimators': [100, 200, 300]})

estimator: GradientBoostingClassifier

GradientBoostingClassifier(random_state=42)

GradientBoostingClassifier

GradientBoostingClassifier(random_state=42)

​

filename = 'trained_clf_gboost_on_traindata.sav'

pickle.dump(clf_gboost, open(filename, 'wb'))

Save cross-validation results

df_clf_gboost = pd.DataFrame(clf_gboost.cv_results_)

​

df_clf_gboost.to_csv(os.path.join(results_path,"gboost_cv_results.csv"))

df_clf_gboost

	mean_fit_time 	std_fit_time 	mean_score_time 	std_score_time 	param_max_depth 	param_n_estimators 	params 	split0_test_score 	split1_test_score 	split2_test_score 	split3_test_score 	split4_test_score 	mean_test_score 	std_test_score 	rank_test_score
0 	2614.299360 	9.099203 	0.310966 	0.002034 	3 	100 	{'max_depth': 3, 'n_estimators': 100} 	0.651975 	0.653600 	0.646300 	0.651500 	0.652600 	0.651195 	0.002546 	9
1 	5235.116350 	13.629643 	0.599004 	0.011473 	3 	200 	{'max_depth': 3, 'n_estimators': 200} 	0.669025 	0.669000 	0.663500 	0.669175 	0.669325 	0.668005 	0.002256 	6
2 	7642.973242 	71.442550 	0.507261 	0.015560 	3 	300 	{'max_depth': 3, 'n_estimators': 300} 	0.678950 	0.678450 	0.672675 	0.679550 	0.678650 	0.677655 	0.002518 	5
3 	3503.082673 	18.487867 	0.415220 	0.006028 	4 	100 	{'max_depth': 4, 'n_estimators': 100} 	0.663825 	0.660225 	0.655225 	0.662375 	0.664075 	0.661145 	0.003261 	8
4 	6887.899182 	83.366777 	0.602796 	0.129157 	4 	200 	{'max_depth': 4, 'n_estimators': 200} 	0.679475 	0.677225 	0.673250 	0.679900 	0.679500 	0.677870 	0.002495 	4
5 	9696.451344 	125.771665 	0.653208 	0.012448 	4 	300 	{'max_depth': 4, 'n_estimators': 300} 	0.688825 	0.685200 	0.683275 	0.688025 	0.688325 	0.686730 	0.002140 	2
6 	4298.379048 	70.677612 	0.447520 	0.065963 	5 	100 	{'max_depth': 5, 'n_estimators': 100} 	0.670275 	0.668450 	0.662025 	0.667850 	0.668750 	0.667470 	0.002838 	7
7 	7590.459716 	180.395606 	0.486268 	0.026785 	5 	200 	{'max_depth': 5, 'n_estimators': 200} 	0.685325 	0.686225 	0.678525 	0.683675 	0.686100 	0.683970 	0.002870 	3
8 	10522.496810 	180.969857 	0.675348 	0.030328 	5 	300 	{'max_depth': 5, 'n_estimators': 300} 	0.692375 	0.693975 	0.687750 	0.692225 	0.692350 	0.691735 	0.002094 	1

Evaluate best-classifier on test data

y_predict = clf_gboost.predict(X_test)

df_cr = pd.DataFrame(classification_report(y_test, y_predict, output_dict=True))

df_cr.to_csv(os.path.join(results_path,"gboost_test_results.csv"))

df_cr

	0 	1 	accuracy 	macro avg 	weighted avg
precision 	0.689562 	0.676317 	0.6827 	0.682940 	0.682940
recall 	0.664600 	0.700800 	0.6827 	0.682700 	0.682700
f1-score 	0.676851 	0.688341 	0.6827 	0.682596 	0.682596
support 	5000.000000 	5000.000000 	0.6827 	10000.000000 	10000.000000



=========================
POS 
=========================


import os

import pickle

import pandas as pd

import numpy as np

Save path for evaluation reports

results_path = "./results"

Load training data

Load pickled embedded training data

path_prefix = "./../../get_text_detect_space/datasets/pos/embed/"

dataset_path = "webtext_xl-1542M.pos_embed.pickle"

with open(os.path.join(path_prefix, dataset_path), "rb") as f:

    training_data = pickle.load(f)

​

X = training_data["X"]

y = training_data["y"]

X

array([[ 2.76196080e-01,  5.97072176e-02, -1.75234160e-01, ...,
        -5.05302750e-05,  3.08036225e-04,  4.49667451e-03],
       [ 1.76427847e-01, -1.36848282e-02,  1.94081052e-02, ...,
        -2.31185733e-02, -1.06349651e-03, -6.36850336e-03],
       [ 3.00999734e-01,  7.07699521e-02,  4.49760848e-02, ...,
        -6.50049729e-03, -2.29744011e-02, -1.36919613e-02],
       ...,
       [ 3.03001908e-01, -8.60717365e-02,  6.73780364e-02, ...,
         2.23266651e-02, -5.90136098e-03, -6.57305439e-03],
       [ 1.96314299e-01, -2.86477333e-02, -6.43956552e-02, ...,
        -1.33383768e-02,  1.49197168e-02,  3.75897600e-02],
       [ 2.95057276e-01,  1.36087709e-01,  6.60210385e-03, ...,
        -1.06794292e-02,  6.33336589e-03,  5.11625681e-03]])

Load test data

Load pickled embedded test data

path_prefix = "./../../get_text_detect_space/datasets/pos/embed/"

dataset_names = ["webtext.test_xl-1542M.test.pos_embed.pickle"]

# , "gpt2.xl-1542M.test.machine_embed.pickle"]

with open(os.path.join(path_prefix, dataset_names[0]), "rb") as f:

    test_data = pickle.load(f)

    X_test = test_data["X"]

    y_test = test_data["y"]

if len(dataset_names)>1:

    for dataset_name in dataset_names[1:]:

        with open(os.path.join(path_prefix, dataset_name), "rb") as f:

            test_data = pickle.load(f)

            X_test = np.concatenate((X_test, test_data["X"]))

            y_test += test_data["y"]

X_test

array([[ 0.03456433,  0.00284847,  0.02317966, ...,  0.06093252,
        -0.01726509, -0.01650083],
       [ 0.29024403,  0.10092605, -0.08028228, ...,  0.00875937,
        -0.00962994, -0.00285187],
       [ 0.22424307, -0.03866471,  0.00745562, ...,  0.01225226,
         0.01454416, -0.02042341],
       ...,
       [ 0.33834934, -0.08588207, -0.02701483, ..., -0.00609007,
         0.00459895,  0.0048186 ],
       [ 0.2475311 , -0.04341262, -0.0432451 , ...,  0.01100336,
        -0.01325901,  0.01135516],
       [ 0.30625864, -0.09640497, -0.05410161, ...,  0.00278548,
         0.00502804, -0.00166508]])

Train classifiers

from sklearn.model_selection import GridSearchCV

from sklearn.metrics import classification_report

## GradientBoosting

from sklearn.ensemble import GradientBoostingClassifier

gboost = GradientBoostingClassifier(random_state=42)

parameters = {'n_estimators':[100, 200, 300], 'max_depth':[3, 4, 5]}

# parameters = {'n_estimators':[100, 200, 300], 'max_depth':[3, 4, 5], 'learning_rate':[0.2, 0.5, 1.0]}

​

clf_gboost = GridSearchCV(gboost, parameters, n_jobs=-1)

clf_gboost.fit(X, y)

GridSearchCV

GridSearchCV(estimator=GradientBoostingClassifier(random_state=42), n_jobs=-1,
             param_grid={'max_depth': [3, 4, 5],
                         'n_estimators': [100, 200, 300]})

estimator: GradientBoostingClassifier

GradientBoostingClassifier(random_state=42)

GradientBoostingClassifier

GradientBoostingClassifier(random_state=42)

Save cross-validation results

df_clf_gboost = pd.DataFrame(clf_gboost.cv_results_)

​

df_clf_gboost.to_csv(os.path.join(results_path,"gboost_cv_pos_results.csv"))

df_clf_gboost

	mean_fit_time 	std_fit_time 	mean_score_time 	std_score_time 	param_max_depth 	param_n_estimators 	params 	split0_test_score 	split1_test_score 	split2_test_score 	split3_test_score 	split4_test_score 	mean_test_score 	std_test_score 	rank_test_score
0 	2624.502299 	15.466873 	0.243126 	0.010153 	3 	100 	{'max_depth': 3, 'n_estimators': 100} 	0.662700 	0.661900 	0.660200 	0.663000 	0.662025 	0.661965 	0.000973 	9
1 	5220.052750 	53.032160 	0.437142 	0.077415 	3 	200 	{'max_depth': 3, 'n_estimators': 200} 	0.672625 	0.671375 	0.671975 	0.672850 	0.673525 	0.672470 	0.000738 	6
2 	7622.130860 	168.292886 	0.390650 	0.029468 	3 	300 	{'max_depth': 3, 'n_estimators': 300} 	0.679200 	0.676425 	0.677725 	0.677375 	0.676375 	0.677420 	0.001034 	5
3 	3431.191924 	40.729748 	0.290370 	0.048281 	4 	100 	{'max_depth': 4, 'n_estimators': 100} 	0.668300 	0.669025 	0.668100 	0.665975 	0.668275 	0.667935 	0.001030 	8
4 	6631.505816 	201.076624 	0.499095 	0.138678 	4 	200 	{'max_depth': 4, 'n_estimators': 200} 	0.677825 	0.678500 	0.678950 	0.677275 	0.677325 	0.677975 	0.000657 	4
5 	9548.058559 	121.768784 	0.537699 	0.023085 	4 	300 	{'max_depth': 4, 'n_estimators': 300} 	0.681525 	0.683225 	0.683600 	0.680000 	0.680775 	0.681825 	0.001388 	2
6 	4381.163130 	32.582734 	0.378633 	0.026582 	5 	100 	{'max_depth': 5, 'n_estimators': 100} 	0.672625 	0.674025 	0.672375 	0.670500 	0.672675 	0.672440 	0.001129 	7
7 	7591.126578 	162.899287 	0.408237 	0.026638 	5 	200 	{'max_depth': 5, 'n_estimators': 200} 	0.679350 	0.682475 	0.682750 	0.680025 	0.680975 	0.681115 	0.001330 	3
8 	10624.138893 	302.307753 	0.591061 	0.024562 	5 	300 	{'max_depth': 5, 'n_estimators': 300} 	0.683125 	0.685775 	0.685925 	0.681775 	0.683275 	0.683975 	0.001618 	1

Evaluate best-classifier on test data

y_predict = clf_gboost.predict(X_test)

df_cr = pd.DataFrame(classification_report(y_test, y_predict, output_dict=True))

df_cr.to_csv(os.path.join(results_path,"gboost_test_pos_results.csv"))

df_cr

	0 	1 	accuracy 	macro avg 	weighted avg
precision 	0.676608 	0.671662 	0.6741 	0.674135 	0.674135
recall 	0.667000 	0.681200 	0.6741 	0.674100 	0.674100
f1-score 	0.671770 	0.676398 	0.6741 	0.674084 	0.674084
support 	5000.000000 	5000.000000 	0.6741 	10000.000000 	10000.000000
