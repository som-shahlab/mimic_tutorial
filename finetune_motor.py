"""
FEMR also supports generating tabular feature representations, an important baseline for EHR modeling
"""

import os
import shutil
import meds_reader
import pandas as pd
import femr.featurizers
import pyarrow.csv as pacsv
import meds
import pickle
from config import label_names
import numpy as np
import sklearn.linear_model
import optuna
import functools
import lightgbm as lgb


def logistic_objective(trial, *, train_data, dev_data):
    c = trial.suggest_float('c', 1e-4, 1e4, log=True)

    model = sklearn.linear_model.LogisticRegression(C=c)

    model.fit(train_data['features'], train_data['boolean_values'])

    y_pred = model.predict_log_proba(dev_data['features'])[:, 1]

    error = -sklearn.metrics.roc_auc_score(dev_data['boolean_values'], y_pred)

    return error 
    

def main():
    if os.path.exists('models'):
        shutil.rmtree('models')

    os.mkdir('models')

    with meds_reader.PatientDatabase("../mimic-iv-demo-meds-reader", num_threads=6) as database:
        for label_name in label_names:
            labels = pacsv.read_csv(os.path.join('labels', label_name + '.csv')).cast(meds.label).to_pylist()

            with open(os.path.join('features', label_name + '_motor.pkl'), 'rb') as f:
                features = pickle.load(f)
        
            labeled_features = femr.featurizers.join_labels(features, labels)

            patient_ids = np.unique(labeled_features['patient_ids'])
            np.random.seed(4534353)
            np.random.shuffle(patient_ids)

            frac_train = 0.7
            frac_dev = 0.15

            num_train = int(frac_train * len(patient_ids))
            num_dev = int(frac_dev * len(patient_ids))

            train_patients = patient_ids[:num_train]
            dev_patients = patient_ids[num_train: num_train + num_dev]
            test_patients = patient_ids[num_train + num_dev:]

            train_mask = np.isin(labeled_features['patient_ids'], train_patients)
            dev_mask = np.isin(labeled_features['patient_ids'], dev_patients)
            test_mask = np.isin(labeled_features['patient_ids'], test_patients)

            def apply_mask(values, mask):
                def apply(k, v):
                    if len(v.shape) == 1:
                        return v[mask]
                    elif len(v.shape) == 2:
                        return v[mask, :]
                    else:
                        assert False, f"Cannot handle {k} {v.shape}"

                return {k: apply(k, v) for k, v in values.items()}
            
            train_data = apply_mask(labeled_features, train_mask)
            dev_data = apply_mask(labeled_features, dev_mask)
            test_data = apply_mask(labeled_features, test_mask)

            logistic_study = optuna.create_study()  # Create a new study.
            logistic_study.optimize(functools.partial(logistic_objective, train_data=train_data, dev_data=dev_data), n_trials=10)  # Invoke optimization of the objective function.

            final_train_data = apply_mask(labeled_features, train_mask | dev_mask)
            
            model = sklearn.linear_model.LogisticRegression(C=logistic_study.best_params['c'])
            model.fit(final_train_data['features'], final_train_data['boolean_values'])

            y_pred = model.predict_log_proba(test_data['features'])[:, 1]

            final_auroc = sklearn.metrics.roc_auc_score(test_data['boolean_values'], y_pred)

            print(final_auroc)


if __name__ == "__main__":
    main()