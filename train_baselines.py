"""
FEMR also supports generating tabular feature representations, an important baseline for EHR modeling
"""

import femr.splits
import os
import shutil
import meds_reader
import pandas as pd
import femr.featurizers
import pyarrow.csv as pacsv
import meds
import pickle
from config import label_names, num_threads, database_path
import numpy as np
import sklearn.linear_model
import optuna
import functools
import lightgbm as lgb
import pandas as pd
import polars as pl
import json

def lightgbm_objective(trial, *, train_data, dev_data, num_trees = None, return_model=False):
    param = {
        "objective": "binary",
        "metric": "auc",
        "verbosity": -1,

        "lambda_l1": trial.suggest_float("lambda_l1", 1e-8, 10.0, log=True),
        "lambda_l2": trial.suggest_float("lambda_l2", 1e-8, 10.0, log=True),
        "num_leaves": trial.suggest_int("num_leaves", 2, 256),
        "feature_fraction": trial.suggest_float("feature_fraction", 0.4, 1.0),
        "bagging_fraction": trial.suggest_float("bagging_fraction", 0.4, 1.0),
        "bagging_freq": trial.suggest_int("bagging_freq", 1, 7),
        "min_child_samples": trial.suggest_int("min_child_samples", 5, 100),
    }

    dtrain = lgb.Dataset(train_data['features'], label=train_data['boolean_values'])
    ddev = lgb.Dataset(dev_data['features'], label=dev_data['boolean_values'])

    if num_trees is None:
        callbacks = [lgb.early_stopping(10)]
        gbm = lgb.train(param, dtrain, num_boost_round=1000, valid_sets=(ddev,), callbacks=callbacks)
    else:
        gbm = lgb.train(param, dtrain, num_boost_round=num_trees)
    
    y_pred = gbm.predict(dev_data['features'], raw_score=True)

    error = -sklearn.metrics.roc_auc_score(dev_data['boolean_values'], y_pred)

    if num_trees is None:
        trial.set_user_attr("num_trees", gbm.best_iteration + 1)

    if return_model:
        return error, gbm
    else:
        return error


def main():
    if not os.path.exists('results'):
        os.mkdir('results')

    for label_name in label_names:
        print(label_name)
        labels = pd.read_parquet(os.path.join('labels', label_name + '.parquet'))

        with open(os.path.join('features', label_name + '.pkl'), 'rb') as f:
            features = pickle.load(f)

        labeled_features = femr.featurizers.join_labels(features, labels)

        if not os.path.exists('results/' + label_name):
            os.mkdir('results/' + label_name)

        if not os.path.exists('predictions/' + label_name):
            os.mkdir('predictions/' + label_name)

        splits = pl.read_parquet(os.path.join(database_path, 'metadata', 'subject_splits.parquet'))
        train_subject_ids = list(splits.filter(pl.col('split') != 'held_out').select('subject_id').to_series())

        train_split = femr.splits.generate_hash_split(train_subject_ids, 17, frac_test=0.20)
        
        train_mask = np.isin(labeled_features['subject_ids'], train_split.train_subject_ids)
        dev_mask = np.isin(labeled_features['subject_ids'], train_split.test_subject_ids)
        held_out_mask = ~np.isin(labeled_features['subject_ids'], train_subject_ids)

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
        held_out_data = apply_mask(labeled_features, held_out_mask)

        lightgbm_study = optuna.create_study()  # Create a new study.
        lightgbm_study.optimize(functools.partial(lightgbm_objective, train_data=train_data, dev_data=dev_data), n_trials=30)  # Invoke optimization of the objective function.

        final_train_data = apply_mask(labeled_features, train_mask | dev_mask)

        final_lightgbm_auroc, gbm = lightgbm_objective(
            lightgbm_study.best_trial, 
            train_data=final_train_data, 
            dev_data=held_out_data, 
            num_trees = lightgbm_study.best_trial.user_attrs['num_trees'],
            return_model = True)
        print(label_name)

        final_lightgbm_auroc *= -1

        predictions = gbm.predict(labeled_features['features'], raw_score=True)

        data = pd.DataFrame.from_dict({
            'boolean_prediction': predictions, 
            'subject_id': labeled_features['subject_ids'], 
            'prediction_time': labeled_features['times'],
            'boolean_value': labeled_features['boolean_values'],
        })

        data.to_parquet(os.path.join('predictions', label_name, 'count_lightgbm.parquet'), index=False)

        print('lightgbm', final_lightgbm_auroc, label_name)

        with open(os.path.join('results', label_name, 'counts_lightgbm.json'), 'w') as f:
            json.dump({'auroc': final_lightgbm_auroc}, f)

        logistic_model = sklearn.linear_model.LogisticRegressionCV(scoring='roc_auc')
        logistic_model.fit(final_train_data['features'], final_train_data['boolean_values'])

        logistic_y_pred = logistic_model.predict_log_proba(held_out_data['features'])[:, 1]

        final_logistic_auroc = sklearn.metrics.roc_auc_score(held_out_data['boolean_values'], logistic_y_pred)

        with open(os.path.join('results', label_name, 'counts_logistic.json'), 'w') as f:
            json.dump({'auroc': final_logistic_auroc}, f)

        
        logistic_y_pred = logistic_model.predict_log_proba(labeled_features['features'])[:, 1]

        data = pd.DataFrame.from_dict({
            'boolean_prediction': predictions, 
            'subject_id': labeled_features['subject_ids'], 
            'prediction_time': labeled_features['times'],
            'boolean_value': labeled_features['boolean_values'],
        })

        data.to_parquet(os.path.join('predictions', label_name, 'count_logistic.parquet'), index=False)

        print('logistic', final_logistic_auroc, label_name)


if __name__ == "__main__":
    main()
