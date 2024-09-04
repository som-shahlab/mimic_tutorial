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
from config import label_names, database_path
import numpy as np
import sklearn.linear_model
import functools
import femr.splits
import polars as pl
import json


def main():
    if not os.path.exists('results'):
        os.mkdir('results')

    if not os.path.exists('predictions'):
        os.mkdir('predictions')

    for label_name in label_names:
        labels = pd.read_parquet(os.path.join('labels', label_name + '.parquet'))

        if not os.path.exists('results/' + label_name):
            os.mkdir('results/' + label_name)

        if not os.path.exists('predictions/' + label_name):
            os.mkdir('predictions/' + label_name)

        with open(os.path.join('features', label_name + '_motor.pkl'), 'rb') as f:
            features = pickle.load(f)
    
        labeled_features = femr.featurizers.join_labels(features, labels)

        splits = pl.read_parquet(os.path.join(database_path, 'metadata', 'subject_splits.parquet'))
        train_subject_ids = list(splits.filter(pl.col('split') != 'held_out').select('subject_id').to_series())
        
        train_mask = np.isin(labeled_features['subject_ids'], train_subject_ids)
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
        held_out_data = apply_mask(labeled_features, held_out_mask)

        model = sklearn.linear_model.LogisticRegressionCV(scoring='roc_auc')
        model.fit(train_data['features'], train_data['boolean_values'])

        y_pred = model.predict_log_proba(held_out_data['features'])[:, 1]

        final_auroc = sklearn.metrics.roc_auc_score(held_out_data['boolean_values'], y_pred)

        with open(os.path.join('results', label_name, 'motor_logistic.json'), 'w') as f:
            json.dump({'auroc': final_auroc}, f)

        predictions = model.predict_log_proba(labeled_features['features'])[:, 1]

        data = pd.DataFrame.from_dict({
            'boolean_prediction': predictions, 
            'subject_id': labeled_features['subject_ids'], 
            'prediction_time': labeled_features['times'],
            'boolean_value': labeled_features['boolean_values'],
        })

        data.to_parquet(os.path.join('predictions', label_name, 'motor_logistic.parquet'), index=False)

        print(label_name, final_auroc)


if __name__ == "__main__":
    main()
