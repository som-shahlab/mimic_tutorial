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
import functools
import femr.splits


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

            main_split = femr.splits.PatientSplit.load_from_csv('pretraining_data/main_split.csv')

            train_mask = np.isin(labeled_features['patient_ids'], main_split.train_patient_ids)
            test_mask = np.isin(labeled_features['patient_ids'], main_split.test_patient_ids)

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
            test_data = apply_mask(labeled_features, test_mask)

            model = sklearn.linear_model.LogisticRegressionCV(scoring='roc_auc')
            model.fit(train_data['features'], train_data['boolean_values'])

            y_pred = model.predict_log_proba(test_data['features'])[:, 1]

            final_auroc = sklearn.metrics.roc_auc_score(test_data['boolean_values'], y_pred)

            print(label_name, final_auroc)


if __name__ == "__main__":
    main()
