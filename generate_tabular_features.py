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
import config

def main():
    if not os.path.exists('features'):
        os.mkdir('features')

    with meds_reader.SubjectDatabase(config.database_path, num_threads=config.num_threads) as database:
        for label_name in config.label_names:
            labels = pd.read_parquet(os.path.join('labels', label_name + '.parquet'))

            featurizer = femr.featurizers.FeaturizerList([
                femr.featurizers.AgeFeaturizer(is_normalize=True),
                femr.featurizers.CountFeaturizer(),
            ])

            print("Preprocessing")

            featurizer.preprocess_featurizers(database, labels)

            print("Done preprossing, about to featurize")

            with open(os.path.join('features', label_name + '_featurizer.pkl'), 'wb') as f:
                pickle.dump(featurizer, f)

            features = featurizer.featurize(database, labels)

            print("Done featurizing")

            with open(os.path.join('features', label_name + '.pkl'), 'wb') as f:
                pickle.dump(features, f)

if __name__ == "__main__":
    main()
