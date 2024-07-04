import femr.transforms
import config
import meds_reader
import femr.models.transformer
import pyarrow.csv as pacsv
import os
import pickle
import meds
import pathlib

def main():
    with meds_reader.PatientDatabase(config.database_path, num_threads=6) as database:

        pretraining_data = pathlib.Path('pretraining_data')

        ontology_path = pretraining_data / 'ontology.pkl'

        with open(ontology_path, 'rb') as f:
            ontology = pickle.load(f)

        for label_name in config.label_names:
            labels = pacsv.read_csv(os.path.join('labels', label_name + '.csv')).cast(meds.label).to_pylist()

            features = femr.models.transformer.compute_features(
                db=database, model_path='motor_model', labels=labels, ontology=ontology)

            with open(os.path.join('features', label_name + '_motor.pkl'), 'wb') as f:
                pickle.dump(features, f)



if __name__ == "__main__":
    main()