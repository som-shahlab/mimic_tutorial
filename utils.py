import femr.ontology
import config
import os
import pickle
import pathlib

def create_or_get_ontology():
    ontology_path = pathlib.Path('ontology.pkl')
    if not ontology_path.exists():
        print("Creating ontology")
        ontology = femr.ontology.Ontology(config.athena_path, 
                                          code_metadata_path=os.path.join(config.database_path, 'metadata', 'codes.parquet'))

        with open(ontology_path, 'wb') as f:
            pickle.dump(ontology, f)
    
    with open(ontology_path, 'rb') as f:
        return pickle.load(f)
