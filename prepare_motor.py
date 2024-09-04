import femr.ontology
import pathlib
import config
import meds_reader
import pickle
import femr.splits
import femr.models.tokenizer
import femr.models.tasks
import femr.models.processor
import os
import utils

import polars as pl

def main():
    pretraining_data = pathlib.Path('pretraining_data')

    if not pretraining_data.exists():
        pretraining_data.mkdir()

    feature_ontology_path = pretraining_data / 'feature_ontology.pkl'

    if not feature_ontology_path.exists():
        print("Creating ontology")
        ontology = utils.create_or_get_ontology()

        with meds_reader.SubjectDatabase(config.database_path, num_threads=2) as database:
            print("Pruning the ontology")
            ontology.prune_to_dataset(
                database,
                prune_all_descriptions=True,
                remove_ontologies={'SPL', 'HemOnc', 'LOINC'}
            )

        with open(feature_ontology_path, 'wb') as f:
            pickle.dump(ontology, f)
    else:
        with open(feature_ontology_path, 'rb') as f:
            ontology = pickle.load(f)

    splits = pl.read_parquet(os.path.join(config.database_path, 'metadata', 'subject_splits.parquet'))
    train_subject_ids = list(splits.filter(pl.col('split') != 'held_out').select('subject_id').to_series())

    with meds_reader.SubjectDatabase(config.database_path, num_threads=config.num_threads) as database:
        train_split = femr.splits.generate_hash_split(train_subject_ids, 17, frac_test=0.1)

        main_database = database.filter(train_subject_ids)
        train_database = main_database.filter(train_split.train_subject_ids)
        val_database = main_database.filter(train_split.test_subject_ids)

        tokenizer_path = pretraining_data / 'tokenizer'
        if not tokenizer_path.exists():
            print("Train tokenizer")
            tokenizer = femr.models.tokenizer.train_tokenizer(
                main_database, vocab_size=1024 * 16, is_hierarchical=True, ontology=ontology)

            # Save the tokenizer to the same directory as the model
            tokenizer.save_pretrained(tokenizer_path)
        else:
            tokenizer = femr.models.tokenizer.FEMRTokenizer.from_pretrained(tokenizer_path, ontology=ontology)


        task_path = pretraining_data / 'motor_task.pkl'

        if not task_path.exists():
            # Second, we need to prefit the MOTOR model. This is necessary because piecewise exponential models are unstable without an initial fit
            print("Train MOTOR task")

            motor_task = femr.models.tasks.MOTORTask.fit_pretraining_task_info(
                main_database, tokenizer, num_tasks=8 * 1024, num_bins=8, final_layer_size=512)
            
            with open(task_path, 'wb') as f:
                pickle.dump(motor_task, f)

        else:
            with open(task_path, 'rb') as f:
                motor_task = pickle.load(f)


        processor = femr.models.processor.FEMRBatchProcessor(tokenizer, motor_task)

        example_subject_id = list(train_database)[0]
        example_subject = train_database[example_subject_id]

        # We can do this one subject at a time
        print("Convert a single subject")
        example_batch = processor.collate([processor.convert_subject(example_subject, tensor_type='pt')])

        train_batches_path = pretraining_data / 'train_batches'

        if not train_batches_path.exists():
            print("Convert batches")
            # But generally we want to convert entire datasets
            train_batches = processor.convert_dataset(train_database, tokens_per_batch=16 * 1024, num_proc=config.num_threads)

            print("Convert batches to pytorch")
            # Convert our batches to pytorch tensors
            train_batches.set_format("pt")
            train_batches.save_to_disk(train_batches_path)

        val_batches_path = pretraining_data / 'val_batches'

        if not val_batches_path.exists():
            print("Convert val batches")
            val_batches = processor.convert_dataset(val_database, tokens_per_batch=16 * 1024, num_proc=config.num_threads)
            # Convert our batches to pytorch tensors
            val_batches.set_format("pt")
            val_batches.save_to_disk(val_batches_path)
        
if __name__ == "__main__":
    main()
