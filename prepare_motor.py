import femr.ontology
import pathlib
import config
import meds_reader
import pickle
import femr.splits
import femr.models.tokenizer
import femr.models.tasks
import femr.models.processor

def main():
    pretraining_data = pathlib.Path('pretraining_data')


    with meds_reader.PatientDatabase(config.database_path, num_threads=32) as database:
        ontology_path = pretraining_data / 'ontology.pkl'
        if not ontology_path.exists():
            print("Creating ontology")
            ontology = femr.ontology.Ontology(config.athena_path, code_metadata=database.metadata['code_metadata'])
            print("Pruning the ontology")
            ontology.prune_to_dataset(
                database,
                prune_all_descriptions=True,
                remove_ontologies={'SPL', 'HemOnc', 'LOINC'}
            )

            with open(ontology_path, 'wb') as f:
                pickle.dump(ontology, f)
        else:
            with open(ontology_path, 'rb') as f:
                ontology = pickle.load(f)

        
        main_split = femr.splits.generate_hash_split(list(database), 97, frac_test=0.15)
        main_split.save_to_csv(pretraining_data / 'main_split.csv')

        train_split = femr.splits.generate_hash_split(main_split.train_patient_ids, 17, frac_test=0.05)

        main_database = database.filter(main_split.train_patient_ids)
        train_database = main_database.filter(train_split.train_patient_ids)
        val_database = main_database.filter(train_split.test_patient_ids)

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

        example_patient_id = list(train_database)[0]
        example_patient = train_database[example_patient_id]

        # We can do this one patient at a time
        print("Convert a single patient")
        example_batch = processor.collate([processor.convert_patient(example_patient, tensor_type='pt')])

        train_batches_path = pretraining_data / 'train_batches'

        if not train_batches_path.exists():
            print("Convert batches")
            # But generally we want to convert entire datasets
            train_batches = processor.convert_dataset(train_database, tokens_per_batch=16 * 1024, num_proc=32)

            print("Convert batches to pytorch")
            # Convert our batches to pytorch tensors
            train_batches.set_format("pt")
            train_batches.save_to_disk(train_batches_path)

        val_batches_path = pretraining_data / 'val_batches'

        if not val_batches_path.exists():
            print("Convert val batches")
            val_batches = processor.convert_dataset(val_database, tokens_per_batch=16 * 1024, num_proc=32)
            # Convert our batches to pytorch tensors
            val_batches.set_format("pt")
            val_batches.save_to_disk(val_batches_path)
        
if __name__ == "__main__":
    main()
