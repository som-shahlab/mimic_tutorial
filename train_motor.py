import transformers
import pathlib
import femr.models.transformer
import pickle
import datasets
import femr.models.tokenizer
import femr.models.processor

def main():
    pretraining_data = pathlib.Path('pretraining_data')

    ontology_path = pretraining_data / 'ontology.pkl'
    with open(ontology_path, 'rb') as f:
        ontology = pickle.load(f)

    tokenizer_path = pretraining_data / 'tokenizer'
    tokenizer = femr.models.tokenizer.FEMRTokenizer.from_pretrained(tokenizer_path, ontology=ontology)


    task_path = pretraining_data / 'motor_task.pkl'
    with open(task_path, 'rb') as f:
        motor_task = pickle.load(f)


    processor = femr.models.processor.FEMRBatchProcessor(tokenizer, motor_task)

    train_batches_path = pretraining_data / 'train_batches'
    train_batches = datasets.Dataset.load_from_disk(train_batches_path)

    val_batches_path = pretraining_data / 'val_batches'
    val_batches = datasets.Dataset.load_from_disk(val_batches_path)

    # Finally, given the batches, we can train CLMBR.
    # We can use huggingface's trainer to do this.

    transformer_config = femr.models.config.FEMRTransformerConfig(
        vocab_size=tokenizer.vocab_size, 
        is_hierarchical=tokenizer.is_hierarchical, 
        n_layers=2,
        hidden_size=64, 
        intermediate_size=64*2,
        n_heads=8,
    )

    config = femr.models.config.FEMRModelConfig.from_transformer_task_configs(transformer_config, motor_task.get_task_config())

    model = femr.models.transformer.FEMRModel(config)

    collator = processor.collate

    trainer_config = transformers.TrainingArguments(
        per_device_train_batch_size=1,
        per_device_eval_batch_size=1,

        output_dir='tmp_trainer',
        remove_unused_columns=False,
        num_train_epochs=100,

        eval_steps=20,
        evaluation_strategy="steps",

        logging_steps=20,
        logging_strategy='steps',

        prediction_loss_only=True,
    )

    trainer = transformers.Trainer(
        model=model,
        data_collator=processor.collate,
        train_dataset=train_batches,
        eval_dataset=val_batches,
        args=trainer_config,
    )

    trainer.train()

    model.save_pretrained('motor_model')
    tokenizer.save_pretrained('motor_model')


if __name__ == "__main__":
    main()