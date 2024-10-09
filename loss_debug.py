import transformers
import pathlib
from tqdm import tqdm
import torch
import sys
import femr.models.transformer
import pickle
import datasets
import femr.models.tokenizer
import femr.models.processor

def main():
    pretraining_data = pathlib.Path('pretraining_data')

    feature_ontology_path = pretraining_data / 'feature_ontology.pkl'

    with open(feature_ontology_path, 'rb') as f:
        ontology = pickle.load(f)

    tokenizer_path = pretraining_data / 'tokenizer'
    tokenizer = femr.models.tokenizer.HierarchicalTokenizer.from_pretrained(tokenizer_path, ontology=ontology)


    task_path = pretraining_data / 'motor_task.pkl'
    with open(task_path, 'rb') as f:
        motor_task = pickle.load(f)


    processor = femr.models.processor.FEMRBatchProcessor(tokenizer, motor_task)

    train_batches_path = pretraining_data / 'train_batches'
    train_batches = datasets.Dataset.load_from_disk(train_batches_path)

    val_batches_path = pretraining_data / 'val_batches'
    val_batches = datasets.Dataset.load_from_disk(val_batches_path)
    val_batches = val_batches.select(range(120))
    
    train_batches.set_format("pt")

#    loader = torch.utils.data.DataLoader(train_batches, num_workers=4, pin_memory=True, collate_fn=processor.collate, shuffle=False)#
    loader = torch.utils.data.DataLoader(train_batches, num_workers=4, pin_memory=True, collate_fn=processor.collate, shuffle=False, sampler=[110])#
#                                                                                                                                              , 10570])
    
    model = femr.models.transformer.FEMRModel.from_pretrained("tmp_trainer_1e-3/checkpoint-9500", task_config=motor_task.get_task_config())
    model = model.to(torch.device("cuda"))

    with torch.no_grad():
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            for i, batch in enumerate(loader):
                batch = femr.models.transformer.to_device(batch, torch.device('cuda'))
                loss, _ = model(**batch)
                if loss.item() > 0.3:
                    print(loss, i)
                    with open(f'bad_batch_{i}.pkl', 'wb') as f:
                        pickle.dump(batch, f)
if __name__ == "__main__":
    main()
