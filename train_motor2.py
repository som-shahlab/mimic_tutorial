import transformers
import pathlib
import torch
import sys
import femr.models.transformer
import pickle
import datasets
import femr.models.tokenizer
import femr.models.processor
import shutil
import os
import datetime

import schedulefree

from tqdm import tqdm

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
    train_batches.set_format('pt')


    val_batches_path = pretraining_data / 'val_batches'
    val_batches = datasets.Dataset.load_from_disk(val_batches_path)
    val_batches = val_batches.select(range(120))
    val_batches.set_format('pt')

    train_loader = torch.utils.data.DataLoader(train_batches, num_workers=4, pin_memory=True, collate_fn=processor.collate, shuffle=True)
    val_loader = torch.utils.data.DataLoader(val_batches, num_workers=4, pin_memory=True, collate_fn=processor.collate)

    # Finally, given the batches, we can train CLMBR.
    # We can use huggingface's trainer to do this.

    transformer_config = femr.models.config.FEMRTransformerConfig(
        vocab_size=tokenizer.vocab_size, 
        is_hierarchical=True, 
        n_layers=6,
        use_normed_ages=True,
        use_bias=False,
        hidden_act='swiglu',
    )

    config = femr.models.config.FEMRModelConfig.from_transformer_task_configs(transformer_config, motor_task.get_task_config())

    model = femr.models.transformer.FEMRModel(config)

    # model = femr.models.transformer.FEMRModel.from_pretrained('tmp_trainer_1e-4/checkpoint-500')
    model = model.to(torch.device("cuda"))

    learning_rate = float(sys.argv[1])

    weight_decay = 0.1
    log_steps = 500
    warmup_steps = 500
    schedule_free = True

    decay_params = transformers.trainer_pt_utils.get_parameter_names(model, transformers.pytorch_utils.ALL_LAYERNORM_LAYERS)
    decay_params = [name for name in decay_params if "bias" not in name]

    params = [
        {"params": [p for n, p in model.named_parameters() if (n in decay_params and p.requires_grad)],
         "weight_decay": weight_decay
         },
        {"params": [p for n, p in model.named_parameters() if (n not in decay_params and p.requires_grad)],
        "weight_decay": 0}
    ]

    if schedule_free:
        optimizer = schedulefree.AdamWScheduleFree(
            params,
            lr=learning_rate,
            betas=[0.9, 0.95],
            warmup_steps=warmup_steps,
        )
    else:
        optimizer = torch.optim.AdamW(
            params,
            lr=learning_rate,
            betas=[0.9, 0.95],
        )
        scheduler = transformers.optimization.get_constant_schedule_with_warmup(optimizer, 500)

    target_dir = 'train_model4'

    os.mkdir(target_dir)

    def compute_val_loss():
        with torch.no_grad():
            total_loss = None
            num_batches = 0
            for batch in tqdm(val_loader):
                batch = femr.models.transformer.to_device(batch, torch.device('cuda'))
                with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                    loss, results = model(**batch)
                del batch
                if total_loss is None:
                    total_loss = loss
                else:
                    total_loss += loss

                num_batches += 1

            return total_loss.item() / num_batches


    best_val_loss = None
    train_loss = None

    log_file = open(os.path.join(target_dir, 'log'), 'w')
    print("Starting to train", learning_rate, weight_decay, schedule_free)
    print("Starting to train", learning_rate, weight_decay, schedule_free, file=log_file, flush=True)

    for epoch in range(10):
        for i, batch in enumerate(tqdm(train_loader)):
            if i % log_steps == 0:
                model.eval()
                if schedule_free:
                    optimizer.eval()

                if os.path.exists(os.path.join(target_dir, 'latest')):
                    shutil.rmtree(os.path.join(target_dir, 'latest'))

                model.save_pretrained(os.path.join(target_dir, 'latest'))
                tokenizer.save_pretrained(os.path.join(target_dir, 'latest'))

                val_loss = compute_val_loss()

                if best_val_loss is None or val_loss < best_val_loss:
                    best_val_loss = val_loss

                    if os.path.exists(os.path.join(target_dir, 'best')):
                        shutil.rmtree(os.path.join(target_dir, 'best'))

                    shutil.copytree(os.path.join(target_dir, 'latest'), os.path.join(target_dir, 'best'))

                if train_loss is not None:
                    train_loss_val = train_loss.item() / log_steps
                else:
                    train_loss_val = 0
                train_loss = None
                print("Got loss at", epoch, i, train_loss_val, val_loss, best_val_loss, datetime.datetime.now())
                print("Got loss at", epoch, i, train_loss_val, val_loss, best_val_loss, datetime.datetime.now(), file=log_file, flush=True)

                model.train()

                if schedule_free:
                    optimizer.train()

            batch = femr.models.transformer.to_device(batch, torch.device('cuda'))
            optimizer.zero_grad()

            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                loss = model(**batch)[0]
                
            # del batch
            if train_loss is None:
                train_loss = loss.clone().detach()
            else:
                train_loss += loss.clone().detach()
            # print(model.state_dict())
            loss.backward()
            optimizer.step()
            # print(model.state_dic

            if not schedule_free:
                scheduler.step()
                
if __name__ == "__main__":
    main()
