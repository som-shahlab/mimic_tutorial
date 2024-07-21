FEMR MIMIC-IV Tutorial
======================

This tutorial illustrates how to use the MOTOR foundation model with the FEMR EHR modeling library on MEDS formatted MIMIC-IV data.

This involves multiple steps, including ETLs of MIMIC data, training baselines, training a MOTOR foundation model and applying that MOTOR foundation model.

The end result is that MOTOR provides better models than LightGBM, which in turn provides better models than logistic regression.

|  | Logistic Regression AUROC | LightGBM AUROC | MIMIC MOTOR AUROC |
| :---- | :---- | :---- | :---- |
| Inpatient Mortality | 0.833 | 0.880 | 0.937 |
| Long Admission | 0.710 | 0.760 | 0.820 | 


Step 1. ETLing into MEDS
------------------------

The first step of running this code requires an ETL into MEDS.

First, download the required software, meds_etl

```bash
pip install meds_etl[cpp]==0.2.3
```

Then, download MIMIC-IV following the instructions on https://physionet.org/content/mimiciv/2.2/

```bash
wget -r -N -c -np --user USERNAME --ask-password https://physionet.org/files/mimiciv/2.2/
```

Finally, run the mimic ETL on that downloaded data

```bash
meds_etl_mimic physionet.org/files/mimic-iv/ mimic-iv-meds --num_proc 16 --num_shards 16 --backend cpp
```

Step 2. Converting into meds_reader
------------------------

The FEMR library uses the [meds_reader](https://github.com/EthanSteinberg/meds_reader) utility for processing MEDS data. This requires a second preprocessing step

```bash
pip install meds_reader==0.0.6
```

```bash
meds_reader_convert mimic-iv-meds mimic-iv-meds-reader --num_threads 16
```

Step 3. Downloading Athena
-------------------------

FEMR uses OHDSI's Athena tool for ontology processing. Go to https://athena.ohdsi.org/ and download the folder.

You can create an account for free.

Note: Make sure to run the CPT4 fixer script in the Athena download before continuing!

Update config.py within this repository to the path of your Athena download.


Step 3. Installing FEMR
------------------------

Finally, you need to install FEMR.

This code currently uses a branch of FEMR, mimic_tutorial, in order to function. However, soon these changes will be merged upstream into femr 2.4

```bash
git clone https://github.com/som-shahlab/femr
git checkout mimic_tutorial
pip install -e .
pip install xformers
```

Make sure to also install the correct gpu enabled version of PyTorch


Step 4. Generating Splits and Preparing For Pretraining
------------------------

This code uses a global patient split for correctness, with 85% for training and hyperparameter tuning and 15% for the test set.

We have a single script, prepare_motor that generates these splits and then training things like tokenizers to prepare for pretraining

```bash
python prepare_motor.py
```

Step 5. Generate Labels
------------------------

We use FEMR's built-in labeling tools to define two prediction tasks: long length of stay (7 days or more) and inpatient mortality. Both predictions are made 48 hours after admission.

```bash
python generate_labels.py
```


Step 6. Generate Tabular Features
------------------------

We use FEMR's built-in labeling tools to define two prediction tasks: long length of stay (7 days or more) and inpatient mortality. Both predictions are made 48 hours after admission.

```bash
python generate_tabular_features.py
```

Step 7. Train Tabular Baselines
------------------------

We can then train baselines on those labels and tabular features. We train two baselines in particular, LightGBM and logistic regresison.

```bash
python train_baselines.py
```

Step 8. Pretrain MOTOR
------------------------

Pretraining MOTOR on MIMIC-IV is pretty fast and should take a day at most on an A100.
You could probably also train on smaller GPUs, even 16GB but that might require some hyperparameter tweaks.

```bash
python pretrain_motor.py 1e-4 # Use a learning rate of 1e-4
```

When the model appears to have converged (after roughly 70,000 steps seems good enough for my experiments), copy the checkpoint to the `motor_model` directory.

```bash
cp -r tmp_trainer_1e-4/checkpoint-68000 motor_model
```


Step 9. Generate MOTOR Embedddings
------------------------

We can then use MOTOR as an embedding model to obtain patient representations

```bash
python generate_motor_features.py
```

Step 10. Train Logistic Regression On MOTOR Embeddings
------------------------

Finally we can use MOTOR by training a linear head (aka a logistic regression model) on top of the frozen MOTOR embeddings to solve our prediction tasks.

```bash
python finetune_motor.py
```
