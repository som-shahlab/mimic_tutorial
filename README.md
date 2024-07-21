FEMR MIMIC-IV Tutorial
======================

This tutorial illustrates how to use the MOTOR foundation model with the FEMR EHR modeling library on MEDS formatted MIMIC-IV data.

This involves multiple steps, including ETLs of MIMIC data, training baselines, training a MOTOR foundation model and applying that MOTOR foundation model.

Step 1. ETLing into MEDS
------------------------

The first step of running this code requires an ETL into MEDS.

First, download the required software, meds_etl

```bash
pip install meds_etl==0.2.3
```

Then, download MIMIC-IV following the instructions on https://physionet.org/content/mimiciv/2.2/

```bash
wget -r -N -c -np --user USERNAME --ask-password https://physionet.org/files/mimiciv/2.2/
```

Finally, run the mimic ETL on that downloaded data

```
