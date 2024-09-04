import polars as pl
labels = pl.read_parquet('../meds_transform_test/MEDS_Extract_v0.0.7_test/task_labels/mortality/in_icu/first_24h/**/*.parquet')
labels = labels.sort([pl.col('subject_id'), pl.col('prediction_time')])

labels.write_parquet('labels/mortality.parquet')
