"""
One of FEMR's main features is utilities for helping write labeling functions.

The following are two simple labelers for inpatient mortality and long admission for MIMIC-IV.
"""

import femr.labelers
import meds_reader
import meds
import datetime

import os
import shutil
import pyarrow as pa
import pyarrow.csv as pacsv

from typing import List, Mapping
import pandas as pd
import config

class MIMICInpatientMortalityLabeler(femr.labelers.Labeler):
    def __init__(self, time_after_admission: datetime.timedelta):
        self.time_after_admission = time_after_admission

    def label(self, patient: meds_reader.Patient) -> List[femr.labelers.Label]:
        admission_ranges = set()
        death_times = set()

        for event in patient.events:
            if event.code.startswith('event_type//ADMISSION'):
                admission_ranges.add((event.time, datetime.datetime.fromisoformat(event.end)))

            if event.code == meds.death_code:
                death_times.add(event.time)

        assert len(death_times) in (0, 1)

        if len(death_times) == 1:
            death_time = list(death_times)[0]
        else:
            death_time = datetime.datetime(9999, 1, 1) # Very far in the future
            
        
        labels = []

        for (admission_start, admission_end) in admission_ranges:
            prediction_time = admission_start + self.time_after_admission
            if prediction_time >= admission_end:
                continue

            if prediction_time >= death_time:
                continue

            is_death = death_time < admission_end
            labels.append(femr.labelers.Label(patient_id=patient.patient_id, prediction_time=prediction_time, boolean_value=is_death))
        
        return labels


class MIMICLongAdmissionLabeler(femr.labelers.Labeler):
    def __init__(self, time_after_admission: datetime.timedelta, admission_length: datetime.timedelta):
        self.time_after_admission = time_after_admission
        self.admission_length = admission_length

    def label(self, patient: meds_reader.Patient) -> List[femr.labelers.Label]:
        admission_starts = dict()
        admission_ends = dict()

        for event in patient.events:
            if event.code.startswith('HOSPITAL_ADMISSION'):
                admission_starts[event.hadm_id] = event.time
            if event.code.startswith('HOSPITAL_DISCHARGE'):
                admission_ends[event.hadm_id] = event.time

        assert admission_starts.keys() == admission_ends.keys(), f'{patient} {admission_starts.keys()} {admission_ends.keys()}'

        admission_ranges = {(admission_starts[k], admission_ends[k]) for k in admission_starts.keys()}

        labels = []
        for (admission_start, admission_end) in admission_ranges:
            prediction_time = admission_start + self.time_after_admission
            if prediction_time >= admission_end:
                continue

            is_long_admission = (admission_end - admission_start) > self.admission_length

            labels.append(femr.labelers.Label(patient_id=patient.patient_id, prediction_time=prediction_time, boolean_value=is_long_admission))
        
        return labels
    

labelers: Mapping[str, femr.labelers.Labeler] = {
    'death': MIMICInpatientMortalityLabeler(time_after_admission=datetime.timedelta(hours=48)),
    'long_los': MIMICLongAdmissionLabeler(time_after_admission=datetime.timedelta(hours=48), admission_length=datetime.timedelta(days=7)),
}

def main():
    if os.path.exists('labels'):
        shutil.rmtree('labels')
    os.mkdir('labels')

    with meds_reader.PatientDatabase(config.database_path, num_threads=config.num_threads) as database:
        for label_name in config.label_names:
            labeler = labelers[label_name]
            labels = labeler.apply(database)

            labels.to_parquet(os.path.join('labels', label_name + '.parquet'), index=False)

if __name__ == "__main__":
    main()
