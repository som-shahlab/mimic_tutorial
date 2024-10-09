import meds_reader
import config
import pickle


database = meds_reader.SubjectDatabase(config.database_path)

offset = 217

with open('bad_batch_0.pkl', 'rb') as f:
    batch = pickle.load(f)['batch']

label_index = batch['transformer']['label_indices'][0, offset]

print(label_index)

subject_id = batch['subject_ids'][0, label_index]
time = batch['transformer']['timestamps'][0, label_index].cpu().numpy().astype('datetime64[s]')

print(subject_id, time)

subject = database[subject_id]

for event in subject.events:
    print(dict(event))