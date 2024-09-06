import config
import meds_reader
import pickle
import collections

def get_unique_codes(subjects):
    code_counts = collections.defaultdict(int)

    for subject in subjects:
        unique_codes = set()
        for event in subject.events:
            unique_codes.add(event.code)

        for code in unique_codes:
            code_counts[code] += 1

    return code_counts


def main():
    code_counts = collections.defaultdict(int)
    with meds_reader.SubjectDatabase(config.database_path, num_threads=config.num_threads) as database:
        for codes in database.map(get_unique_codes):
            for k, v in codes.items():
                code_counts[k] += v
    
    with open('all_codes.pkl', 'wb') as f:
        pickle.dump(code_counts, f)

if __name__ == "__main__":
    main()
