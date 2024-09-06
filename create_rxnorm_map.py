import pickle
import json
import polars as pl
import tqdm

a = pickle.load(open('all_codes.pkl', 'rb'))

drug_codes = {c for c in a if c.startswith('MEDICATION//')}
# drug_codes = list(drug_codes)[:100]


parts = set()

part_map = {}

for drug_code in drug_codes:
    a = drug_code.split('//')[1:]
    part_map[drug_code] = [('MEDICATION_PART//' + v) for v in a]
    parts.update(a)

import urllib.request

event_txt = set(pl.read_csv('event_txt.csv').to_dict()['event_txt'])
event_txt.update(('', 'START', 'STOP'))

for part in tqdm.tqdm(parts):
    # This is because of a bug where in the ETL where medications and medication types are mixed
    if part in event_txt:
        continue

    params = urllib.parse.urlencode({'term': part})
    # print(params)
    with urllib.request.urlopen('https://rxnav.nlm.nih.gov/REST/approximateTerm.json?' + params) as f:
        result = json.load(f)
        # print(result)
        candidates = result['approximateGroup'].get('candidate', [])
        candidates = [c for c in candidates if c['source'] == 'RXNORM']
        if len(candidates) == 0:
            print("Could not map", part, params)
            continue
        best = candidates[0]['rxcui']


        with urllib.request.urlopen('https://rxnav.nlm.nih.gov/REST/rxcui/' + best + '/allrelated.json') as f:
            related = json.load(f)
            groups = related['allRelatedGroup']['conceptGroup']
            in_groups = [g for g in groups if g['tty'] == 'IN' and 'conceptProperties' in g]
            if len(in_groups) == 0:
                print("Could not get ingredient", part, best)
                continue
            else:
                assert len(in_groups) == 1
                in_group = in_groups[0]['conceptProperties']
                ingredients = [i['rxcui'] for i in in_group]
                # print("Ingredients", part, best, ingredients)

        part_map['MEDICATION_PART//' + part] = [('RxNorm/' +i) for i in ingredients]

with open('rxmap.pkl', 'wb') as f:
    pickle.dump(part_map, f)
