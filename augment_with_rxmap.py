import config
import shutil
import polars as pl
import os
import pickle

if os.path.exists(config.database_path):
    shutil.rmtree(config.database_path)

shutil.copytree(config.original_database_path, config.database_path, copy_function=os.link)

code_path = os.path.join(config.database_path, 'metadata', 'codes.parquet')

code = pl.read_parquet(code_path)

code_data = code.to_dicts()

os.unlink(code_path)

with open('rxmap.pkl', 'rb') as f:
    rxmap = pickle.load(f)

code_data_map = {c['code']: c for c in code_data}

for k, v in rxmap.items():
    if k not in code_data_map:
        code_data_map[k] = {'code': k}
    
    d = code_data_map[k]
    if d.get('parent_codes', []) != []:
        print("Hope", d, v)
    d['parent_codes'] = v


df = pl.from_dicts(list(code_data_map.values()), code.schema)

print(df)

df.write_parquet(code_path)
