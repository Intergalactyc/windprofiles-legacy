import pandas as pd
from multiprocessing import Pool
import os

def analyze_directory(path: str|os.PathLike, analysis, rules: dict = None, nproc = 1, index = None, limit = None, **kwargs) -> pd.DataFrame:
    # analysis should be a function which takes a single arg (to unpack as `filepath, {rules (if not None)}, <kwargs>`) and returns a dict
    dir_path = os.path.abspath(path)
    if rules is None:
        directory = [(os.path.join(dir_path, filename), *kwargs) for filename in os.listdir(path)]
    else:
        directory = [(os.path.join(dir_path, filename), rules, *kwargs) for filename in os.listdir(path)]
    if limit is not None:
        directory = directory[:limit]
    pool = Pool(processes = nproc)
    result = pool.map(analysis, directory)
    pool.close()
    pool.join()
    print(f'Completed analysis of directory {path}')
    df = pd.DataFrame(result)
    if index is not None and index in df.columns:
        df.set_index(index, inplace = True)
        df.sort_index(ascending = True)
    return df

if __name__ == '__main__':
    print('called')
