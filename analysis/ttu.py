import windprofiles.sonic as sonic
import windprofiles.preprocess as preprocess
import pandas as pd
import numpy as np
from ttu_definitions import *

WE_RULES = {
    
}

def get_datetime_from_filename(filename: str):
    DATE_STR = filename.split('_')[4]
    YEAR = int(DATE_STR[1:5])
    MONTH = int(DATE_STR[5:7])
    DAY = int(DATE_STR[7:9])
    TIME_STR = filename.split('_')[5]
    HOUR = int(TIME_STR[1:3])
    MIN = int(TIME_STR[3:5])
    START_TIME = pd.Timestamp(year = YEAR, month = MONTH, day = DAY, hour = HOUR, minute = MIN, tz = 'UTC')
    return START_TIME

def load_and_format_file(filename):
    df = pd.read_csv(filename, compression = 'gzip', header = None)
    df.rename(columns = {i : SOURCE_HEADERS[i] for i in range(120)}, inplace = True)
    df = preprocess.rename_headers(df, HEADER_MAP, True, True)
    remap = {}
    heights = set()
    for col in df.columns:
        col_type, boom_number = col.split('_')
        remap[col] = f'{col_type}_{int(HEIGHTS[int(boom_number)])}'
        heights.add(HEIGHTS[int(boom_number)])
    df.rename(columns = remap, inplace = True)
    heights_list = list(heights)
    heights_list.sort()
    return df, heights_list

def process_file(args):
    filepath, rules = args
    TIMESTAMP = get_datetime_from_filename(filepath)
    df, heights_available = load_and_format_file(filepath)
    print(heights_available)
    result = {'time' : TIMESTAMP}

    print(df)

    return result

def process_day(day, rules):
    results = sonic.analyze_directory(path = f'{SOURCE_DIRECTORY}/0{day}',
                                      analysis = process_file,
                                      rules = rules,
                                      nproc = 1,
                                      index = 'time',
                                      limit = 1)
    return results

if __name__ == '__main__':
    for i in range(1, 8):
        day_summary = process_day(day = i, rules = WE_RULES)
        print(day_summary)
        day_summary.to_csv(f'{OUTPUT_DIRECTORY}/{i}.csv')
