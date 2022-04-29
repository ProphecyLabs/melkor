import pandas as pd
import json
from pandas_profiling import ProfileReport

# TODO: Add feature descriptions to ProfileReport
# TODO: Log warnings thrown by ProfileReport

if __name__ == "__main__":

    with open("resources/paths.json", "r") as f:
        dct_paths = json.load(f)

    df_clean = pd.read_csv(dct_paths["data"]["clean"])

    master = pd.read_csv(dct_paths["data"]["clean"])

    ProfileReport(df_clean).to_file(dct_paths["log"]["report_clean"])

    ProfileReport(master).to_file(dct_paths["log"]["report_master"])
