import pandas as pd
import yaml
from pandas_profiling import ProfileReport

# TODO: Add feature descriptions to ProfileReport
# TODO: Log warnings thrown by ProfileReport

file_in = "master.csv"

if __name__ == "__main__":

    with open("resources/paths.yaml") as f:
        dct_paths = yaml.safe_load(f)

    df = pd.read_csv(dct_paths["data"] + "/" + file_in)

    ProfileReport(df).to_file(dct_paths["log"] + "/report" + file_in[:-4])
