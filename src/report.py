import pandas as pd
import pickle
from pandas_profiling import ProfileReport

# TODO: Add feature descriptions to ProfileReport
# TODO: Log warnings thrown by ProfileReport

if __name__ == "__main__":

    # import dataset after cleaning step
    df_clean = pd.read_csv("resources/data/df_clean.csv")

    # import master dataset
    master = pd.read_csv("resources/data/master.csv")

    # generate report for dataset after cleaning step
    ProfileReport(df_clean).to_file("log/report_clean.html")

    # generate report for master dataset
    ProfileReport(master).to_file("log/report_master.html")
