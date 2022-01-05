import pandas as pd


def read_csv_file(filepath):
    """
    Loads data from a csv file.

    Parameters
    ----------
    filepath : str
        Path to the csv file.

    Returns
    -------
    data : pandas.DataFrame
        Dataframe containing the data.
    """
    data = pd.read_csv(filepath)
    return data