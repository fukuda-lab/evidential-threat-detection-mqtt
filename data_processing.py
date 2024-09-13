from config import ConfigFullstack
import logging
from constants import PHIST_BIN_COUNT
import numpy as np
import pandas as pd
from functools import partial
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import itertools
from datetime import datetime, timedelta
from functools import partial

def process_ppi(row):
    """
    Processes a single row of PPI (first N packet information, usually N=30) data to extract and transform packet-related metrics.

    Parameters:
    - row (pd.Series): A pandas Series object representing a single row of a dataframe. Expected to contain the following fields:
        - PPI_PKT_LENGTHS: A string representation of packet lengths, separated by "|" and enclosed in square brackets.
        - PPI_PKT_DIRECTIONS: A string representation of packet directions, separated by "|" and enclosed in square brackets.
        - PPI_PKT_TIMES: A string representation of packet timestamps in ISO format, separated by "|" and enclosed in square brackets.
        - PPI_PKT_FLAGS: A string representation of packet flags, separated by "|" and enclosed in square brackets.

    Returns:
    - Tuple containing:
        - np.array: An array of time differences between consecutive packets, with a length of 30. Missing values are filled with zeros.
        - np.array: An array of packet directions, with a length of 30. Missing values are filled with zeros.
        - np.array: An array of packet sizes, with a length of 30. Missing values are filled with zeros.
        - np.array: An array of packet flags, with a length of 30. Missing values are filled with zeros.
        - int: The total number of packets.
        - float: The duration between the first and last packet in seconds.
        - int: The number of round trips, calculated based on direction changes from positive to negative.

    Notes:
    - If PPI_PKT_LENGTHS is an empty list (represented as "[]"), the function returns arrays of zeros for time differences, directions, sizes, and flags, along with zeros for the total number of packets, duration, and number of round trips.
    - The function uses the datetime and timedelta modules to calculate time differences and duration.
    """
    if row.PPI_PKT_LENGTHS == "[]":
        return np.zeros(30),np.zeros(30),np.zeros(30),np.zeros(30), 0, 0, 0
    sizes = list(map(int, row.PPI_PKT_LENGTHS[1:-1].split("|")))
    directions = list(map(int, row.PPI_PKT_DIRECTIONS[1:-1].split("|")))
    times = list(map(datetime.fromisoformat, row.PPI_PKT_TIMES[1:-1].split("|")))
    flags = list(map(lambda x: int(x), row.PPI_PKT_FLAGS[1:-1].split("|")))
    time_differences = [int((e - s) / timedelta(milliseconds=1)) for s, e in zip(times, times[1:])]
    time_differences.insert(0, 0)
    ppi_roundtrips = len(list(itertools.groupby(itertools.dropwhile(lambda x: x < 0, directions), key=lambda i: i > 0))) // 2
    ppi_len = len(sizes)
    ppi_duration = (times[-1] - times[0]).total_seconds()
    time_differences = time_differences + [0] * (30 - len(time_differences))
    directions = directions + [0] * (30 - len(directions))
    sizes = sizes + [0] * (30 - len(sizes))
    flags = flags + [0] * (30 - len(flags))
    return np.array(time_differences), np.array(directions), np.array(sizes), np.array(flags), ppi_len, ppi_duration, ppi_roundtrips

def process_phist(row):
    """
    Processes a single row of PHIST (Packet Histograms) data to extract and transform packet history metrics.

    Parameters:
    - row (pd.Series): A pandas Series object representing a single row of a dataframe. Expected to contain the following fields:
        - S_PHISTS_SIZES: A string representation of source packet sizes, separated by "|" and enclosed in square brackets.
        - D_PHISTS_SIZES: A string representation of destination packet sizes, separated by "|" and enclosed in square brackets.
        - S_PHISTS_IPT: A string representation of inter-packet times for source packets, separated by "|" and enclosed in square brackets.
        - D_PHISTS_IPT: A string representation of inter-packet times for destination packets, separated by "|" and enclosed in square brackets.

    Returns:
    - Tuple containing four numpy arrays:
        - np.array: An array of source packet sizes.
        - np.array: An array of destination packet sizes.
        - np.array: An array of source inter-packet times.
        - np.array: An array of destination inter-packet times.

    Notes:
    - If any of the input fields (S_PHISTS_SIZES, D_PHISTS_SIZES, S_PHISTS_IPT, D_PHISTS_IPT) is an empty list (represented as "[]"), the function returns four arrays of zeros.
    - The function assumes that the number of elements in each of the input fields is equal and matches the predefined PHIST_BIN_COUNT.
    - It raises an AssertionError if the lengths of the input fields do not match or do not equal PHIST_BIN_COUNT.
    """
    if row.S_PHISTS_SIZES == "[]" or row.D_PHISTS_SIZES == "[]" or row.S_PHISTS_IPT == "[]" or row.D_PHISTS_IPT == "[]":
        return np.zeros(8), np.zeros(8), np.zeros(8), np.zeros(8)
    phist_src_sizes = list(map(int, row.S_PHISTS_SIZES[1:-1].split("|")))
    phist_dst_sizes = list(map(int, row.D_PHISTS_SIZES[1:-1].split("|")))
    phist_src_ipt = list(map(int, row.S_PHISTS_IPT[1:-1].split("|")))
    phist_dst_ipt = list(map(int, row.D_PHISTS_IPT[1:-1].split("|")))
    assert len(phist_src_sizes) == len(phist_dst_sizes) == len(phist_src_ipt) == len(phist_dst_ipt) == PHIST_BIN_COUNT
    return np.array(phist_src_sizes), np.array(phist_dst_sizes), np.array(phist_src_ipt), np.array(phist_dst_ipt)

def add_tcp_flags(df: pd.DataFrame) -> pd.DataFrame:
    """
    Adds TCP flags to the dataframe based on the TCP_FLAGS and TCP_FLAGS_REV columns.

    Parameters:
    - df (pd.DataFrame): The input dataframe.

    Returns:
    - pd.DataFrame: The modified dataframe with added TCP flags.

    TCP Flags:
    - CWR (Congestion Window Reduced): Indicates that the sender received a TCP segment with the ECE flag set and reduced its congestion window.
    - ECE (ECN-Echo): Indicates that the sender received a TCP segment with the ECN (Explicit Congestion Notification) flag set.
    - URG (Urgent): Indicates that the Urgent pointer field is significant.
    - ACK (Acknowledgment): Indicates that the Acknowledgment field is significant.
    - PSH (Push): Pushes the buffered data to the receiving application.
    - RST (Reset): Resets the connection.
    - SYN (Synchronize): Synchronizes sequence numbers to initiate a connection.
    - FIN (Finish): Indicates the end of data transmission.

    """
    df["FLAG_CWR"] = (df.TCP_FLAGS & 128) != 0
    df["FLAG_CWR_REV"] = (df.TCP_FLAGS_REV & 128) != 0
    df["FLAG_ECE"] = (df.TCP_FLAGS & 64) != 0
    df["FLAG_ECE_REV"] = (df.TCP_FLAGS_REV & 64) != 0
    df["FLAG_URG"] = (df.TCP_FLAGS & 32) != 0
    df["FLAG_URG_REV"] = (df.TCP_FLAGS_REV & 32) != 0
    df["FLAG_ACK"] = (df.TCP_FLAGS & 16) != 0
    df["FLAG_ACK_REV"] = (df.TCP_FLAGS_REV & 16) != 0
    df["FLAG_PSH"] = (df.TCP_FLAGS & 8) != 0
    df["FLAG_PSH_REV"] = (df.TCP_FLAGS_REV & 8) != 0
    df["FLAG_RST"] = (df.TCP_FLAGS & 4) != 0
    df["FLAG_RST_REV"] = (df.TCP_FLAGS_REV & 4) != 0
    df["FLAG_SYN"] = (df.TCP_FLAGS & 2) != 0
    df["FLAG_SYN_REV"] = (df.TCP_FLAGS_REV & 2) != 0
    df["FLAG_FIN"] = (df.TCP_FLAGS & 1) != 0
    df["FLAG_FIN_REV"] = (df.TCP_FLAGS_REV & 1) != 0

    flag_columns = [col for col in df.columns if col.startswith("FLAG")]
    df[flag_columns] = (df[flag_columns] != 0).astype(int)

    return df

def get_normalized_matrix(matrix: np.ndarray, scaler=None):
    """
    Normalize a matrix using a scaler.

    Parameters:
    - matrix (np.ndarray): The input matrix to be normalized.
    - scaler (object, optional): The scaler object to be used for normalization. If None, a new MinMaxScaler object will be created.

    Returns:
    - Tuple containing:
        - np.ndarray: The normalized matrix.
        - object: The scaler object used for normalization.

    """
    flatten_matrix = matrix.reshape(-1, 1)
    if scaler is None:
        scaler = MinMaxScaler(feature_range=(0, 1))
        return scaler.fit_transform(flatten_matrix).reshape(matrix.shape), scaler
    return scaler.transform(flatten_matrix).reshape(matrix.shape), scaler

def get_ppi_matrices(df: pd.DataFrame, train_indices, val_indices, test_indices):
    """
    Get the PPI matrices and split them into train, validation, and test sets.

    Args:
        df (pd.DataFrame): The input DataFrame containing the PPI matrices.
        train_indices (list): The indices of the training samples.
        val_indices (list): The indices of the validation samples.
        test_indices (list): The indices of the test samples.

    Returns:
        tuple: A tuple containing the normalized PPI matrices for the train, validation, and test sets.

    """
    ppi_size_matrix = np.stack(df["PPI_SIZE"])
    ppi_iat_matrix = np.stack(df["PPI_IAT"])
    ppi_dir_matrix = np.stack(df["PPI_DIR"])
    ppi_flag_matrix = np.stack(df["PPI_FLAG"])
    train_ppi_size_matrix = ppi_size_matrix[train_indices]
    train_ppi_iat_matrix = ppi_iat_matrix[train_indices]
    train_ppi_dir_matrix = ppi_dir_matrix[train_indices]
    train_ppi_flag_matrix = ppi_flag_matrix[train_indices]
    val_ppi_size_matrix = ppi_size_matrix[val_indices]
    val_ppi_iat_matrix = ppi_iat_matrix[val_indices]
    val_ppi_dir_matrix = ppi_dir_matrix[val_indices]
    val_ppi_flag_matrix  = ppi_flag_matrix[val_indices]
    test_ppi_size_matrix = ppi_size_matrix[test_indices]
    test_ppi_iat_matrix  = ppi_iat_matrix[test_indices]
    test_ppi_dir_matrix  = ppi_dir_matrix[test_indices]
    test_ppi_flag_matrix  = ppi_flag_matrix[test_indices]
    train_ppi = np.stack([train_ppi_size_matrix, train_ppi_iat_matrix, train_ppi_dir_matrix, train_ppi_flag_matrix], axis=1)
    val_ppi = np.stack([val_ppi_size_matrix, val_ppi_iat_matrix, val_ppi_dir_matrix, val_ppi_flag_matrix], axis=1)
    test_ppi = np.stack([test_ppi_size_matrix, test_ppi_iat_matrix, test_ppi_dir_matrix, test_ppi_flag_matrix], axis=1)
    return train_ppi, val_ppi, test_ppi

def flatten_columns(df, columns):
    """
    Flattens the specified columns in a DataFrame by expanding them into separate columns.

    Args:
        df (pandas.DataFrame): The DataFrame to flatten.
        columns (list): A list of column names to flatten.

    Returns:
        pandas.DataFrame: The DataFrame with flattened columns.

    Example:
        >>> df = pd.DataFrame({'A': [1, 2, 3], 'B': [[4, 5], [6, 7], [8, 9]]})
        >>> flatten_columns(df, ['B'])
           A  B_0  B_1
        0  1    4    5
        1  2    6    7
        2  3    8    9
    """
    for col in columns:
        expanded_cols = pd.DataFrame(df[col].tolist(), index=df.index)
        expanded_cols = expanded_cols.rename(columns=lambda x: f'{col}_{x}')
        df = pd.concat([df, expanded_cols], axis=1)
    df.drop(columns=columns, inplace=True)
    return df



def get_dataset(dataset_config: ConfigFullstack) -> pd.DataFrame:
    """
    Process the CSV dataset according to the given configuration.

    Args:
        dataset_config (DatasetConfig): The configuration for the dataset.

    Returns:
        pd.DataFrame: The processed dataset as a pandas DataFrame.
    """
    prefix = f"{dataset_config.dataset_folder}/{dataset_config.dataset_name}"
    if dataset_config.dataset_load_if_exists:
        logging.info(f"Attempting to load {dataset_config.dataset_name}.preprocessed.pkl")
        try:
            df = pd.read_pickle(f"{prefix}.preprocessed.pkl")
            logging.info(f"Loaded {prefix}.preprocessed.pkl")
            logging.info(f"With columns: {df.columns}")
            logging.info(f"With shape: {df.shape}")
            return df
        except Exception as e:
            logging.error(f"Could not load {dataset_config.dataset_name}.preprocessed.pkl: {e}")
            logging.info(f"Could not load {dataset_config.dataset_name}.preprocessed.pkl, processing the dataset from scratch.")
    df = pd.read_csv(f"{prefix}.csv")
    logging.info(f"Loaded {prefix}.csv")
    logging.info(f"With columns: {df.columns}")
    logging.info(f"With shape: {df.shape}")

    # convert the time columns to duration in seconds
    df["DURATION"] = (pd.to_datetime(df.TIME_LAST) - pd.to_datetime(df.TIME_FIRST)) / pd.Timedelta(seconds=1)
    process_ppi_fn = partial(process_ppi)
    df[["PPI_IAT", "PPI_DIR", "PPI_SIZE", "PPI_FLAG", "PPI_LEN", "PPI_DURATION", "PPI_ROUNDTRIPS"]] = df.apply(process_ppi_fn, axis=1, result_type="expand")
    df[["PHIST_SRC_SIZES", "PHIST_DST_SIZES", "PHIST_SRC_IPT", "PHIST_DST_IPT"]] = df.apply(process_phist, axis=1, result_type="expand")
    df = add_tcp_flags(df)
    df = flatten_columns(df, ["PHIST_SRC_SIZES", "PHIST_DST_SIZES", "PHIST_SRC_IPT", "PHIST_DST_IPT"])
    df.reset_index(drop=True, inplace=True)
    if dataset_config.dataset_save:
        df.to_pickle(f"{prefix}.preprocessed.pkl")
        logging.info(f"Saved {prefix}.preprocessed.pkl")
    return df
