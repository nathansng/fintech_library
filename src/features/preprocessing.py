import pandas as pd
import ast
import torch


def preprocess_trends(trends, points, device, feature_cfg):
    """Takes in trends and points and returns training, validation, and test sets
    from samples from the data

    Args:
        trends (tf tensor): Data to extract trends from
        points (tf tensor): Tensor of data points
        device (object): Device for computations
        feature_cfg (dict): Hyperparameters

    Returns:
        tf tensor: Trend data
    """
    trend_X, trend_y = extract_data(trends, **feature_cfg)
    points_X = points[feature_cfg['num_input']+feature_cfg['num_output']:]

    trend_data = train_valid_test_split(trend_X, trend_y, device=device)
    point_data = train_valid_test_split(points_X, device=device)

    # Reshape trend outcomes for models
    y_labels = [i for i in trend_data.data_labels if i.startswith("y_")]
    for label in y_labels:
        trend_data.data[label][0] = trend_data.data[label][0].reshape(trend_data.data[label][0].shape[0], 2)

    trend_data.merge(point_data)

    return trend_data


def pad_data(data):
    """ Pad all rows with 0's to match longest row

    Args:
        data (Pandas DataFrame): Data to pad

    Returns:
        Pandas DataFrame: Padded data
    """
    max_len = data.apply(len).max()

    pad_row = lambda x: ([0] * (max_len - len(x))) + x
    padded_data = data.apply(pad_row)
    return padded_data


def convert_data_points(data):
    """ Converts DataFrame to tensor

    Args:
        data (Pandas DataFrame): [duration, slope, data_points]

    Returns:
        list[tf tensor]: (trends) contains trend duration, slope and (data_pts) contains original data points
    """

    # Extract trends data
    trends = data[['trend_duration', 'trend_slope']]
    trends = torch.tensor(trends.values)

    # Extract data points
    data_pts = data['trend_points']
    if type(data_pts[0]) == str:
        data_pts = data_pts.apply(ast.literal_eval)
    data_pts = pad_data(data_pts)
    data_pts = torch.tensor(data_pts)

    return trends, data_pts


def extract_data(data, num_input, num_output):
    """ Extracts m sequential data to use to predict n next data

    Args:
        data (tf tensor): Data to extract
        num_input (int): Number of input data
        num_output (int): Number of output data

    Returns:
        list[tf tensor]: input and output data

    """
    num_rows = data.shape[0] - num_input - num_output
    if len(data.shape) == 2:
        input_data = torch.zeros(num_rows, num_input, data.shape[-1])
        output_data = torch.zeros(num_rows, num_output, data.shape[-1])
    else:
        input_data = torch.zeros(num_rows, num_input)
        output_data = torch.zeros(num_rows, num_output)

    for i in range(num_rows):
        input_data[i] = (data[i:i+num_input])
        output_data[i] = (data[i+num_input:i+num_input+num_output])
    return input_data, output_data


def train_valid_test_split(X, y=None, props=None, device=None):
    """ Separates data into train, validation, and test sets

    Args:
        X (tf tensor): Feature data
        y (tf tensor): Target data
        props (list[int]): Train/Val splits (Test split is inferred from these numbers)
        device (object): Device for computation

    Returns:
        SplitData object: Contains train/val/test splits of the data

    """
    if not props:
        props = (0.5, 0.25)
    elif len(props) != 3:
        print("Wrong number of parameters")
        return None

    train_size = int(X.shape[0] * props[0])
    valid_size = int(X.shape[0] * props[1])

    data = SplitData()

    data.add("X_train", X[:train_size].to(device))
    data.add("X_valid", X[train_size: (train_size + valid_size)].to(device))
    data.add("X_test", X[(train_size + valid_size):].to(device))

    if y != None:
        data.add("y_train", y[:train_size].to(device))
        data.add("y_valid", y[train_size: (train_size + valid_size)].to(device))
        data.add("y_test", y[(train_size + valid_size):].to(device))

    return data


class SplitData:
    """
    Stores train, valid, and test data for easy access
    """
    def __init__(self, data_labels = ["X_train", "y_train", "X_valid", "y_valid", "X_test", "y_test"]):
        self.data_labels = data_labels
        self.data = {i: [] for i in self.data_labels}

    def add(self, label, data):
        """ Adds data to SplitData object

        Args:
            label (string): Data label type (ex: "X_train", "X_valid" etc.)
            data (tf tensor): Data to add

        Returns:
            None
        """
        if type(data) == list:
            for i in data:
                self.data[label].append(i)
        else:
            self.data[label].append(data)

    def get(self, label):
        """ Retrieves data from SplitData object

        Args:
            label (string): Data label type (ex: "X_train", "X_valid" etc.)

        Returns:
            tf tensor: Data corresponding to label
        """
        if len(self.data[label]) > 1:
            return self.data[label]
        return self.data[label][0]

    def merge(self, data):
        """ Merges data in other SplitData object with current SplitData object

        Args:
            data (SplitData object): SplitData object

        Returns:
            None
        """
        for label in data.data:
            if label in self.data_labels:
                for i in data.data[label]:
                    self.data[label].append(i)
