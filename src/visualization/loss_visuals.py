import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import os


def create_dir(path):
    """Creates directories in path if it doesn't exist. Used to store saved visualization plots.

    Args:
        path (str): Relative path to create directories for

    Returns:
        None
    """

    Path(path).mkdir(parents=True, exist_ok=True)


def visualize_loss(data, title, xlabel, ylabel, savefig=False, path=None, img_name=None):
    """ Creates overlapping line plots for multiple sources of losses.

    Args:
        data (list or matrix): Losses to plot in a line plot, different sources of losses must be in different lists or rows
        title (str): Title of plot
        xlabel (str): x-axis label of plot
        ylabel (str): y-axis label of plot
        savefig (bool, optional): If *True*, saves a copy of the graph in a png file, must also specify path and img_name
        path (str, optional): Path to store saved plot image in, doesn't work if savefig = *False*
        img_name (str, optional): Name of saved plot image, doesn't work if savefig = *False*

    Returns:
        None
    """

    sns.set()
    for i in data:
        plt.plot(i)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)

    if savefig:
        path = os.path.join(path, img_name)
        plt.savefig(path)
        print("Plot saved at " + path)

    plt.show()