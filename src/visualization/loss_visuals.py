import pandas as pd
import matplotlib.pyplot as plt 
import seaborn as sns
from pathlib import Path
import os


def create_dir(path): 
    """
    Creates folders for images
    """
    Path(path).mkdir(parents=True, exist_ok=True)

    
def visualize_loss(data, title, xlabel, ylabel, savefig=False, path=None, img_name=None): 
    """
    Takes 1d or 2d data to visualize on line plot 
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