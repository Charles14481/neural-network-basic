# Creates MLP with forward and backward passes

import graph
import node
import csv
import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
import random

class DataHandler:
    @staticmethod
    def setup_data(
        data_file,
        target,
        norm=None,
        use_cat=True,
    ):
        """Gets data from data_file, splits 80-20, and returns a tuple of train/validation input/observed datasets"""
        print("\n", "Setting up data".center(60,'-'))
        np.set_printoptions(suppress=True, precision=3, floatmode="fixed", override_repr=True)
        # Get data
        data = None
        with open(data_file, 'r') as csvfile:
            reader = csv.DictReader(csvfile)
            data = list(reader) # list of dict of info
        
        # Remove rows with n/a values
        empty = {None, '', 'n/a'}
        data = [row for row in data if all(row[key] not in empty for key in row.keys())]

        # Deal with categorical data
        cat = list()

        for key in data[0]:
            try:
                float(data[0][key])
            except Exception:
                cat.append(key)

        if use_cat:
            found = dict.fromkeys(cat, list())

            for row in data:
                for key in cat:
                    if row[key] not in found[key]:
                        found[key].append(row[key])
                    row[key] = found[key].index(row[key])
        
        else:
            for row in data:
                for key in cat:
                    del row[key]
        
        # Convert numerical data to floats
        for row in data:
            for key in row.keys():
                row[key] = float(row[key])
        
        # DataHandler.head(data, length=10, name="pre-z")

        # Normalize data
        if (norm):
            data = DataHandler.normalize(data)
        
        # Split data 80-20 and seperate actual values
        random.shuffle(data) # don't shuffle if there is correlation between date and target
        rows = len(data)
        train_data = data[:int(rows*0.8)]
        test_data = data[int(rows*0.8):]
        
        # DataHandler.head(train_data, length=10, name="train_data")

        # Seperate actual prices from input
        train_truths = np.fromiter(map(lambda row: row[target], train_data), dtype=np.float64)
        test_truths = np.fromiter(map(lambda row: row[target], test_data), dtype=np.float64)

        for row in train_data:
            del row[target]
        for row in test_data:
            del row[target]

        print(f"columns: {len(data[0].keys())}")
        print(f"data lengths: train {len(train_data)}, test {len(test_data)}")

        # Model.head(train_data)
        # Model.head(train_truths)
        # Model.head(test_data)
        # Model.head(test_truths)
        
        print("Data processed")
        print('-' * 60, "\n")

        return train_data, train_truths, test_data, test_truths
    
    @staticmethod
    def normalize(data):
        """For each column, resize values to become z scores"""
        rows = len(data)
        cols = len(data[0])
        keys = list(data[0].keys())

        # Get mean
        mean = np.zeros(cols, dtype=float)

        for row in range(rows):
            for i, val in enumerate(data[row].values()):
                mean[i] += val

        mean /= rows
        
        # Get stdev
        stdev = np.zeros(cols, dtype=float)

        stdev = np.zeros(cols, dtype=float)
        for row in range(rows):
            for i, val in enumerate(data[row].values()):
                stdev[i] += (val - mean[i]) ** 2
        
        stdev = np.sqrt(stdev / (rows - 1))

        for row in range(rows):
            for i in range(cols):
                z = (data[row][keys[i]]-mean[i])/stdev[i]
                data[row][keys[i]] = z

        print("\tmeans are:", ', '.join(f"{x:.3f}" for x in mean))
        print("\tstdevs are:", ', '.join(f"{x:.3f}" for x in stdev))

        return data
    
    @staticmethod
    def head(data, length=10, percision=3, name="data"):
        """Print first few values of list of dictionary with formatting or list"""
        length = min(length, len(data))
        print("\n", f" head of {name} length={length} ".center(40,'-'))
        for i in range(min(length, len(data))):
            if isinstance(data[0], dict):
                s = str([f"{k}: {v:.{percision}f}" for k,v in zip(data[i].keys(), data[i].values())])
            else:
                s = data[i]
            print(f"\t[{i}]:", s)
        print('-' * 40, "\n")