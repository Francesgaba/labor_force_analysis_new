#!/usr/bin/env python
# coding: utf-8

# In[ ]:
import pandas as pd
import numpy as np

class labor_force_data_summary:
    """A class for loading and exploring labor force data."""

    def __init__(self, data_url):
        """
        Initialize the LaborForceDataAnalysis object with a URL to load the data from.
        """
        self.data_url = data_url
        self.df = None

    def load_data(self):
        """Load the dataset from the provided URL"""
        self.df = pd.read_csv(self.data_url, sep=",")
        print("Data loaded successfully.")


    def display_head(self):
        """display the first few rows of the dataset."""
        return self.df.head()

    def display_info(self):
        """display information about the dataset."""
        return self.df.info()

    def display_description(self):
        """display statistical summary of the dataset."""
        return self.df.describe()

    def display_missing_values(self):
        """display the count of missing values in the dataset."""
        return self.df.isnull().sum()

    def display_data_types(self):
        """display the data types of each column in the dataset."""
        return self.df.dtypes
          
