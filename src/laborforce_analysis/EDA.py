#!/usr/bin/env python
# coding: utf-8

# In[3]:


import matplotlib.pyplot as plt
import seaborn as sns

class continuous_variable_visualization:
    """
    A class for visualizing continuous variables using histograms and boxplots.

    This class utilizes both Matplotlib and Seaborn libraries to create visualizations, allowing
    for a comparison between the plotting capabilities of the two libraries.

    """

    def __init__(self, df):
        """
        Initialize a ContinuousVariableVisualization object with a DataFrame and predefined continuous variables.

        Args:
            df (pd.DataFrame): The DataFrame containing the data for visualization.
        """
        self.df = df
        self.continuous_variables = ["lwg", "inc"]

    def plot_histograms_and_boxplots(self):
        """
        Plot histograms and boxplots for the specified continuous variables.

        This method iterates through the predefined continuous variables and creates a 2x2 subplot structure
        for each variable. It plots Matplotlib histograms, Seaborn histograms, Matplotlib boxplots, and Seaborn
        boxplots for each variable, allowing for a visual comparison between the two libraries.

        """
        for col in self.continuous_variables:
            fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(15, 6))

            # Matplotlib histogram
            axes[0, 0].hist(self.df[col], bins=20, edgecolor='black', alpha=0.7)
            axes[0, 0].set_title(f'Matplotlib - Distribution of {col}')

            # Seaborn histogram
            sns.histplot(self.df[col], kde=True, ax=axes[0, 1])
            axes[0, 1].set_title(f'Seaborn - Distribution of {col}')

            # Matplotlib boxplot
            axes[1, 0].boxplot(self.df[col], patch_artist=True)
            axes[1, 0].set_title(f'Matplotlib - Boxplot of {col}')

            # Seaborn boxplot
            sns.boxplot(x=self.df[col], ax=axes[1, 1])
            axes[1, 1].set_title(f'Seaborn - Boxplot of {col}')

            plt.tight_layout()
            plt.show()



# In[ ]:


import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

class Categorical_variable:
    """
    A class for visualizing categorical and discrete data in a DataFrame.

    This class provides methods to create bar chart plots using both Matplotlib and Seaborn for predefined
    categorical and discrete attributes. It creates side-by-side visualizations for comparison.
    """

    def __init__(self, df):
        """
        Initialize the Categorical_variable class with a DataFrame and predefined categorical attributes.

        Args:
            df (pd.DataFrame): The DataFrame containing the data for visualization.
        """
        self.df = df
        self.categorical_attributes = [ 'wc', 'hc', 'age', 'k5', 'k618']  # Predefine your categorical attributes here

    def plot_distributions(self):
        """
        Plot the distributions of predefined categorical and discrete variables.

        This method creates a figure with two subplots for each attribute, one using Matplotlib to generate
        a bar chart and the other using Seaborn to create a count plot. The subplots are displayed side by side.
        """
        for col in self.categorical_attributes:
            fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(20, 3))

            # Matplotlib bar chart plot
            axes[0].bar(self.df[col].value_counts().index, self.df[col].value_counts())
            axes[0].set_title(f'Matplotlib - Distribution of {col}')
            axes[0].tick_params(axis='x', rotation=90)

            # Seaborn count plot
            sns.countplot(data=self.df, x=col, ax=axes[1])
            axes[1].set_title(f'Seaborn - Distribution of {col}')
            axes[1].tick_params(axis='x', rotation=90)

            plt.tight_layout()
            plt.show()
    
    def target_variable(self):
        
        plt.figure(figsize=(5, 3))
        sns.countplot(data=self.df, x='lfp_encoded')
        plt.title('Distribution of labor force participation')
        plt.xticks(rotation=90) 
        plt.tight_layout()
        
        plt.show()
    
    def null_error_rate(self):
        most_common_class = self.df['lfp_encoded'].value_counts().idxmax()

        null_error_rate = 1 - (self.df['lfp_encoded'].value_counts().max() / self.df['lfp_encoded'].count())

        print("Most Common Class:", most_common_class)
        print("Null Error Rate:", null_error_rate)
# In[5]:


import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

class correlation_matrix:
    """
    A class for encoding categorical variables and visualizing correlation matrices.

    This class allows for the encoding of specified categorical variables in a DataFrame and
    provides methods to visualize the correlation matrix using both Matplotlib and Seaborn for comparison.

    Args:
        df (pd.DataFrame): The DataFrame containing the data for visualization.
    """

    def __init__(self, df):
        """
        Initialize the CorrelationMatrixVisualization with a DataFrame.

        Args:
            df (pd.DataFrame): The DataFrame containing the data for visualization.
        """
        self.df = df
        self.default_exclude = ['rowname'] if 'rowname' in df.columns else []

    def encode_categorical_variables(self, columns, mapping):
        """
        Encode specified categorical variables in the DataFrame.

        Args:
            columns (list): A list of column names to be encoded.
            mapping (dict): A dictionary defining the mapping for encoding.


        """
        for col in columns:
            if col in self.df.columns:
                self.df[f'{col}_encoded'] = self.df[col].map(mapping)

    def plot_correlation_matrix(self, exclude_columns=None):
        """
        Plot the correlation matrix of the DataFrame using both Matplotlib and Seaborn.

        Args:
            exclude_columns (list): A list of column names to exclude from the correlation matrix.
        """
       
        # Calculate the correlation matrix
        correlation_matrix = self.df.drop(columns=exclude_columns).corr()

        # Matplotlib correlation matrix visualization.
        plt.figure(figsize=(5, 4))
        plt.matshow(correlation_matrix, fignum=1)
        plt.colorbar()
        plt.xticks(range(len(correlation_matrix.columns)), correlation_matrix.columns, rotation=90)
        plt.yticks(range(len(correlation_matrix.columns)), correlation_matrix.columns)
        plt.title('Matplotlib - Correlation Matrix', pad=20)
        plt.show()

        # Seaborn heatmap visualization of the correlation matrix.
        plt.figure(figsize=(5, 4))
        sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f")
        plt.title('Seaborn - Correlation Matrix')
        plt.show()

