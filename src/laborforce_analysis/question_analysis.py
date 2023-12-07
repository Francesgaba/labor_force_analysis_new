#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

class labor_force_participation:
    def __init__(self, df):
        """
        Initialize the LaborForceParticipationVisualization class with a DataFrame.

    
        df (pd.DataFrame): The DataFrame containing the data for visualization.
        """
        self.df = df

    def plot_lfp_by_k5(self):
        """
        Plot the Labor Force Participation Rate by the presence of children under 5.

        This method creates a binary column based on the 'k5' variable, groups the data by
        the new binary column, and calculates the mean of 'lfp_encoded'. It then plots a bar
        chart to visualize the labor force participation rate based on whether there are
        children under 5 in the household.

        """
        # Create a binary column based on 'k5' variable
        self.df['k5_binary'] = self.df['k5'].apply(lambda x: 1 if x > 0 else 0)

        # Group by the new binary column and calculate mean of 'lfp_encoded', reset index for easier plotting
        lfp_rate_by_k5 = self.df.groupby('k5_binary')['lfp_encoded'].mean().reset_index()

        # Plotting
        plt.figure(figsize=(8, 6))
        sns.barplot(x='k5_binary', y='lfp_encoded', data=lfp_rate_by_k5)

        plt.xlabel('Children under 5')
        plt.ylabel('Labor Force Participation Rate')
        plt.title('Labor Force Participation by Young Children')
        plt.xticks([0, 1], ['No Children under 5', 'At Least One Child under 5'])

        plt.show()

# In[ ]:


import matplotlib.pyplot as plt
import pandas as pd

class Couple_Education_LFP:
    """
    A class for analyzing and visualizing labor force participation rates based on the educational background of couples.

    Args:
        df (pd.DataFrame): The DataFrame containing the data for visualization.
    """

    def __init__(self, df):
        """
        Initialize the CouplesEducationLFPVisualization class.

        Args:
            df (pd.DataFrame): The DataFrame containing the data for visualization.
        """
        self.df = df

    def prepare_data(self):
        """

        This method creates several columns in the DataFrame to categorize couples based on their
        educational background, including 'BothCollege', 'WifeOnly', 'HusbandOnly', and 'Neither'.

        """
        self.df['BothCollege'] = (self.df['wc_encoded'] == 1) & (self.df['hc_encoded'] == 1)
        self.df['WifeOnly'] = (self.df['wc_encoded'] == 1) & (self.df['hc_encoded'] == 0)
        self.df['HusbandOnly'] = (self.df['wc_encoded'] == 0) & (self.df['hc_encoded'] == 1)
        self.df['Neither'] = (self.df['wc_encoded'] == 0) & (self.df['hc_encoded'] == 0)

    def plot_lfp_by_education(self):
        """
        Plot the labor force participation rate by the educational background of couples.

        This method groups the data by 'BothCollege', 'WifeOnly', 'HusbandOnly', 'Neither', 
        calculates the normalized value counts as a proportion of 'lfp_encoded' for each group, 
        and then plots a stacked bar chart to visualize the labor force participation rate
        based on the educational background of couples.

        """
        grouped = self.df.groupby(['BothCollege', 'WifeOnly', 'HusbandOnly', 'Neither'])['lfp_encoded'].value_counts(normalize=True).unstack().fillna(0)

        grouped.plot(kind='bar', stacked=True)
        plt.title('Labor Force Participation by Educational Background of Couples')
        plt.xlabel('Educational Background')
        plt.ylabel('Proportion in Labor Force')
        plt.xticks(ticks=[0, 1, 2, 3], labels=['Both College', 'Wife Only', 'Husband Only', 'Neither'], rotation=0)
        plt.legend(title='Labor Force Participation', labels=['Not Participating', 'Participating'])
        plt.show()  # Plot the chart


# In[ ]:


import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

class older_children_lfp:
    """
    A class for analyzing and visualizing labor force participation rates based on the number of older children.
    """

    def __init__(self, df):
        """
        Initializes the OlderChildrenLFPVisualization with a DataFrame.
        """
        self.df = df

    def plot_lfp_rate_by_children_number(self):
        """
        Plots the labor force participation rate by the number of older children.
        """
        # Create a grouped variable for the number of older children
        children_column = 'k618'
        grouped_column_name = 'k618_grouped'
        self.df[grouped_column_name] = self.df[children_column].apply(
            lambda x: '0' if x == 0 else ('1' if x == 1 else ('2' if x == 2 else '3+'))
        )

        # Calculate the mean labor force participation rate for each group
        target_column = 'lfp_encoded'
        lfp_rate = self.df.groupby(grouped_column_name)[target_column].mean().reset_index()

        # Plotting
        plt.figure(figsize=(8, 6))
        sns.barplot(x=grouped_column_name, y=target_column, data=lfp_rate, 
                    order=['0', '1', '2', '3+'])

        plt.xlabel('Number of Children Aged 6-18')
        plt.ylabel('Labor Force Participation Rate')
        plt.title('Labor Force Participation by Number of Older Children')
        plt.xticks(range(4), ['No Children aged 6-18', 'One Child', 'Two Children', 'Three or More Children'])

        plt.show()



# In[ ]:



import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

class Age_Group_LFP:
    """
    A class for analyzing and visualizing labor force participation rates based on age groups.

    Args:
        df (pd.DataFrame): The DataFrame containing the data for visualization.
    """

    def __init__(self, df):
        """
        Initialize the AgeGroupLFPVisualization class with a DataFrame.

        """
        self.df = df

    def create_age_group_variable(self):
        """
        Create an age group variable in the DataFrame.

        This method adds a new column 'age_group' to the DataFrame, categorizing individuals into
        age groups 'Before 45' and '45 and After' based on their age.

        """
        self.df['age_group'] = self.df['age'].apply(lambda x: 'Before 45' if x < 45 else '45 and After')

    def plot_lfp_by_age_group(self):
        """
        Plot the labor force participation rate by age group.

        This method groups the data by 'age_group', calculates the mean of 'lfp_encoded' for each group,
        and then plots a bar chart to visualize the average labor force participation rate by age group.

        """
        age_group_lfp = self.df.groupby('age_group')['lfp_encoded'].mean().reset_index()

        plt.figure(figsize=(8, 6))
        sns.barplot(x='age_group', y='lfp_encoded', data=age_group_lfp)

        plt.xlabel('Age Group')
        plt.ylabel('Average Labor Force Participation Rate')
        plt.title('Labor Force Participation by Age Group of Married Women')

        plt.show()  # Plot the chart

# In[ ]:

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.pipeline import Pipeline
import numpy as np
from sklearn.model_selection import cross_val_score


class labor_force_model:
    """
    A class for training and evaluating a logistic regression model on labor force participation data.

    This class automates the process of data preparation, model training, and evaluation. It assumes that
    the dataset is preloaded into a DataFrame and the target variable is 'lfp_encoded'.

    """

    def __init__(self,df):
        """
        Initialize the LaborForceModel class by loading the dataset and preparing features and target.
        """
        self.df = df
        self.X = self.df.drop('lfp_encoded', axis=1)
        self.y = self.df['lfp_encoded']
        self.pipeline = Pipeline([
            ('scaler', StandardScaler()),  # Feature scaling
            ('logistic_regression', LogisticRegression(class_weight='balanced'))  # Logistic regression with balanced class weight
        ])

    def train_and_evaluate(self):
        """
        Train the logistic regression model and evaluate its performance.

        The method splits the data into training and test sets, fits a logistic regression model with
        balanced class weights, and then evaluates the model's performance in terms of accuracy,
        precision, recall, and F1 score.

        The results are printed out to the console.
        """
        # Split the data into training and test sets
        X_train, X_test, y_train, y_test = train_test_split(self.X, self.y, test_size=0.3, random_state=42)

        # fit the pipeline
        self.pipeline.fit(X_train, y_train)

        # Predict on the test data
        y_pred = pipeline.predict(X_test)

        # Evaluate the model
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)

        # Print the evaluation metrics
        print(f'Model Accuracy: {accuracy}')
        print(f'Model Precision: {precision}')
        print(f'Model Recall: {recall}')
        print(f'Model F1 Score: {f1}')

    def compute_accuracy(self):
        """ Compute and print the cross-validation accuracy score. """
        scores = cross_val_score(self.pipeline, self.X, self.y, cv=5, scoring='accuracy')
        print(f'Cross-Validation Accuracy Scores: {scores}')
        print(f'Average CV Accuracy: {np.mean(scores)}')

    def compute_precision(self):
        """ Compute and print the cross-validation precision score. """
        scores = cross_val_score(self.pipeline, self.X, self.y, cv=5, scoring='precision')
        print(f'Cross-Validation Precision Scores: {scores}')
        print(f'Average CV Precision: {np.mean(scores)}')

    def compute_recall(self):
        """ Compute and print the cross-validation recall score. """
        scores = cross_val_score(self.pipeline, self.X, self.y, cv=5, scoring='recall')
        print(f'Cross-Validation Recall Scores: {scores}')
        print(f'Average CV Recall: {np.mean(scores)}')

    def compute_f1(self):
        """ Compute and print the cross-validation F1 score. """
        scores = cross_val_score(self.pipeline, self.X, self.y, cv=5, scoring='f1')
        print(f'Cross-Validation F1 Scores: {scores}')
        print(f'Average CV F1: {np.mean(scores)}')
