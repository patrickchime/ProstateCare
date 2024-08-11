import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
import streamlit as st

class ProstateDataModeling:
    
    def __init__(self, model=LogisticRegression()):
        self.model = model
        self.scaler = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None

        
    def read_data(self, file_path):
        
        # Read data
        df_prostate = pd.read_csv(file_path)
        return df_prostate

    def stacked_barchart(self, df, age_group, age, target):
    
        # Define the age bins and labels
        age_bins = [40, 45, 50, 55,  60, 65,  70, 75, 80]
        age_labels = ['40-44', '45-49', '50-54', '55-59', '60-64', '65-69', '70-74', '75+']
        
        # Group ages into bins
        df[age_group] = pd.cut(df[age], bins=age_bins, labels=age_labels, right=False)
    
        # Group by 'age_group' and count occurrences of each binary value in 'Target'
        grouped_counts = df.groupby(age_group)[target].value_counts().unstack().fillna(0)
        
        # Rename columns for clarity (optional)
        grouped_counts.columns = ['No prostate', 'Prostate']
        
        # Order the age groups from low to high
        grouped_counts = grouped_counts.reindex(index=age_labels)
        
        # Create the figure and axis
        fig, ax = plt.subplots()
        
        # Make bar chart of grouped_counts
        grouped_counts.plot(kind='barh', stacked=True, color=['blue', 'orange'], edgecolor='black', ax=ax)
        
        # Set labels and title
        ax.set_xlabel('Count')
        ax.set_ylabel('Age Group')
        ax.set_title('Prostate Cancer Prevalence by Age Group')
        ax.legend(title='Target Variable')
        
        # Remove top and right spines (axes)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        
        # Return the figure
        return fig


    def split_data(self, df, target_column, test_size=0.2, random_state=42):
        """
        Splits data into train and test data
        """
        # Split data
        X = df.drop(target_column, axis=1)
        y = df[target_column]
        
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(X,y, test_size=0.2, random_state=42)

    def scale_data(self):
        # Scale data
        self.scaler = StandardScaler()
        self.X_train = self.scaler.fit_transform(self.X_train)
        self.X_test = self.scaler.transform(self.X_test)
        return self.scaler

    def train_model(self):
        # Build model
        self.model.fit(self.X_train, self.y_train)
        # Return model
        return self.model

    def evaluate_model(self):
        pred = self.model.predict(self.X_test)
        accuracy = accuracy_score(self.y_test, pred)
        return accuracy
    
    def make_predictions(self, new_data):
        if self.scaler:
            new_data = self.scaler.transform(new_data)
            # Make predictions
            pred = self.model.predict(new_data)
            return pred
   