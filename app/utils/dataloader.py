import pandas as pd
import numpy as np

from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import minmax_scale


class DataLoader():
    def fit(self, dataset):
        self.dataset = dataset.copy()

    def load_data(self):
        self.dataset['Age'] = minmax_scale(self.dataset['Age'])

        le = LabelEncoder()

        le.fit(self.dataset['Gender'])
        self.dataset['Gender'] = le.transform(self.dataset['Gender'])

        le.fit(self.dataset['Previously_Insured'])
        self.dataset['Previously_Insured'] = le.transform(
            self.dataset['Previously_Insured'])

        le.fit(self.dataset['Vehicle_Age'])
        self.dataset['Vehicle_Age'] = le.transform(self.dataset['Vehicle_Age'])

        le.fit(self.dataset['Vehicle_Damage'])
        self.dataset['Vehicle_Damage'] = le.transform(
            self.dataset['Vehicle_Damage'])

        le.fit(self.dataset['Policy_Sales_Channel'])
        self.dataset['Policy_Sales_Channel'] = le.transform(
            self.dataset['Policy_Sales_Channel'])

        drop_elements = [
            'id', 'Driving_License', 'Annual_Premium', 'Region_Code', 'Vintage'
        ]

        self.dataset = self.dataset.drop(drop_elements, axis=1)

        return self.dataset
