import pandas as pd
import numpy as np

from sklearn.preprocessing import LabelEncoder


def normalize(data):
    normalized_data = (data - data.min()) / (data.max() - data.min())
    return normalized_data


class DataLoader():
    def fit(self, dataset):
        self.dataset = dataset.copy()

    def load_data(self):
        self.dataset['Age'] = normalize(self.dataset['Age'])
        # self.dataset['Annual_Premium'] = normalize(
        #     self.dataset['Annual_Premium'])

        # n_bins_vintage = (self.dataset['Vintage'].max() -
        #                   self.dataset['Vintage'].min()) / 50
        # self.dataset['Vintage'] = pd.cut(self.dataset['Vintage'],
        #                                  n_bins_vintage)

        le = LabelEncoder()

        le.fit(self.dataset['Gender'])
        self.dataset['Gender'] = le.transform(self.dataset['Gender'])

        # le.fit(self.dataset['Driving_Lisence'])
        # self.dataset['Driving_Lisence'] = le.transform(
        #     self.dataset['Driving_Lisence'])

        # le.fit(self.dataset['Region_Code'])
        # self.dataset['Region_Code'] = le.transform(self.dataset['Region_Code'])

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

        # le.fit(self.dataset['Vintage'])
        # self.dataset['Vintage'] = le.transform(self.dataset['Vintage'])

        drop_elements = [
            'id', 'Driving_Lisence', 'Annual_Premium', 'Region_Code', 'Vintage'
        ]

        self.dataset = self.dataset.drop(drop_elements, axis=1)

        return self.dataset
