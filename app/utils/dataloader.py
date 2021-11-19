import pandas as pd
import numpy as np

from imblearn.combine import SMOTEENN
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import minmax_scale


class DataLoader():
    def fit(self, dataset):
        self.dataset = dataset.copy()

    def load_data(self):
        self.dataset['Age'] = minmax_scale(self.dataset['Age'])
        # self.dataset['Annual_Premium'] = normalize(
        #     self.dataset['Annual_Premium'])

        # n_bins_vintage = (self.dataset['Vintage'].max() -
        #                   self.dataset['Vintage'].min()) / 50
        # self.dataset['Vintage'] = pd.cut(self.dataset['Vintage'],
        #                                  n_bins_vintage)

        le = LabelEncoder()

        le.fit(self.dataset['Gender'])
        self.dataset['Gender'] = le.transform(self.dataset['Gender'])

        # le.fit(self.dataset['Driving_License'])
        # self.dataset['Driving_License'] = le.transform(
        #     self.dataset['Driving_License'])

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
            'id', 'Driving_License', 'Annual_Premium', 'Region_Code', 'Vintage'
        ]

        self.dataset = self.dataset.drop(drop_elements, axis=1)

        return self.dataset


class Resampler():
    def fit(self, X, y):
        sampler = SMOTEENN(sampling_strategy=0.62, random_state=0)

        return sampler.fit_resample(X, y)