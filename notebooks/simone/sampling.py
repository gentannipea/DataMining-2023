import pandas as pd
import numpy as np
import random

# df_fields_subset = subset of the fields to sample from
# num_samples = number of samples needed
# replacement = True -> sampling with replacement is performed
# replacement = False -> sampling without replacement is performed
# return: a DataFrame containing the sampled points
def sample(df_fields_subset: pd.core.frame.DataFrame, num_samples=1, replacement=True):
    num_instances = df_fields_subset.shape[0]
    samples_indexes = []
    if(not replacement):
        samples_indexes = random.sample(range(0, num_instances), num_samples)
    else:
        for i in range(0, num_samples):
            samples_indexes.append(random.randint(0, num_instances-1))
    return df_fields_subset.iloc[samples_indexes]


# Read the data
INCIDENTS = '../../dataset/data-raw/incidents.csv'
POVERTYYEAY = '../../dataset/data-raw/povertyByStateYear.csv'
STATEDISHOUSE = '../../dataset/data-raw/year_state_district_house.csv'

incidents = pd.read_csv(INCIDENTS)
poverty = pd.read_csv(POVERTYYEAY)
state_district_house = pd.read_csv(STATEDISHOUSE)

incidents['state'] = incidents['state'].str.upper()
poverty['state'] = poverty['state'].str.upper()
state_district_house['state'] = state_district_house['state'].str.upper()

print(sample(df_fields_subset=incidents[['n_participants', 'n_males']], num_samples=3, replacement=True))