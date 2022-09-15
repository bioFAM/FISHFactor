import requests
import zipfile
import io
import pandas as pd
from scipy.io import loadmat

urls = ['https://zenodo.org/record/2669683/files/seqFISH%2B_NIH3T3_point_locations.zip',
        'https://zenodo.org/record/2669683/files/ROIs_Experiment1_NIH3T3.zip',
        'https://zenodo.org/record/2669683/files/ROIs_Experiment2_NIH3T3.zip',
]

for i, url in enumerate(urls):
    r = requests.get(url)
    z = zipfile.ZipFile(io.BytesIO(r.content))
    z.extractall()

# load raw data
locations_exp1 = loadmat('RNA_locations_run_1.mat')['tot']
locations_exp2 = loadmat('RNA_locations_run_2.mat')['tot']
locations = [locations_exp1, locations_exp2]
gene_names = loadmat('all_gene_Names.mat')['allNames'][:, 0]

# put everything in one large dataframe
data = []
for experiment in range(len(locations)):
    for fov in range(locations[experiment].shape[0]):
        for cell in range(locations[experiment].shape[1]):
            print('Experiment %s, FOV %s, Cell %s' %(experiment, fov, cell))
            for gene in range(locations[experiment].shape[2]):
                # check if points exist
                if locations[experiment][fov, cell, gene].shape[1] != 3:
                    continue

                temp_data = pd.DataFrame(locations[experiment][fov, cell, gene])
                temp_data.columns = ['x', 'y', 'z']
                temp_data['experiment'] = experiment + 1
                temp_data['fov'] = fov
                temp_data['cell0'] = cell
                temp_data['gene'] = gene_names[gene][0]

                data.append(temp_data)
data = pd.concat(data, axis=0).reset_index(drop=True)

data.drop(columns=['z'], inplace=True)
data['cell'], _ = pd.factorize(
    data['experiment'].astype(str)
    + '_'
    + data['fov'].astype(str)
    + '_'
    + data['cell0'].astype(str)
    )

data.to_feather('data_preprocessed.feather')