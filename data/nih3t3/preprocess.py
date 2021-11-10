import pandas as pd
from scipy.io import loadmat

# load raw data
locations_exp1 = loadmat('RNA_locations_run_1.mat')['tot']
locations_exp2 = loadmat('RNA_locations_run_2.mat')['tot']
locations = [locations_exp1, locations_exp2]
gene_names = loadmat('all_gene_Names.mat')['allNames'][:, 0]

# put everything in one large dataframe
print("Creating DataFrame. This may take a while...")
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
                temp_data.columns = ["x", "y", "z"]
                temp_data["experiment"] = experiment + 1
                temp_data["fov"] = fov
                temp_data["cell"] = cell
                temp_data["gene"] = gene_names[gene][0]

                data.append(temp_data)
data = pd.concat(data, axis=0).reset_index(drop=True)

# save to disk
data.to_feather('preprocessed_data.feather')