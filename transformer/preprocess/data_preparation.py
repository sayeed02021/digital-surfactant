import pandas as pd
import os

from sklearn.model_selection import train_test_split

import utils.file as uf
import configuration.config_default as cfgd
import preprocess.property_change_encoder as pce

SEED = 42
SPLIT_RATIO = 0.8


def get_smiles_list(file_name):
    """
    Get smiles list for building vocabulary
    :param file_name:
    :return:
    """
    pd_data = pd.read_csv(file_name, sep=",")

    print("Read %s file" % file_name)
    smiles_list = pd.unique(pd_data[['Source_Mol', 'Target_Mol']].values.ravel('K'))
    print("Number of SMILES in chemical transformations: %d" % len(smiles_list))

    return smiles_list






def split_data(input_transformations_path, LOG=None, SEED=42):
    """
    Split data into Train, Test, and Validation set , write to files
    :param input_transformations_path:L
    :return: dataframe
    

    CASE 1 : train_test_split.csv exits in the same folder 

    - Train: pairs where BOTH Source_Mol and Target_Mol are in train_test_split.csv
    - Test: pairs where BOTH Source_Mol and Target_Mol are NOT in train_test_split.csv
    - Validation: random mix from Train + Test

    CASE 2 : Random train test split 
    """
    # Load main dataset
    data = pd.read_csv(input_transformations_path, sep=",")
    if LOG:
        LOG.info("Read %s file" % input_transformations_path)
    # Locate parent folder
    parent = os.path.dirname(input_transformations_path)

    # Load smiles list
    split_filename = "train_test_split.csv"
    split_path = os.path.join(parent, split_filename)

    if os.path.exists(split_path):
        if LOG:
            LOG.info(f"train_test_split.csv found ")

        smiles_list = pd.read_csv(split_path)["smiles"].unique().tolist()

   # Train: both Source_Mol and Target_Mol in smiles_list
        mask_train = data["Source_Mol"].isin(smiles_list) & data["Target_Mol"].isin(smiles_list)
        train = data[mask_train].reset_index(drop=True)

    # Test: both Source_Mol and Target_Mol not in smiles_list
        mask_test = ~data["Source_Mol"].isin(smiles_list) & ~data["Target_Mol"].isin(smiles_list)
        test = data[mask_test].reset_index(drop=True)

    # Validation: random 10% from train+test
        combined = pd.concat([train, test]).reset_index(drop=True)
        _, validation = train_test_split(combined, test_size=0.1, random_state=SEED)


    else:
        if LOG:
            LOG.info("train_test_split.csv NOT found using random split")

        train, test = train_test_split(data, test_size=0.1, random_state=SEED)
        train, validation = train_test_split(train, test_size=0.1, random_state=SEED)


    if LOG:
        LOG.info("Train, Validation, Test: %d, %d, %d" % (len(train), len(validation), len(test)))

    # Save splits
    train.to_csv(os.path.join(parent, "train.csv"), index=False)
    test.to_csv(os.path.join(parent, "test.csv"), index=False)
    validation.to_csv(os.path.join(parent, "validation.csv"), index=False)

    return train, validation, test


def save_df_property_encoded(file_name, property_change_encoder, LOG=None):
    data = pd.read_csv(file_name, sep=",")
    for property_name in cfgd.PROPERTIES:
        if property_name == 'pCMC':
            encoder, start_map_interval = property_change_encoder[property_name]
            data['Delta_{}'.format(property_name)] = \
                data['Delta_{}'.format(property_name)].apply(lambda x:
                                                                 pce.value_in_interval(x, start_map_interval), encoder)
        elif property_name in ['Area_min', 'AW_ST_CMC']:
            data['Delta_{}'.format(property_name)] = data.apply(
                lambda row: prop_change(row['Source_Mol_{}'.format(property_name)],
                                        row['Target_Mol_{}'.format(property_name)],
                                        cfgd.PROPERTY_THRESHOLD[property_name]), axis=1)

    output_file = file_name.split('.csv')[0] + '_encoded.csv'
    LOG.info("Saving encoded property change to file: {}".format(output_file))
    data.to_csv(output_file, index=False)
    return output_file

def prop_change(source, target, threshold):
    if source <= threshold and target > threshold:
        return "low->high"
    elif source > threshold and target <= threshold:
        return "high->low"
    elif source <= threshold and target <= threshold:
        return "no_change"
    elif source > threshold and target > threshold:
        return "no_change"
