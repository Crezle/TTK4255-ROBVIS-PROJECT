import glob
import os
import numpy as np

def load_parameters(data_path: str,
                    datatype: str, 
                    dataset: str,
                    files: list):
    """Load parameters latest parameters from specified dataset and files.

    Args:
        data_path (str): Path to the data folder.
        dataset (str): Dataset name.
        files (list): List of files to load.
        
    Returns:
        parameters (dict): Dictionary containing the loaded parameters.
    """
    parameters = {}
    for file in files:
        print(f'Loading {file} from this run...', end=' ')
        try:
            parameters[file] = np.loadtxt(os.path.join(data_path, f'{file}.txt'))
        except FileNotFoundError as e:
            print(f'Could not load {file}. {e}.\nTrying to find older results...', end=' ')
            try:
                dirs = glob.glob(f'output/*/{datatype}/{dataset}')
                dirs.sort(key=os.path.getmtime)
                latest_dir = dirs[-1]

                data_path = os.path.join(latest_dir)
                parameters[file] = np.loadtxt(os.path.join(data_path, f'{file}.txt'))
            except FileNotFoundError as e:
                raise FileNotFoundError(f'Loading older results failed with error: {e}')
        print(f'{file} successfully loaded.')
    return parameters
