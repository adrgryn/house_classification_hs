import os
import sys
import pandas as pd
import requests

# Get the directory of the current script
script_dir = os.path.dirname(os.path.abspath(__file__))

# Construct the path to the Data directory
data_dir = os.path.join(script_dir, 'Data')

# Create the Data directory if it does not exist
if not os.path.exists(data_dir):
    os.mkdir(data_dir)

# Download data if it is unavailable.
data_path = os.path.join(data_dir, 'house_class.csv')
if not os.path.isfile(data_path):
    sys.stderr.write("[INFO] Dataset is loading.\n")
    url = "https://www.dropbox.com/s/7vjkrlggmvr5bc1/house_class.csv?dl=1"
    r = requests.get(url, allow_redirects=True)
    with open(data_path, 'wb') as f:
        f.write(r.content)
    sys.stderr.write("[INFO] Loaded.\n")

def create_df():
    df = pd.read_csv(data_path)
    return df