import pickle

with open("metadata_playbook.pkl", "rb") as f:
    data = pickle.load(f)

# Print nicely
from pprint import pprint
pprint(data)
