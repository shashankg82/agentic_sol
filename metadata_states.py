import pickle

with open("metadata_states.pkl", "rb") as f:
    data = pickle.load(f)

# Print nicely
from pprint import pprint
pprint(data)
