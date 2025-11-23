import pickle

with open("metadata_system_prompts.pkl", "rb") as f:
    data = pickle.load(f)

# Print nicely
from pprint import pprint
pprint(data)
