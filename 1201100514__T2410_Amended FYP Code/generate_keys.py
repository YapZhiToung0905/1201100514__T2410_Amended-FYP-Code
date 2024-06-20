import pickle
from pathlib import Path

import streamlit_authenticator as stauth

# Create a list of admin
name = ["Sean Yap"]

# Define respective name of admin, which will be used for the authentication
adminName = ["Sean"]

# Get the password
password = ["XXX"]

# Use hasher module to convert the plain text passwords to hashed password
hashed_password = stauth.Hasher(password).generate()

# Define file path to pickle file
file_path = Path(__file__).parent / "hashed_pw.pkl"

# open up the file and write binary mode and dump password into it
with file_path.open("wb") as file:
    pickle.dump(hashed_password, file)