import pickle
from pathlib import Path
import sys
import os
import pandas as pd
import numpy as np
import requests
from bs4 import BeautifulSoup
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
import plotly.graph_objects as go
import streamlit_authenticator as stauth
import plotly.express as px
from sklearn.linear_model import LinearRegression
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.service import Service
from webdriver_manager.chrome import ChromeDriverManager
import time
from sklearn.metrics import mean_absolute_error, mean_squared_error
from surprise.reader import Reader
from surprise import Dataset
from surprise import KNNBasic
from surprise import SVD
from surprise import accuracy
from surprise.model_selection import train_test_split
from collections import defaultdict
from surprise.model_selection.validation import cross_validate
from PIL import Image
from pivottablejs import pivot_ui
from st_aggrid import AgGrid, GridOptionsBuilder, GridUpdateMode
import extra_streamlit_components as stx
import plotly.graph_objects as go
import plotly.express as px
from streamlit_plotly_events import plotly_events
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction import text
from surprise.prediction_algorithms.predictions import Prediction
from sklearn.metrics import precision_recall_fscore_support

st.set_page_config(page_title = "Admin", page_icon = ":bust_in_silhouette:", layout = "wide")

# --- ADMIN AUTHENTICATION ---
name = ["Yap Zhi Toung"]
adminName = ["Sean"]

# Load hashed password
file_path = Path(__file__).parent / "hashed_pw.pkl"
with file_path.open("rb") as file:
    hashed_password = pickle.load(file)

authenticator = stauth.Authenticate(name, adminName, hashed_password, "homePage", "abcdef", cookie_expiry_days=30)

name, authentication_status, adminName = authenticator.login("Admin Login", "main")

if authentication_status == False: st.error("Invalid name or password. Please try again!")
if authentication_status == None: st.warning("Please enter your name and password.")

if authentication_status:
    # --- READ DATASET ---
    # Read data as a data frame.
    df = pd.read_csv('Reviews.csv')
    df =  df.iloc[:2300]

    #Scrape function and updated the csv files
    # def scrape_product_names_selenium(asins):
    #     options = webdriver.ChromeOptions()
    #     options.add_argument('--headless')
    #     driver = webdriver.Chrome(service=Service(ChromeDriverManager().install()), options=options)

    #     product_mapping = {}
    #     for asin in asins:
    #         url = f"https://www.amazon.com/dp/{asin}"
    #         driver.get(url)
    #         time.sleep(5)

    #         try:
    #             product_title = driver.find_element(By.ID, "productTitle").text
    #             product_mapping[asin] = product_title
    #         except Exception as e:
    #             print(f"Error fetching product title for {asin}: {e}")
    #             product_mapping[asin] = asin

    #     driver.quit()
    #     return product_mapping

    # def save_product_mapping(product_mapping, csv_path='product_names.csv'):
    #     df = pd.DataFrame(list(product_mapping.items()), columns=['ProductId', 'ProductName'])
    #     df.to_csv(csv_path, index=False)

    def load_product_mapping(csv_path='Product_Data.csv'):
        if os.path.exists(csv_path):
            df = pd.read_csv(csv_path)
            return dict(zip(df['ProductId'], df['ProductName']))
        return {}

    # Assuming df is the DataFrame that contains the ProductId column
    df_new = pd.DataFrame({'ProductId': ['B001E4KFG0', 'B00813GRG4', 'B000LQOCH0', 'B000UA0QIQ', 'B006K2ZZ7K', 'B000E7L2R4', 'B00171APVA', 'B0001PB9FE', 'B0009XLVG0', 'B001GVISJM', 'B00144C10S', 'B0001PB9FY', 'B003F6UO7K', 'B001EO5QW8', 'B000G6RPMY', 'B002GWHC0G', 'B004N5KULM', 'B001EO5TPM', 'B005DUM9UQ', 'B000E7VI7S', 'B001GVISJC', 'B006SQBRMA', 'B0059WXJKM', 'B001EPPI84', 'B004X2KR36', 'B005R8JE8O', 'B0066DMI6Y', 'B003ZFRKGO', 'B0019CW0HE', 'B004K2IHUO', 'B001REEG6C', 'B000J0HIT2', 'B0037LW78C', 'B0026Y3YBK', 'B003SE19UK', 'B003OB0IB8', 'B002SRYRE8', 'B001GVISJW', 'B0017I8UME', 'B0064KU9HO', 'B0037ZFEW4', 'B00374XSVY', 'B005P0HHGK', 'B002HQAXUW', 'B000SV90J8', 'B0036VM05I', 'B000ITVLE2', 'B003TQQKFQ', 'B007B9J6G2', 'B0064KO0BU', 'B001IUKD76', 'B0081XN2HQ', 'B0025VRCJY', 'B005CJVVJ8', 'B001KUUNP6', 'B000NY8ODS', 'B00029XIZI', 'B004V6AH34', 'B0028C44Z0', 'B0009XLVGA', 'B002TDK0VK', 'B0040WAG7Q', 'B001EO5ZMO', 'B001SATU8E', 'B007JFV6RK', 'B001L4ELRW', 'B003SO503C', 'B003ZFXJDW', 'B001HTL6CY', 'B000WFRMRW', 'B001EO5ZME', 'B0093NIWVO', 'B001EO5ZMY', 'B007TFONH0', 'B000GGKQSO', 'B007J32WX4', 'B0017129DC', 'B0002567IW', 'B0048IC328', 'B0025WIAN0', 'B00821UN4M', 'B00473RWXY', 'B002MV23XM', 'B000LKZK7C', 'B001D07IPG', 'B000H13270', 'B001UJEN6C', 'B003EMU7EU', 'B0064KOUNI', 'B003YDP5PA', 'B002X9JNYU', 'B001EPQ0J0', 'B000O9Y62A', 'B00283TPYE', 'B001FB69YY', 'B004OYBN7C', 'B000JEHAHS', 'B003VTN95K', 'B007DJ0O9I', 'B00469VSJI', 'B002SRAU80', 'B0067LDV66', 'B000HKYP9A', 'B00067AD4U', 'B0029OWDAU', 'B000SEJ84M', 'B001LMNXFA', 'B002J2DO8W', 'B000U9V5AU', 'B001SB099M', 'B0041QJSJS', 'B00437JI8Q', 'B002O3VHXU', 'B0057FTBYO', 'B0087HW5E2', 'B003YXWAF8', 'B000UXA3L8', 'B004AVYUOW', 'B001ESOQAM', 'B001ESOQA2', 'B003XT4AV0', 'B001FA1MCO', 'B001ELL6O8', 'B006CGURWM', 'B00250I0EG', 'B000C21OOM', 'B003D4O92K', 'B001IZM8A6', 'B0089PI9OC', 'B00061KYVI', 'B003XV5LHK', 'B005WU7V00', 'B005CFC9XY', 'B000SEJ842', 'B004X8TK8I', 'B000G6RYNE', 'B001EPPFGO', 'B007F96QQQ', 'B002HQH04O', 'B0025Z7CGI', 'B007SESJWC', 'B001EPPCNK', 'B002U56JXU', 'B002BCD2OG', 'B00112B9T2', 'B000G6MBX2', 'B001F33UPI', 'B008BEGP9W', 'B002U56JXK', 'B007P2PSKS', 'B004A8QK98', 'B001HTKZ5S', 'B001BB3LW6', 'B005Y0DXBO', 'B001LO4ZWI', 'B000QWXG9O', 'B000Y2EJHY', 'B004WJAUBE', 'B0035YE9CS', 'B006JWQFC0', 'B009HINRX8', 'B0016J4QKO', 'B001EPPE42', 'B000NF69ZM', 'B005YNDIAW', 'B005MZIJBU', 'B002KXDK48', 'B001FKQQDO', 'B000WFM204', 'B007ZENY5W', 'B002ONIVX4', 'B002GUWBMC', 'B0018DQFPC', 'B000UZMJZO', 'B00285FF6O', 'B0067R3Q9M', 'B004S0AQHA', 'B0030C9A60', 'B004ET7MG8', 'B003AO5DLO', 'B0007NG56I', 'B001EQ55BI', 'B000IXUISS', 'B001ELL9X6', 'B000UWSQT0', 'B0041CKRJC', 'B0007NG568', 'B000VKYKTG', 'B000HDMUQ2', 'B0041QIHC2', 'B001ELL9XG', 'B002DXZI40', 'B000ER6YO0', 'B003KDCJYY', 'B000MTIYF2', 'B002E0RIHM', 'B000S806VM', 'B000YT5DBS', 'B005NEXK6Y', 'B00448SNSA', 'B0002XIB2Y', 'B000F0G75W', 'B001EQ4DVQ', 'B004DTNJU2', 'B00473OV2E', 'B0048IACB2', 'B006F2NYI2', 'B004A8VV42', 'B002XG21MO', 'B006GK4XVA', 'B0049UVNYY', 'B0062KYM9C', 'B0002MKFEM', 'B0040TPNO0', 'B0048IK8UC', 'B000LRG11O', 'B001E6KBSK', 'B0017SRF52', 'B001HTKS1Y', 'B000FL08PG', 'B0018AMWES', 'B001TZSDJK', 'B000CQG862', 'B001SAX7Z6', 'B000XB80CG', 'B001GCVLXG', 'B0017ZBPTW', 'B001QXUTLU', 'B004XDMS3C', 'B0043PU4VS', 'B001GE3T1G', 'B0035Q0N0I', 'B0028PDGQA', 'B003UY9GTE', 'B006WVH7NE', 'B0030N5K1I', 'B004FD13RW', 'B0025ULYKI', 'B003UDV9SG', 'B00469VHRG', 'B002N2XXUC', 'B0028SWACS', 'B001KVLDBI', 'B000UUYOPW', 'B003UDSXU8', 'B002OHOC6A', 'B002NVPPHC', 'B001ELL7JM', 'B004SR97LO', 'B004OHKJWE', 'B0017WFN4S', 'B004134H9W', 'B007NWPB70', 'B000084E1U', 'B003JNWQPC', 'B001DR488E', 'B004IF3TAQ', 'B000IUOBMA', 'B001HTI226', 'B004XRJ1W4', 'B004MTMYNQ', 'B00283V4GQ', 'B000JWEEB0', 'B005J4ZMR8', 'B008L19ZQ0', 'B00845LR4G', 'B005O072PC', 'B006QFTT4Q', 'B000K296BW', 'B002G8N4ZW', 'B000OK7UDG', 'B002483RNI', 'B00002Z754', 'B001209QMU', 'B001FA1L7K', 'B004QXELMK', 'B0045TK2ZK', 'B002T62G7S', 'B002CTJG02', 'B001FA1L7A', 'B003O7A70Y', 'B000LKZTSC', 'B007237380', 'B000WNJ73Q', 'B002WJYCR4', 'B001FA1L7U', 'B000FAMUO4', 'B003MCEV0I', 'B002ULEFYE', 'B00110GBII', 'B004EKO2HE', 'B000VX9XFE', 'B00126EQBU', 'B008MMLXEK', 'B002TM37RU', 'B000I5DJVE', 'B000ODRY9I', 'B001JTCPE0', 'B003Y0ZJUE', 'B0046HFHD8', 'B005151BV6', 'B0006GWXYY', 'B007KDXVII', 'B0013JQON4', 'B001HTN94C', 'B00068PCTU', 'B001EO7GAI', 'B005CV7TOC', 'B0043H35N0', 'B005LURDB8', 'B00061NJ06', 'B0025VPBQ0', 'B0000VLH8S', 'B001E50UEQ', 'B003OB2EP6', 'B006ZLAH4M', 'B004S4V9Y0', 'B001EO7E2I', 'B0040WHJQM', 'B002BKTWL0', 'B0041T6KT6', 'B009UOFU20', 'B001IB69B4', 'B00132EDUW', 'B000GULKW6', 'B00032EZRY', 'B000YZTAS4', 'B000LRFZE8', 'B0004MUZKO', 'B000WFL0HK', 'B000F7PW8S', 'B002JLT6QC', 'B002KGN4LE', 'B003UIDAHY', 'B001LQNX8S', 'B000E1HUEE', 'B0045TDE4Q', 'B000KM7DVC', 'B00116629A', 'B000E7WM0K', 'B000E7WM0U', 'B0052BRVTK', 'B001EU5S9S', 'B00443YFCS', 'B0009QWTMM', 'B001CWZXIY', 'B001RVFDOO', 'B005FG6KZ8', 'B0016PKA1C', 'B000F4EU52', 'B005ATI9EU', 'B001HBTGI8', 'B00473VNHK', 'B004H4R0YO', 'B001FQ0UEE', 'B000X61Y60', 'B0017165OG', 'B000OR0WFM', 'B005HGAVGA', 'B003FA0M1O', 'B000F52NU0', 'B001E5E29A', 'B005HGAVGK', 'B003NRLKOM', 'B007POA176', 'B0001OINNQ', 'B000EPP56U', 'B00061EXBU', 'B0064MEUS6', 'B000EGX2E6', 'B0017OV6LA', 'B003SQ9WHA', 'B004ET9OIW', 'B001HTG9VW', 'B002BB7EWI', 'B004K6781Y', 'B000N5XCPM', 'B004HOSHU0', 'B000LKTDNE', 'B002SW7ZOW', 'B000QUZ9LO', 'B000EPP56K', 'B008YAXFWI', 'B001FPT2MG', 'B000LRKO9E', 'B002RKGEII', 'B0007T3V82', 'B002C4HZ00', 'B000ARTNR4', 'B002WUQUIC', 'B0000DC5IY', 'B001HKZDYU', 'B003NZH3VS', 'B003HG6U3A', 'B000E1HVF2', 'B001CWZXW0', 'B0052GPN0O', 'B00182I57O', 'B0016D2MY2', 'B003YMAET8', 'B005QSIQKC', 'B002MB2J6I', 'B003WWJ8LA', 'B0001VWE02', 'B004NC7IFQ', 'B00176AIDU']})

    # Extract ASINs from the DataFrame
    asins = df["ProductId"].tolist()

    # Load existing product mappings
    product_mapping = load_product_mapping()

    # --- DATA CLEANING ---
    # Step 1: Handling missing data 
    # Replace all missing values with NaN 
    df.isnull().sum()

    # Replace missing values in 'ProfileName' and 'Summary' columns with NaN
    # replacing NULLs with empty string
    df.fillna("", inplace=True)

    # Step 2: Remove the irrelevant observations
    # Drop Id, HelpfulnessNumerator, HelpfulnessDenomoninator, Summary attributes.
    df.drop(labels=['Id','HelpfulnessNumerator','HelpfulnessDenominator','Summary'],axis=1,inplace=True)
    
    # --- DATA TRANSFORMATION ---
    # Step 1: Perform data transformation
    df[['Score']].describe().transpose()

    df.rename(columns={'Score': 'Rating'}, inplace=True)
    df.rename(columns={'Text': 'Review'}, inplace=True)

    # --- MODELLING ---
    # Keep the users where the user has rated a rating a Product more than 4,5
    no_of_users = df['UserId'].value_counts()
    df_final = df[df['UserId'].isin(no_of_users[no_of_users >= 5].index)]
    new_df = df_final[['UserId', 'ProductId', 'Rating']]
    reader = Reader(rating_scale=(1, 5))
    data = Dataset.load_from_df(new_df,reader)

    # Splitting the dataset
    trainset, testset = train_test_split(data, test_size=0.2,random_state=10)

    # Train the model using the training set
    # Use user_based true/false to switch between user-based or item-based collaborative filtering
    algo_knn = KNNBasic(k=50, sim_options={'name': 'cosine','shrinkage': 0})
    algo_svd = SVD()
    algo_knn.fit(trainset)
    algo_svd.fit(trainset)
    
    # Run the trained model against the test set.
    knn_test_pred = algo_knn.test(testset)
    svd_test_pred = algo_svd.test(testset)
    
   # Combine predictions of KNN and SVD
    combined_predictions = []
    for knn_pred, svd_pred in zip(knn_test_pred, svd_test_pred):
        combined_rating = (knn_pred.est + svd_pred.est) / 2
        combined_pred = (knn_pred.uid, knn_pred.iid, knn_pred.r_ui, combined_rating, None)
        combined_predictions.append(combined_pred)

    # Compute RMSE and MAE for combined predictions
    combined_rmse = accuracy.rmse(combined_predictions)
    combined_mae = accuracy.mae(combined_predictions)

    # Mixed Hybridization
    predictions_cf = svd_test_pred
    predictions_knn = knn_test_pred

   # Mixing results and adding mock true ratings and additional value
    final_predictions = [(pred_cf.uid, pred_cf.iid, 0, (pred_cf.est + pred_knn.est) / 2, None) for pred_cf, pred_knn in zip(predictions_cf, predictions_knn)]

    # Performance metrics
    accuracy_mae = accuracy.mae(final_predictions)
    accuracy_rmse = accuracy.rmse(final_predictions)

    # Define a threshold for binary classification
    mixed_threshold = 3.5 

    # Convert predicted ratings to binary recommendations
    mixed_binary_predictions = [1 if pred >= mixed_threshold else 0 for _, _, _, pred, _ in final_predictions]

    mixed_true_labels = [pred.r_ui for pred in knn_test_pred]

    # Convert true ratings to binary labels
    mixed_true_labels_binary = [1 if rating >= mixed_threshold else 0 for rating in mixed_true_labels]

    mixed_precision, mixed_recall, mixed_f1_score, _ = precision_recall_fscore_support(mixed_true_labels_binary, mixed_binary_predictions, average='binary')

    ## Precision, Recall and F1-Score ##
    threshold_combined = 3.5
    # Prepare the data for computing precision, recall, and F1-score
    y_true = []
    y_pred = []

    for uid, iid, true_r, est, _ in combined_predictions:
        y_true.append(1 if true_r >= threshold_combined else 0)
        y_pred.append(1 if est >= threshold_combined else 0)

    combined_precision,combined_recall, combined_f1_score, _ = precision_recall_fscore_support(y_true, y_pred, average='binary')

    # Extract true labels and predicted labels for KNN
    true_labels = [pred.r_ui for pred in knn_test_pred]
    predicted_labels = [pred.est for pred in knn_test_pred]

    # Binarize the predictions and true labels
    threshold = 3.5
    true_labels_binary = [1 if rating >= threshold else 0 for rating in true_labels]
    predicted_labels_binary = [1 if rating >= threshold else 0 for rating in predicted_labels]

    # Calculate precision, recall, and F1-score
    knn_precision, knn_recall, knn_f1_score, _ = precision_recall_fscore_support(true_labels_binary, predicted_labels_binary, average='binary')

    # Extract true labels and predicted labels for SVD
    true_labels = [pred.r_ui for pred in svd_test_pred]
    predicted_labels = [pred.est for pred in svd_test_pred]

    # Binarize the predictions and true labels
    threshold_svd = 4
    true_labels_binary_svd = [1 if rating >= threshold_svd else 0 for rating in true_labels]
    predicted_labels_binary_svd = [1 if rating >= threshold_svd else 0 for rating in predicted_labels]

    # Calculate precision, recall, and F1-score
    svd_precision, svd_recall, svd_f1_score, _ = precision_recall_fscore_support(true_labels_binary_svd, predicted_labels_binary_svd, average='binary')

    # Meta-Level Hybridization
    # First is SVD, second is Linear Regression
    X = np.array([pred.est for pred in predictions_cf]).reshape(-1, 1)
    y = np.array([pred.r_ui for pred in predictions_cf])

    # Second level model
    reg = LinearRegression()
    reg.fit(X, y)

    # Predict using meta model
    meta_predictions = reg.predict(X)

    # Calculate and print accuracy metrics
    meta_mae = np.mean(np.abs(y - meta_predictions))
    meta_rmse = np.sqrt(np.mean((y - meta_predictions) ** 2))

    # Define a threshold for binary classification
    meta_threshold = 3.0

    # Convert predicted ratings to binary recommendations
    meta_binary_predictions = [1 if pred >= meta_threshold else 0 for pred in meta_predictions]

    # Convert true ratings to binary labels
    true_labels_binary_meta = [1 if rating >= meta_threshold else 0 for rating in y]

    # Calculate precision, recall, and F1-score
    meta_precision, meta_recall, meta_f1_score, _ = precision_recall_fscore_support(true_labels_binary_meta, meta_binary_predictions, average='binary')

    # ---- SIDE BAR ----
    authenticator.logout("Log Out", "sidebar")
    st.sidebar.title("Hi, Admin!")
    
    st.sidebar.subheader("Please filter here :point_down:")
    st.sidebar.write("Filter by")

    usernames = st.sidebar.multiselect(
        "Select the User:",
        options=df["ProfileName"].unique()
    )

    df['ProductName'] = df['ProductId'].map(product_mapping)

    product_name = st.sidebar.multiselect(
        "Select the Product Name:",
        options=df["ProductName"].unique()
    )
    
    rating = st.sidebar.multiselect(
        "Select the Rating:",
        options=df["Rating"].unique()
    )

    if (len(usernames)!=0):
        df = df[df['ProfileName'].isin(usernames)]
    if (len(product_name)!=0):
        df = df[df['ProductId'].isin(product_name)]
        df = df[df['ProductId'].isin(product_name)]
    if (len(rating)!=0):
        df = df[df['Rating'].isin(rating)]
    
    # ---- MAINPAGE ----
    # Set title
    st.title("Hybrid-Based on Food Recommender System")

    # Create tabs # "Term Frequency Analysis"
    tabsCat = ["Overall Dashboard","Customer Dashboard","KNN Baseline","SVD Baseline","Linear Regression Baseline", "Weighted Hybridization", "Mixed Hybridization","Meta-level Hybridization","Term Frequency Analysis", "Comparison Graph"]
    tab1, tab2, tab3, tab4, tab5, tab6, tab7, tab8, tab9, tab10= st.tabs(tabsCat)

    # GLOBAL
    BLANK_PAGE_ERROR = "[No data is selected]"

    with tab1:  # Tab 1 : Overall Dashboard
        st.header("Overall Dashboard")
        
        # TOP KPI's
        left_column, middle_column, right_column = st.columns(3)
        with left_column:
            st.subheader("Total Records:")
            st.subheader(df.shape[0])
        with middle_column:
            st.subheader("Total Users:")
            st.subheader(len(df["UserId"].unique()))
        with right_column:
            if (np.isnan(df["Rating"].mean())):
                average_rating = 0
            else:
                average_rating = round(df["Rating"].mean(), 1)
            star_rating = ":star:" * int(round(average_rating, 0))

            st.subheader("Average Ratings:")
            st.subheader(f"{average_rating} {star_rating}")

        st.markdown("""---""")

        if (df.shape[0]==0):
            st.write(BLANK_PAGE_ERROR)
        else:
            # Reviews per day plot
            df['Time'] = pd.to_datetime(df['Time'], unit='s')  # Convert time to datetime
            df['Year'] = df['Time'].dt.year  # Extract year component
            df_plot = df.groupby(df['Time']).count()[['Rating']]
            df_plot.index.names = ['Time']
            df_plot.rename(columns = {'Rating':'Number of reviews'}, inplace = True)
            
            fig_reviews_per_day = px.line(df_plot,
                        y='Number of reviews',
                        x =df_plot.index,
                        width=1200, height=500)

            selected_point = plotly_events(fig_reviews_per_day, click_event=True)
            
            
            if (len(selected_point)!=0):
                st.write(f"data on {selected_point[0]['x']}")
                selected_row = df[df.Time == selected_point[0]['x']][['ProductId','ProductId']].groupby('ProductId').count()

                selected_row.rename(columns = {'ProductId':'Count'}, inplace = True)
                fig2 = px.bar(selected_row,
                            y='Count',
                            x =selected_row.index,
                            width=800, height=500,
                            text_auto=True)
                st.plotly_chart(fig2)    

    # # Step 2: Map product names to product IDs in DataFrame
    df['ProductName'] = df['ProductId'].map(product_mapping)

    with tab2:  # Tab 2 : Customer Dashboard
        st.header("Customer Dashboard")
        st.markdown("##")

        if (df.shape[0]==0):
            st.write(BLANK_PAGE_ERROR)
        else:            
            # Rating by User (Bar Chart)
            # Top Rating Count Distribution grouped by Users
            rating_user = df[['ProfileName','Rating']].groupby('ProfileName').count()
            rating_user.rename(columns={'Rating': 'rating_count'}, inplace=True)
        
            fig_user_rating = px.bar(
                rating_user,
                x=rating_user.index,
                y="rating_count",
                title="<b>Top Rating Count Distribution grouped by User</b>",
                color_discrete_sequence=["#0083B8"] * len(rating_user),
                template="plotly_white",
                text_auto=True
            )

            fig_user_rating.update_layout(
                xaxis=dict(tickmode="linear"),
                plot_bgcolor="rgba(0,0,0,0)",
                yaxis=(dict(showgrid=False)),
            )
                
            st.plotly_chart(fig_user_rating, use_container_width=True)

            range = st.slider("A slider for showing the rows of pivot table",0, df.shape[0],0,100,key = "1")
            st.write(f"Range {range} to {range+99} row(s)")            
            st.dataframe(df[range:range+100][["Time","ProfileName","Rating","ProductName","Review"]])

    with tab3:  # Tab 3 : Evaluation Metrics
        st.subheader("K-Nearest Neighbors Algorithm (KNN) Model")
        st.write("User-based Model: Test Set")
        
        # Get Root Mean Squared Error (RMSE)
        st.write("---------------------------")
        st.write("Root Mean Squared Error (RMSE):")
        knn_rmse = accuracy.rmse(knn_test_pred, verbose=True)
        st.write(knn_rmse)

        # Get Mean Absolute Error (MAE)
        st.write("---------------------------")
        st.write("Mean Absolute Error (MAE):")
        knn_mae = accuracy.mae(knn_test_pred, verbose=True)
        st.write(knn_mae)
        
        # Get Precision
        st.write("---------------------------")
        st.write("Precison:")
        st.write(knn_precision)

        # Get F1-score
        st.write("---------------------------")
        st.write("F1-Score:")
        st.write(knn_f1_score)

        # Get Recall
        st.write("---------------------------")
        st.write("Recall:")
        st.write(knn_recall)

        # Perform cross validation
        st.write("---------------------------")
        st.subheader("Evaluating applied model on 5 split(s) using cross-validation")
        st.table(cross_validate(algo_knn, data, measures=['RMSE','MAE'], cv=5, verbose=False))


    with tab4:
        st.subheader("Singular Value Decomposition (SVD) Model")
        
        # Get Root Mean Squared Error (RMSE)
        st.write("---------------------------")
        st.write("Root Mean Squared Error (RMSE):")
        svd_rmse =accuracy.rmse(svd_test_pred, verbose=True)
        st.write(svd_rmse)

        # Get Mean Absolute Error (MAE)
        st.write("---------------------------")
        st.write("Mean Absolute Error (MAE):")
        svd_mae =accuracy.mae(svd_test_pred, verbose=True)
        st.write(svd_mae)

        # Get Precision
        st.write("---------------------------")
        st.write("Precison:")
        st.write(svd_precision)

        # Get F1-score
        st.write("---------------------------")
        st.write("F1-Score:")
        st.write(svd_f1_score)

        # Get Recall
        st.write("---------------------------")
        st.write("Recall:")
        st.write(svd_recall)
        
        # Perform cross validation
        st.write("---------------------------")
        st.subheader("Evaluating applied model on 5 split(s) using cross-validation")
        st.table(cross_validate(algo_svd, data, measures=['RMSE','MAE'], cv=5, verbose=False))


    with tab5:
        st.subheader("Linear Regression Model")
        
        # Get Root Mean Squared Error (RMSE)
        st.write("---------------------------")
        st.write("Root Mean Squared Error (RMSE):")
        st.write(meta_rmse)

        # Get Mean Absolute Error (MAE)
        st.write("---------------------------")
        st.write("Mean Absolute Error (MAE):")
        st.write(meta_mae)

    with tab6: 
        st.subheader("Weighted Hybridization")

         # Get Root Mean Squared Error (RMSE)
        st.write("---------------------------")
        st.write("Root Mean Squared Error (RMSE):")
        st.write("Combined RMSE:", combined_rmse)

        # Get Mean Absolute Error (MAE)
        st.write("---------------------------")
        st.write("Mean Absolute Error (MAE):")
        st.write("Combined MAE:", combined_mae)

        # Get Precision
        st.write("---------------------------")
        st.write("Precison:")
        st.write(combined_precision)

        # Get F1-score
        st.write("---------------------------")
        st.write("F1-Score:")
        st.write(combined_f1_score)

        # Get Recall
        st.write("---------------------------")
        st.write("Recall:")
        st.write(combined_recall)

    with tab7: 
        st.subheader("Mixed Hybridization")

        # Get Root Mean Squared Error (RMSE)
        st.write("---------------------------")
        st.write("Root Mean Squared Error (RMSE):")
        st.write("Combined RMSE:",accuracy_rmse)

        # Get Mean Absolute Error (MAE)
        st.write("---------------------------")
        st.write("Mean Absolute Error (MAE):")
        st.write("Combined MAE:",accuracy_mae)

        # Get Precision
        st.write("---------------------------")
        st.write("Precison:")
        st.write(mixed_precision)

        # Get F1-score
        st.write("---------------------------")
        st.write("F1-Score:")
        st.write(mixed_f1_score)

        # Get Recall
        st.write("---------------------------")
        st.write("Recall:")
        st.write(mixed_recall)


    with tab8: 
        st.subheader("Meta-level Hybridization")

        # Get Root Mean Squared Error (RMSE)
        st.write("---------------------------")
        st.write("Root Mean Squared Error (RMSE):")
        st.write("Combined RMSE:",meta_rmse)

        # Get Mean Absolute Error (MAE)
        st.write("---------------------------")
        st.write("Mean Absolute Error (MAE):")
        st.write("Combined MAE:",meta_mae)

        # Get Precision
        st.write("---------------------------")
        st.write("Precison:")
        st.write(meta_precision)

        # Get F1-score
        st.write("---------------------------")
        st.write("F1-Score:")
        st.write(meta_f1_score)

        # Get Recall
        st.write("---------------------------")
        st.write("Recall:")
        st.write(meta_recall)

    with tab9: # Term Frequency - Inverse Document Frequency
        
        v = TfidfVectorizer(ngram_range=(1,1), stop_words='english')
        tfidf_matrix  = v.fit_transform(df['Review'])
        idf_values = dict(zip(v.get_feature_names_out(), v.idf_))

        topN = st.selectbox(
            label="Select top N data",
            options=['1','2','3','4','5','6','7','8','9','10'],
            index=9
        )
        
        idf_desc = pd.DataFrame(idf_values, index=['IDF Values']).T.sort_values(by=['IDF Values'], ascending=False).head(int(topN))
        idf_asc = pd.DataFrame(idf_values, index=['IDF Values']).T.sort_values(by=['IDF Values'], ascending=True).head(int(topN))
        
        st.write(f"Top {topN} Frequent Word(s) Found")
        st.bar_chart(idf_desc.sort_values(by=['IDF Values'], ascending=False))
        st.table(idf_desc.sort_values(by=['IDF Values'], ascending=False))
        
        st.write("---------------------------")
        st.write(f"Least {topN} Frequent Word(s) Found")
        st.bar_chart(idf_asc.sort_values(by=['IDF Values'], ascending=True))
        st.table(idf_asc.sort_values(by=['IDF Values'], ascending=True))


    with tab10: 
        st.subheader("Graph")
        # Sample data extracted and assumed for the demonstration
        models = ['KNN', 'SVD', 'Weighted Hybridization', 'Mixed Hybridization']
        rmse_values = [knn_rmse, svd_rmse, combined_rmse,accuracy_rmse] 
        mae_values = [knn_mae, svd_mae, combined_mae, accuracy_mae]  
        precision_values = [knn_precision, svd_precision, combined_precision,mixed_precision]  
        recall_values = [knn_recall, svd_recall, combined_recall, mixed_recall]  
        f1_score_values = [knn_f1_score, svd_f1_score, combined_f1_score, mixed_f1_score]

        x = np.arange(len(models))
        width = 0.2

        fig, ax1 = plt.subplots(figsize=(12, 6))

        # Create bar plots for RMSE and MAE
        bar1 = ax1.bar(x - width/2, rmse_values, width, label='RMSE', color='salmon')
        bar2 = ax1.bar(x + width/2, mae_values, width, label='MAE', color='peachpuff')

        # Add some text for labels, title, and custom x-axis tick labels, etc.
        ax1.set_xlabel('Models')
        ax1.set_ylabel('Error Metrics')
        ax1.set_title('Comparison of Evaluation Metrics Across SVD + KNN')
        ax1.set_xticks(x)
        ax1.set_xticklabels(models)
        ax1.legend(loc='upper left')

        # Create another y-axis for the classification metrics
        ax2 = ax1.twinx()
        ax2.set_xlabel('Classification Metrics')

        # Plot lines for Precision, Recall, and F1 Score
        line1 = ax2.plot(x, precision_values, color='blue', marker='o', label='Precision')
        line2 = ax2.plot(x, recall_values, color='green', marker='o', label='Recall')
        line3 = ax2.plot(x, f1_score_values, color='purple', marker='o', label='F1 Score')

        fig.tight_layout()  # to ensure the right y-label is not slightly clipped

        # Create a combined legend
        lines = line1 + line2 + line3
        labels = [l.get_label() for l in lines]
        ax2.legend(lines, labels, loc='upper right')

        # Adding value labels on top of bars and points
        def add_labels(bars, values):
            for bar, value in zip(bars, values): 
                height = bar.get_height()
                ax1.annotate('{}'.format(value),
                            xy=(bar.get_x() + bar.get_width() / 2, height),
                            xytext=(0, 3),  # 3 points vertical offset
                            textcoords="offset points",
                            ha='center', va='bottom')

        for i, (p, r, f) in enumerate(zip(precision_values, recall_values, f1_score_values)):
            ax2.annotate(f'{p}', xy=(x[i], p), textcoords="offset points", xytext=(0, 5), ha='center', color='blue')
            ax2.annotate(f'{r}', xy=(x[i], r), textcoords="offset points", xytext=(0, 5), ha='center', color='green')
            ax2.annotate(f'{f}', xy=(x[i], f), textcoords="offset points", xytext=(0, 5), ha='center', color='purple')

        add_labels(bar1, rmse_values)
        add_labels(bar2, mae_values)

        # Display the plot in Streamlit
        st.pyplot(fig)

        # Example data
        models = ["SVD", "Linear Regression", "Meta-level Hybridization"]
        rmse_values = [svd_rmse, 1.256073,1.256073] 
        mae_values = [svd_mae, 8.88178, 8.88178] 
        precision_values = [1.0, 0, 1.0]  
        recall_values = [1.0, 0, 1.0] 
        f1_score_values = [1.0, 0, 1.0]  

        x = np.arange(len(models))
        width = 0.2

        fig, ax1 = plt.subplots(figsize=(12, 6))

        # Create bar plots for RMSE and MAE
        bar1 = ax1.bar(x - width/2, rmse_values, width, label='RMSE', color='salmon')
        bar2 = ax1.bar(x + width/2, mae_values, width, label='MAE', color='peachpuff')

        # Add some text for labels, title, and custom x-axis tick labels, etc.
        ax1.set_xlabel('Models')
        ax1.set_ylabel('Error Metrics')
        ax1.set_title('Comparison of Evaluation Metrics Across Linear Regression + SVD')
        ax1.set_xticks(x)
        ax1.set_xticklabels(models)
        ax1.legend(loc='upper left')

        # Create another y-axis for the classification metrics
        ax2 = ax1.twinx()
        ax2.set_ylabel('Classification Metrics')

        # Plot lines for Precision, Recall, and F1 Score
        line1 = ax2.plot(x, precision_values, color='blue', marker='o', label='Precision')
        line2 = ax2.plot(x, recall_values, color='green', marker='o', label='Recall')
        line3 = ax2.plot(x, f1_score_values, color='purple', marker='o', label='F1 Score')

        fig.tight_layout()  # to ensure the right y-label is not slightly clipped

        # Create a combined legend
        lines = line1 + line2 + line3
        labels = [l.get_label() for l in lines]
        ax2.legend(lines, labels, loc='upper right')

        # Adding value labels on top of bars and points
        def add_labels(bars, values):
            for bar, value in zip(bars, values):
                height = bar.get_height()
                ax1.annotate('{}'.format(value),
                            xy=(bar.get_x() + bar.get_width() / 2, height),
                            xytext=(0, 3),  # 3 points vertical offset
                            textcoords="offset points",
                            ha='center', va='bottom')

        for i, (p, r, f) in enumerate(zip(precision_values, recall_values, f1_score_values)):
            ax2.annotate(f'{p}', xy=(x[i], p), textcoords="offset points", xytext=(0, 5), ha='center', color='blue')
            ax2.annotate(f'{r}', xy=(x[i], r), textcoords="offset points", xytext=(0, 5), ha='center', color='green')
            ax2.annotate(f'{f}', xy=(x[i], f), textcoords="offset points", xytext=(0, 5), ha='center', color='purple')

        add_labels(bar1, rmse_values)
        add_labels(bar2, mae_values)

        # Display the plot
        st.pyplot(fig)
