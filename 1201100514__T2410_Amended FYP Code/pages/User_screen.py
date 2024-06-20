import pickle
from pathlib import Path
import sys
import os
from bs4 import BeautifulSoup
import requests
import pandas as pd
import numpy as np
import streamlit as st
import streamlit_authenticator as stauth
import plotly.express as px
import math
import re
from surprise import SVD
from IPython.display import display
from PIL import Image
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.by import By
from webdriver_manager.chrome import ChromeDriverManager
import urllib.request
import json
import logging
import time
from urllib.error import HTTPError
from surprise.reader import Reader
from surprise import Dataset
from surprise import KNNBasic
from surprise import accuracy
from surprise.model_selection import train_test_split
from collections import defaultdict
from surprise.model_selection.validation import cross_validate

from collections import defaultdict

from PIL import Image
import streamlit as st
import os

def get_top_n_recommendations(predictions, df, n=5, rating_threshold=3.5):
    # Map the predictions to each user.
    top_n = defaultdict(list)

    # UserId, Item ID, True Rating, Estimated Rating
    for uid, iid, true_r, est, _ in predictions:
        top_n[uid].append((iid, est))

    # Calculate the average rating for each product.
    average_ratings = df.groupby('ProductId')['Rating'].mean()

    # Filter the recommendations for each user.
    for uid, user_ratings in top_n.items():
        # Sort the predictions for each user and retrieve the highest ones.
        user_ratings.sort(key=lambda x: x[1], reverse=True)
        # Filter based on average rating threshold.
        filtered_ratings = [item for item in user_ratings if average_ratings.get(item[0], 0) >= rating_threshold]
        top_n[uid] = filtered_ratings[:n]
    return top_n

# def scrape_product_image_selenium(asins):
#     options = webdriver.ChromeOptions()
#     options.add_argument('--headless')
#     driver = webdriver.Chrome(service=Service(ChromeDriverManager().install()), options=options)

#     product_images = {}
#     for asin in asins:
#         url = f"https://www.amazon.com/dp/{asin}"
#         driver.get(url)
#         time.sleep(5)

#         try:
#             image_tag = driver.find_element(By.ID, "landingImage")
#             image_url = image_tag.get_attribute("src")
#             product_images[asin] = image_url
#         except Exception as e:
#             print(f"Error fetching product image for {asin}: {e}")
#             product_images[asin] = "NA"

#     driver.quit()
#     return product_images

def load_product_mapping(csv_path='Product_Data.csv'):
    if os.path.exists(csv_path):
        df = pd.read_csv(csv_path)
        return dict(zip(df['ProductId'], df['ProductName'])), dict(zip(df['ProductId'], df['ProductImage']))
    return {},{}

# Assuming df is the DataFrame that contains the ProductId column
df = pd.DataFrame({'ProductId': ['B001E4KFG0', 'B00813GRG4', 'B000LQOCH0', 'B000UA0QIQ', 'B006K2ZZ7K', 'B000E7L2R4', 'B00171APVA', 'B0001PB9FE', 'B0009XLVG0', 'B001GVISJM', 'B00144C10S', 'B0001PB9FY', 'B003F6UO7K', 'B001EO5QW8', 'B000G6RPMY', 'B002GWHC0G', 'B004N5KULM', 'B001EO5TPM', 'B005DUM9UQ', 'B000E7VI7S', 'B001GVISJC', 'B006SQBRMA', 'B0059WXJKM', 'B001EPPI84', 'B004X2KR36', 'B005R8JE8O', 'B0066DMI6Y', 'B003ZFRKGO', 'B0019CW0HE', 'B004K2IHUO', 'B001REEG6C', 'B000J0HIT2', 'B0037LW78C', 'B0026Y3YBK', 'B003SE19UK', 'B003OB0IB8', 'B002SRYRE8', 'B001GVISJW', 'B0017I8UME', 'B0064KU9HO', 'B0037ZFEW4', 'B00374XSVY', 'B005P0HHGK', 'B002HQAXUW', 'B000SV90J8', 'B0036VM05I', 'B000ITVLE2', 'B003TQQKFQ', 'B007B9J6G2', 'B0064KO0BU', 'B001IUKD76', 'B0081XN2HQ', 'B0025VRCJY', 'B005CJVVJ8', 'B001KUUNP6', 'B000NY8ODS', 'B00029XIZI', 'B004V6AH34', 'B0028C44Z0', 'B0009XLVGA', 'B002TDK0VK', 'B0040WAG7Q', 'B001EO5ZMO', 'B001SATU8E', 'B007JFV6RK', 'B001L4ELRW', 'B003SO503C', 'B003ZFXJDW', 'B001HTL6CY', 'B000WFRMRW', 'B001EO5ZME', 'B0093NIWVO', 'B001EO5ZMY', 'B007TFONH0', 'B000GGKQSO', 'B007J32WX4', 'B0017129DC', 'B0002567IW', 'B0048IC328', 'B0025WIAN0', 'B00821UN4M', 'B00473RWXY', 'B002MV23XM', 'B000LKZK7C', 'B001D07IPG', 'B000H13270', 'B001UJEN6C', 'B003EMU7EU', 'B0064KOUNI', 'B003YDP5PA', 'B002X9JNYU', 'B001EPQ0J0', 'B000O9Y62A', 'B00283TPYE', 'B001FB69YY', 'B004OYBN7C', 'B000JEHAHS', 'B003VTN95K', 'B007DJ0O9I', 'B00469VSJI', 'B002SRAU80', 'B0067LDV66', 'B000HKYP9A', 'B00067AD4U', 'B0029OWDAU', 'B000SEJ84M', 'B001LMNXFA', 'B002J2DO8W', 'B000U9V5AU', 'B001SB099M', 'B0041QJSJS', 'B00437JI8Q', 'B002O3VHXU', 'B0057FTBYO', 'B0087HW5E2', 'B003YXWAF8', 'B000UXA3L8', 'B004AVYUOW', 'B001ESOQAM', 'B001ESOQA2', 'B003XT4AV0', 'B001FA1MCO', 'B001ELL6O8', 'B006CGURWM', 'B00250I0EG', 'B000C21OOM', 'B003D4O92K', 'B001IZM8A6', 'B0089PI9OC', 'B00061KYVI', 'B003XV5LHK', 'B005WU7V00', 'B005CFC9XY', 'B000SEJ842', 'B004X8TK8I', 'B000G6RYNE', 'B001EPPFGO', 'B007F96QQQ', 'B002HQH04O', 'B0025Z7CGI', 'B007SESJWC', 'B001EPPCNK', 'B002U56JXU', 'B002BCD2OG', 'B00112B9T2', 'B000G6MBX2', 'B001F33UPI', 'B008BEGP9W', 'B002U56JXK', 'B007P2PSKS', 'B004A8QK98', 'B001HTKZ5S', 'B001BB3LW6', 'B005Y0DXBO', 'B001LO4ZWI', 'B000QWXG9O', 'B000Y2EJHY', 'B004WJAUBE', 'B0035YE9CS', 'B006JWQFC0', 'B009HINRX8', 'B0016J4QKO', 'B001EPPE42', 'B000NF69ZM', 'B005YNDIAW', 'B005MZIJBU', 'B002KXDK48', 'B001FKQQDO', 'B000WFM204', 'B007ZENY5W', 'B002ONIVX4', 'B002GUWBMC', 'B0018DQFPC', 'B000UZMJZO', 'B00285FF6O', 'B0067R3Q9M', 'B004S0AQHA', 'B0030C9A60', 'B004ET7MG8', 'B003AO5DLO', 'B0007NG56I', 'B001EQ55BI', 'B000IXUISS', 'B001ELL9X6', 'B000UWSQT0', 'B0041CKRJC', 'B0007NG568', 'B000VKYKTG', 'B000HDMUQ2', 'B0041QIHC2', 'B001ELL9XG', 'B002DXZI40', 'B000ER6YO0', 'B003KDCJYY', 'B000MTIYF2', 'B002E0RIHM', 'B000S806VM', 'B000YT5DBS', 'B005NEXK6Y', 'B00448SNSA', 'B0002XIB2Y', 'B000F0G75W', 'B001EQ4DVQ', 'B004DTNJU2', 'B00473OV2E', 'B0048IACB2', 'B006F2NYI2', 'B004A8VV42', 'B002XG21MO', 'B006GK4XVA', 'B0049UVNYY', 'B0062KYM9C', 'B0002MKFEM', 'B0040TPNO0', 'B0048IK8UC', 'B000LRG11O', 'B001E6KBSK', 'B0017SRF52', 'B001HTKS1Y', 'B000FL08PG', 'B0018AMWES', 'B001TZSDJK', 'B000CQG862', 'B001SAX7Z6', 'B000XB80CG', 'B001GCVLXG', 'B0017ZBPTW', 'B001QXUTLU', 'B004XDMS3C', 'B0043PU4VS', 'B001GE3T1G', 'B0035Q0N0I', 'B0028PDGQA', 'B003UY9GTE', 'B006WVH7NE', 'B0030N5K1I', 'B004FD13RW', 'B0025ULYKI', 'B003UDV9SG', 'B00469VHRG', 'B002N2XXUC', 'B0028SWACS', 'B001KVLDBI', 'B000UUYOPW', 'B003UDSXU8', 'B002OHOC6A', 'B002NVPPHC', 'B001ELL7JM', 'B004SR97LO', 'B004OHKJWE', 'B0017WFN4S', 'B004134H9W', 'B007NWPB70', 'B000084E1U', 'B003JNWQPC', 'B001DR488E', 'B004IF3TAQ', 'B000IUOBMA', 'B001HTI226', 'B004XRJ1W4', 'B004MTMYNQ', 'B00283V4GQ', 'B000JWEEB0', 'B005J4ZMR8', 'B008L19ZQ0', 'B00845LR4G', 'B005O072PC', 'B006QFTT4Q', 'B000K296BW', 'B002G8N4ZW', 'B000OK7UDG', 'B002483RNI', 'B00002Z754', 'B001209QMU', 'B001FA1L7K', 'B004QXELMK', 'B0045TK2ZK', 'B002T62G7S', 'B002CTJG02', 'B001FA1L7A', 'B003O7A70Y', 'B000LKZTSC', 'B007237380', 'B000WNJ73Q', 'B002WJYCR4', 'B001FA1L7U', 'B000FAMUO4', 'B003MCEV0I', 'B002ULEFYE', 'B00110GBII', 'B004EKO2HE', 'B000VX9XFE', 'B00126EQBU', 'B008MMLXEK', 'B002TM37RU', 'B000I5DJVE', 'B000ODRY9I', 'B001JTCPE0', 'B003Y0ZJUE', 'B0046HFHD8', 'B005151BV6', 'B0006GWXYY', 'B007KDXVII', 'B0013JQON4', 'B001HTN94C', 'B00068PCTU', 'B001EO7GAI', 'B005CV7TOC', 'B0043H35N0', 'B005LURDB8', 'B00061NJ06', 'B0025VPBQ0', 'B0000VLH8S', 'B001E50UEQ', 'B003OB2EP6', 'B006ZLAH4M', 'B004S4V9Y0', 'B001EO7E2I', 'B0040WHJQM', 'B002BKTWL0', 'B0041T6KT6', 'B009UOFU20', 'B001IB69B4', 'B00132EDUW', 'B000GULKW6', 'B00032EZRY', 'B000YZTAS4', 'B000LRFZE8', 'B0004MUZKO', 'B000WFL0HK', 'B000F7PW8S', 'B002JLT6QC', 'B002KGN4LE', 'B003UIDAHY', 'B001LQNX8S', 'B000E1HUEE', 'B0045TDE4Q', 'B000KM7DVC', 'B00116629A', 'B000E7WM0K', 'B000E7WM0U', 'B0052BRVTK', 'B001EU5S9S', 'B00443YFCS', 'B0009QWTMM', 'B001CWZXIY', 'B001RVFDOO', 'B005FG6KZ8', 'B0016PKA1C', 'B000F4EU52', 'B005ATI9EU', 'B001HBTGI8', 'B00473VNHK', 'B004H4R0YO', 'B001FQ0UEE', 'B000X61Y60', 'B0017165OG', 'B000OR0WFM', 'B005HGAVGA', 'B003FA0M1O', 'B000F52NU0', 'B001E5E29A', 'B005HGAVGK', 'B003NRLKOM', 'B007POA176', 'B0001OINNQ', 'B000EPP56U', 'B00061EXBU', 'B0064MEUS6', 'B000EGX2E6', 'B0017OV6LA', 'B003SQ9WHA', 'B004ET9OIW', 'B001HTG9VW', 'B002BB7EWI', 'B004K6781Y', 'B000N5XCPM', 'B004HOSHU0', 'B000LKTDNE', 'B002SW7ZOW', 'B000QUZ9LO', 'B000EPP56K', 'B008YAXFWI', 'B001FPT2MG', 'B000LRKO9E', 'B002RKGEII', 'B0007T3V82', 'B002C4HZ00', 'B000ARTNR4', 'B002WUQUIC', 'B0000DC5IY', 'B001HKZDYU', 'B003NZH3VS', 'B003HG6U3A', 'B000E1HVF2', 'B001CWZXW0', 'B0052GPN0O', 'B00182I57O', 'B0016D2MY2', 'B003YMAET8', 'B005QSIQKC', 'B002MB2J6I', 'B003WWJ8LA', 'B0001VWE02', 'B004NC7IFQ', 'B00176AIDU']})

# Extract ASINs from the DataFrame
asins = df["ProductId"].tolist()

# # Load existing product mappings
product_name_mapping, product_image_mapping = load_product_mapping()

st.set_page_config(page_title = "End-User", page_icon = ":bust_in_silhouette:", layout = "wide")

# Function to remove HTML tags and content
def remove_html_tags(text):
    return re.sub(r'<[^>]+>', '', text)

# --- LOAD DATASET ---
# Read data as a data frame
df = pd.read_csv('Reviews.csv')

# Show the Info of the data frame
df.head()

# Check the number of rows and columns
rows, columns = df.shape
# print("No of rows:", rows)
# print("No of columns:", columns)

# --- DATA CLEANING ---
# Step 1: Handling missing data
df.dropna(inplace=True)

# print('Number of missing values')
# print(df.isnull().sum())

# Step 2: Remove the irrelevant observations 
# Drop Id, Time, HelpfulnessNumerator, HelpfulnessDenomoninator, Summary attributes.
df.drop(labels=['Id','Time','HelpfulnessNumerator','HelpfulnessDenominator','Summary'],axis=1,inplace=True)

# Apply the function to the 'text' column to remove html tag
data = df['Text']
df1= pd.DataFrame(data)
df['Text'] = df1['Text'].apply(remove_html_tags)

# --- DATA TRANSFORMATION ---
# Step 1: Perform data transformation
# Summary statistics of 'rating' variable
df[['Score']].describe().transpose()
df.rename(columns={'Score': 'Rating'}, inplace=True)
df.rename(columns={'Text': 'Review'}, inplace=True)

# --- MODELLING ---
# df =  df.iloc[:2000]

# Keep the users where the user has rated more than 5 ratings 
no_of_users = df['ProfileName'].value_counts()
df_final = df[df['ProfileName'].isin(no_of_users[no_of_users >= 50].index)]
new_df = df_final[['ProfileName', 'ProductId', 'Rating']]
reader = Reader(rating_scale=(1, 5))
data = Dataset.load_from_df(new_df,reader)

# Splitting the dataset
trainset, testset = train_test_split(data, test_size=0.2,random_state=10)

algo_knn = KNNBasic(k=50, sim_options={'name': 'cosine','shrinkage': 0})
algo_knn.fit(trainset)

algo_svd = SVD()
algo_svd.fit(trainset)

# Run the trained model against the testset
knn_test_pred = algo_knn.test(testset)
svd_test_pred = algo_svd.test(testset)

# Combine predictions of KNN and SVD
combined_predictions = []
for knn_pred, svd_pred in zip(knn_test_pred, svd_test_pred):
    combined_rating = (knn_pred.est + svd_pred.est) / 2
    combined_pred = (knn_pred.uid, knn_pred.iid, knn_pred.r_ui, combined_rating, None)
    combined_predictions.append(combined_pred)

def mainPage(): 
    st.session_state.page = 1

def logout():
    st.session_state.page = 0
    st.session_state.ProfileName = None

def login(tmp):
    if(tmp in df.ProfileName.array):
        st.session_state.ProfileName = tmp
        mainPage()
    elif (tmp != ""):
        st.error('Invalid username', icon="🚨")
        #st.warning("Opps! This user is not found...")

#userNameList = df["username"].unique()
userName = ['Lynrie "Oh HELL no"']    
userNameList = pd.DataFrame(userName, columns=['ProfileName'])

# Page initialization
if "page" not in st.session_state:
    st.session_state.page = 0

# Username initialization
if "ProfileName" not in st.session_state:
    st.session_state.ProfileName = None


# --- LOGIN PAGE ---
if st.session_state.page == 0:
    st.title("Customer Login")
    tmp = st.text_input('Username', placeholder = "Please type your username here...")

    st.button("Login",on_click=login(tmp),disabled=(st.session_state.page > 3))


# --- MAIN PAGE ---
elif st.session_state.page == 1:

    # LOGOUT button
    st.sidebar.button("Log Out",on_click=logout)
    userName = st.session_state.ProfileName

    # only choose data related to this user 
    df = df[df["ProfileName"]==userName]
    
    # ---- SIDEBAR ----
    st.sidebar.title(f"Welcome {userName}!")
    
    st.sidebar.subheader("Please filter here :point_down:")
    username = userName

    df['ProductName'] = df['ProductId'].map(product_name_mapping)


    product_name = st.sidebar.multiselect(
        "Select the Product:",
        options=df["ProductName"].unique(),
    )

    rating = st.sidebar.multiselect(
        "Select the Rating:",
        options=df["Rating"].unique(),
    )

    if (len(product_name)!=0):
        df = df[df['ProductId'].isin(product_name)]
    if (len(rating)!=0):
        df = df[df['Rating'].isin(rating)]


    # ---- MAINPAGE ----
    # Set title
    st.title("Hybrid-Based on Food Recommender System")

    # Create tabs
    tabsCat = ["Dashboard", "Products Recommender"]
    tab1, tab2= st.tabs(tabsCat)

    # GLOBAL
    BLANK_PAGE_ERROR = "[No data is selected]"
    
    with tab1:  # Tab 1 : Dashboard
        st.subheader("Your Dashboard")
        st.markdown("##")

        # TOP KPI's
        total_purchase = df.shape[0]
        total_rating = df['Rating'].count()
        if total_rating > 0:
            average_rating = round(df['Rating'].mean(), 2)
            star_rating = ":star:" * math.floor(average_rating)

        else:
            average_rating = 0  # Assign a default value if no ratings are available
            star_rating = "No ratings available"  # or any other appropriate message

        left_column, middle_column, right_column = st.columns(3)
        with left_column:
            st.subheader("Total Purchases:")
            st.subheader(f"{total_purchase:,}")
        with middle_column:
            st.subheader("Total Ratings:")
            st.subheader(f"{total_rating:,}")
        with right_column:
            st.subheader("Average Ratings:")
            st.subheader(f"{average_rating} {star_rating}")

        st.markdown("""---""")

        # Create a new column with ratings in descending order
        df['Score_rating'] = df['Rating'].apply(lambda x:6- x)

        # Rating by User (Bar Chart) 
        rating_user = df[['ProfileName', 'Score_rating']].groupby('ProfileName').count()
        rating_user.rename(columns={'Score_rating': 'rating_count'}, inplace=True)

        # Rating by Products (Bar Chart)
        rating_product = df[['ProductId', 'Score_rating']].groupby('ProductId').count()
        rating_product.rename(columns={'Score_rating': 'rating_count'}, inplace=True)

        fig_product_rating = px.bar(
            rating_product,
            x="rating_count",
            y=rating_product.index,
            orientation="h",
            title="<b>Rating by Product</b>",
            color_discrete_sequence=["#0083B8"] * len(rating_product),
            template="plotly_white",
        )

        fig_user_rating = px.bar(
            rating_user,
            x=rating_user.index,
            y="rating_count",
            title="<b>Rating by User</b>",
            color_discrete_sequence=["#0083B8"] * len(rating_user),
            template="plotly_white",
        )

        left_column, right_column = st.columns(2)
        left_column.plotly_chart(fig_user_rating, use_container_width=True)
        right_column.plotly_chart(fig_product_rating, use_container_width=True)

    with tab2:  # Tab 2 : Products Recommender Based on KNN Predictions
        st.subheader("Recommend for You:")
        top_n_recommendations = get_top_n_recommendations(knn_test_pred,df, n=5, rating_threshold=3.5)

        # Print the recommended items for each user
        if username in top_n_recommendations:
            user_recommendations = top_n_recommendations[username]
            items = [iid for (iid, _) in user_recommendations]
            recomm_items = df[df['ProductId'].isin(items)][['ProductId', 'Rating','Review']].drop_duplicates().reset_index(drop=True)
            # Fetch and map product images
            recomm_items['ProductImage'] = recomm_items['ProductId'].map(product_image_mapping)
            recomm_items['ProductName'] = recomm_items['ProductId'].map(product_name_mapping)

            # Display top 5 recommendations in a list
            for index, row in recomm_items.head(5).iterrows():
                product_id = row['ProductId']
                product_name = row['ProductName']
                rating = row['Rating']
                review = row['Review']
                product_image_url = row['ProductImage']

                cols = st.columns([1, 3]) 

                with cols[0]:  # First column for the product image
                    if product_image_url and product_image_url !="NA":
                        try:
                            img = Image.open(urllib.request.urlopen(product_image_url))
                            img = img.resize((100, 100))
                            st.image(img)
                        except Exception as e:
                            st.write("Image not available")
                    else:
                        st.write("Image not available")

                with cols[1]:  # Second column for the product details
                    if product_name:
                        st.write(f"**Product Name:** {product_name}")
                    else:
                        st.write(f"**Product ID:** {product_id}")
                        
                    st.write(f"**Rating:** {rating}")
                    st.write(f"**Review:** {review}")
                    st.markdown("---")  # Separator line
        else:
            st.write("No recommendations available for this user.")