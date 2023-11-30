from pydoc import source_synopsis
import pandas as pd
import numpy as np
import requests
import matplotlib.pyplot as plt
import glob
import os
import csv
import time
import sys
import mysql.connector
from webdriver_manager.chrome import ChromeDriverManager
from time import sleep
from random import randint
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.common.by import By
from mysql.connector import Error
from sqlalchemy import create_engine

#webscrapen van Booking.com
Folderpath = (r"C:\Users\thijs\Desktop\assignment 1") 
path_to_file = "Booking_Reviews.csv" 
num_page = 5 
url=  "https://www.booking.com/hotel/nl/twentyseven-amsterdam.nl.html#tab-reviews" 
driver = webdriver.Chrome(executable_path=r'C:\Program Files (x86)\chromedriver.exe') 
driver.get(url)
alert=WebDriverWait(driver, 15).until(EC.element_to_be_clickable((By.CSS_SELECTOR, "#onetrust-accept-btn-handler"))).click()

scrapedReviews=[]
wait = WebDriverWait(driver, 20)

for i in range(0, num_page):
    time.sleep(2)
    container = driver.find_elements(By.XPATH,value="//div [contains(@class, 'c-review-block')]")
    for j in range(len(container)):
        try:
            rating = container[j].find_element(By.XPATH,value=".//div [contains (@class, 'bui-review-score__badge')]").text
            title = container[j].find_element(By.XPATH,value=".//h3 [contains (@class, 'c-review-block__title c-review__title--ltr')]").text
            review = container[j].find_element(By.XPATH,value= ".//span [contains (@class,'c-review__body')]").text
            scrapedReviews.append([title, review, rating]) 
        except:
            continue
    driver.find_element(By.XPATH,value='.//a [contains(@class, "pagenext")]').click()

scrapedReviewsDF = pd.DataFrame(scrapedReviews, columns=['title','review','rating'])
scrapedReviewsDF['rating']= scrapedReviewsDF['rating'].str.replace(",",".")
scrapedReviewsDF.drop_duplicates(subset=['title','review','rating'],inplace=True)

BookingDF = scrapedReviewsDF.astype({'rating':'float'})
BookingDF['label'] = BookingDF['rating'].apply(lambda x: 1 if x > 5 else 0)
BookingDF = pd.DataFrame(BookingDF, columns=['title', 'review', 'rating','label'])
driver.quit()
print('Ready scraping ....')
bookingpath = os.path.join(Folderpath,path_to_file)
BookingDF.to_csv(bookingpath, sep=',',index= False)

#webscrapen van reviews van Corendon
path_to_file2 = "Corendon_Reviews.csv"
num_page2 = 10
url2=  "https://www.corendon.nl/curacao/willemstad/corendon-mangrove-beach-resort?tab=review-rating-tab#acco-tabs-section&[filters].*.*.*.0|||9404.CWMBR.AMSCUR.190622.5.DZM-X.."
driver = webdriver.Chrome(executable_path=r'C:\Program Files (x86)\chromedriver.exe')
driver.get(url2)
alert=WebDriverWait(driver, 15).until(EC.element_to_be_clickable((By.CSS_SELECTOR, "#st_popup_acceptButton"))).click()
scrapedReviews2=[]
wait = WebDriverWait(driver, 20)

for i in range(0, num_page2):
    time.sleep(2)
    container = driver.find_elements(By.XPATH,value="//div [contains(@class, 'cor-acco-reviews__review')]")
    for j in range(len(container)):
        try:

            rating2 = container[j].find_element(By.XPATH,value=".//div [contains (@itemprop, 'reviewRating')]").text.replace(',','.')
            title2 = container[j].find_element(By.XPATH,value=".//div [contains (@class, 'cor-acco-reviews__review-text-for')]").text
            review2 = container[j].find_element(By.XPATH,value=".//div [contains (@itemprop,'reviewBody')]").text
            scrapedReviews2.append([title2, review2, rating2]) 
        except:
            continue
    driver.find_element(By.XPATH, "//a[@aria-label='Ga naar de volgende pagina']").click()
    #driver.find_element(By.XPATH,value= "a[aria-label='Ga naar de volgende pagina']").click()
scrapedReviewsDF = pd.DataFrame(scrapedReviews2, columns=['title', 'review', 'rating'])

CorendonDF = scrapedReviewsDF.astype({'rating':'float'})
CorendonDF['label'] = CorendonDF['rating'].apply(lambda x: 1 if x > 5 else 0)
CorendonDF = pd.DataFrame(CorendonDF, columns=['title', 'review', 'rating','label'])
driver.quit()
print( 'Ready scraping ....')
corendonpath = os.path.join(Folderpath, path_to_file2)
CorendonDF.to_csv(corendonpath, sep=',',index= False)

#Het scrijven van eigen reviews.
#Het maken van de header
geschrevenpath = os.path.join(Folderpath,"Geschreven_Reviews.csv" )

review_header = ['title','review','rating', 'label']

#De rows met reviews
review_data = pd.DataFrame()
review_data = [
    ['Fantastisch','Ik heb het erg naar mijn zin gehad in dit hotel.',8.0, 1],
    ['Vreselijk','Bijzonder slechte service en vieze kamers.',2.0, 0],
    ['Uitmuntend','Ik vind dit het beste hotel waar ik ooit ben geweest.',10.0, 1],
    ['Matig', 'Het hotels ik OK maar het kan beter.',5.0, 0],
    ['Goed', 'Best wel goed hotel. goede service.',7.0, 1],
    ['Fantastisch','Ik raad dit hotel zeker aan.',8.0, 1],
    ['prima', 'gewoon een goed hoterl.',7.8, 1],
    ['goed', 'Ik zal hier zeker in de toekomst terugkomen.',7.5, 1],
    ['Slecht', 'Ik raad dit hotel niet aan. blijf weg.',2.3, 0],
    ['Meh', 'niet het beste hotel.',5.5, 1]
]
Geschreven_reviews = pd.DataFrame (review_data, columns = review_header)
print(Geschreven_reviews)
with open(geschrevenpath, 'w') as file:
    writer = csv.writer(file)
    writer.writerow(review_header)
    writer.writerows(review_data)

#inlezen van de CSV van Kaggle.
#het droppen van alle onnodige kollommen en samenvoegen van positive review en negative review.
def return_df():
    df = pd.read_csv(r'C:\Users\thijs\Desktop\assignment 1\hotel reviews map\Hotel_Reviews.csv', sep=',')
    df.rename(columns = {'Hotel_Name': 'title', 'Reviewer_Score': 'rating'},inplace = True)
    df.drop(df.columns.difference(['title', 'rating', 'Negative_Review', 'Positive_Review']), 1, inplace=True)
    df = df.melt(id_vars=['title', 'rating'], value_vars=['Positive_Review', 'Negative_Review'], var_name='label', value_name='review')
    df['label'].replace({"Negative_Review": 0, "Positive_Review": 1}, inplace=True)
    df['rating'] = df['rating'].astype(float)
    df['label'] = df['rating'].apply(lambda x: 1 if x > 5 else 0)
    return df


# Een nieuwe CSV maken en in de map zetten met de andere CSV's.
df = return_df()
nieuw_hotel_reviews_path = os.path.join(Folderpath, "Nieuw_Hotel_Reviews.csv")
df.to_csv(nieuw_hotel_reviews_path, sep=',', index=False)

nieuw_hotel_reviews = pd.read_csv(nieuw_hotel_reviews_path)
nieuw_hotel_reviews = nieuw_hotel_reviews.reindex(columns=['title', 'review', 'rating', 'label'])

# De nieuwe dataframe wordt opgeslagen als CSV file.S
nieuw_hotel_reviews.to_csv(nieuw_hotel_reviews_path, index=False)

# Het samenvoegen van alle CSV's in een dataframe.
files = os.path.join("C:\\Users\\thijs\Desktop\\assignment 1\\", "*.csv")
files = glob.glob(files)

print(files)

df1 = pd.concat(map(pd.read_csv, files), ignore_index=True)
df1.rename(columns={'rating': 'rating', 'title': 'title'}, inplace=True)
print(df1)

dataframe_path = os.path.join(Folderpath, "Dataframe.csv")
df1.to_csv(dataframe_path, index=False)

df1.drop_duplicates(inplace=True)

#uploaden van de dataset in de database
USER = 'veldkat1'
PASSWORD = '+#$pPXlpGtwZVM'
HOST = 'oege.ie.hva.nl'
PORT = '3306'
DATABASE = 'zveldkat1'

def upload_data(df1, hotel_reviews):
    engine = create_engine((f'mysql+mysqlconnector://{USER}:{PASSWORD}@{HOST}:{PORT}/{DATABASE}'),
                           connect_args={'connect_timeout':300})
    try:
        df1.to_sql(name=hotel_reviews, con=engine, if_exists='replace', index=False, chunksize=1000)
        print("Data uploaded successfully!")
    except Exception as e:
        print("Something went wrong:", e)

upload_data(df1,"hotel_reviews")


