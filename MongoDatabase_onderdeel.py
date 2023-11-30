if __name__ == "__main__":
    import pandas as pd 
    import pymongo
    from pymongo import MongoClient
    import os
    huidige_map = os.getcwd()
    print("Huidige werkmap:", huidige_map)

    # Maak een verbinding met MongoDB
    client = pymongo.MongoClient("mongodb://localhost:27017/")
    db = client["Assignment2"]

    # Upload de Kaggle-dataset (hotel_reviews_Origineel.csv)
    kaggle_data = pd.read_csv(r"C:\Users\thijs\Desktop\assignment 2\datasets\Hotel_Reviews.csv")
    kaggle_collection = db["kaggle_dataset"]
    kaggle_collection.insert_many(kaggle_data.to_dict('records'))

    # Upload de bewerkte dataset (cleaned_data.csv)
    cleaned_data = pd.read_csv(r"C:\Users\thijs\Desktop\assignment 2\datasets\cleaned_data.csv")
    cleaned_collection = db["cleaned_data"]
    cleaned_collection.insert_many(cleaned_data.to_dict('records'))

    # Sluit de MongoDB-verbinding
    print("Klaar met uploaden van data")
    client.close()

import pandas as pd 
import pymongo
from pymongo import MongoClient
import os

# Functie voor het ophalen van de data
def laad_data(data):
    print('\nlaad de data van MongoDB')
    client = pymongo.MongoClient("localhost:27017")
    db = client["Assignment2"]
    collection = db[data]
    return pd.DataFrame(list(collection.find({})))

# Functie voor het uploaden
def upload_data(df, collection_name):
    print('\nUpload data naar MongoDB')
    client = pymongo.MongoClient("localhost:27017")
    db = client["Assignment2"]
    collection = db[collection_name]
    collection.insert_many(df.to_dict('records'))
    print('\ndata is succesvol geupload')

# functie voor de aggregate
def aggregate(collection_name):
    print('\nAggregate')
    client = pymongo.MongoClient("localhost:27017")
    db = client["Assignment2"]
    collection = db[collection_name]
 
    pipeline = [
        {
            "$match": { 'Total_Number_of_Reviews': { '$gt': 5000 } }
        },
    ]
    return pd.DataFrame(list(collection.aggregate(pipeline)))

