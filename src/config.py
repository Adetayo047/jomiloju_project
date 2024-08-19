from typing import List
import pickle
from collections import defaultdict
import random
import os

import requests
import joblib
import pandas as pd
import numpy as np
from PIL import Image


USER_IDS = [f'U{str(i).zfill(3)}' for i in range(100)]
user_id = random.choice(USER_IDS)


folder_path = "images"
restaurants = pickle.load(open('artifacts/restaurants_list.pkl','rb'))
similarity = pickle.load(open('artifacts/restaurants_similarity.pkl','rb'))
cf_model = joblib.load('models/cf_model.pkl')


# Extract the first item from each list in the columns
restaurants['cuisine_type'] = restaurants['cuisine_type'].apply(lambda x: x[0] if isinstance(x, list) and len(x) > 0 else x)
restaurants['location'] = restaurants['location'].apply(lambda x: x[0] if isinstance(x, list) and len(x) > 0 else x)
restaurants['price_range'] = restaurants['price_range'].apply(lambda x: x if not isinstance(x, list) else [x[0], x[1]] if len(x) == 2 else x)

restaurants['price_range'] = restaurants['price_range'].apply(
    lambda x: f"{x[0]}-{x[1]}" if isinstance(x, list) and len(x) == 2 else x
)


# Function to load images from a folder
def load_images_from_folder(image_name: str, folder_path: str = folder_path):
    img_path = os.path.join(folder_path, image_name)
    image = Image.open(img_path)
    return image


def fetch_restaurant_poster(restaurant_id):
    restaurant_poster = load_images_from_folder(f"{restaurant_id}.png")
    return restaurant_poster


def run_collabotive_filtering(
    user_id: str, 
    restaurant_ids: List[str], 
    cuisine_type: str = None, 
    location: str = None, 
    price_range: str = None, 
    algo=cf_model, 
    num_results_to_return: int = 5
):
    predictions = []

    for restaurant_id in restaurant_ids:
        # Filter the DataFrame based on the restaurant_id
        filtered_restaurants = restaurants[restaurants['restaurant_id'] == restaurant_id]

        # Skip if the restaurant does not exist
        if filtered_restaurants.empty:
            continue

        restaurant_info = filtered_restaurants.iloc[0]

        # Filter by cuisine_type if specified
        if cuisine_type and restaurant_info['cuisine_type'] != cuisine_type:
            continue

        # Filter by location if specified
        if location and restaurant_info['location'] != location:
            continue

        # Filter by price_range if specified
        if price_range and restaurant_info['price_range'] != price_range:
            continue

        # Make a prediction
        pred = algo.predict(user_id, restaurant_id)
        predictions.append((user_id, restaurant_id, pred.est))

    # If no predictions were made, return an empty dictionary or a default message
    if not predictions:
        return {user_id: []}

    # Get top N predictions
    top_n_predictions = get_top_n(predictions, n=num_results_to_return)

    # Ensure top_n_predictions contains recommendations for the user
    if user_id not in top_n_predictions or not top_n_predictions[user_id]:
        return {user_id: []}

    for i in range(len(top_n_predictions[user_id])):
        restaurant_id = top_n_predictions[user_id][i][0]
        
        # Safeguard: Ensure that there are no out-of-bounds errors
        restaurant_info = restaurants[restaurants['restaurant_id'] == restaurant_id]
        if restaurant_info.empty:
            continue
        
        restaurant_info = restaurant_info.iloc[0]

        # Collect additional details
        restaurant_name = restaurant_info['restaurant_name']
        cuisine_type = restaurant_info['cuisine_type']
        location = restaurant_info['location']
        price_range = restaurant_info['price_range']

        # Extend the list with additional details
        top_n_predictions[user_id][i].extend([restaurant_name, cuisine_type, location, price_range])

    return top_n_predictions



def get_top_n(predictions, n=6):
    top_n = defaultdict(list)
    for uid, iid, est in predictions:
        top_n[uid].append([iid, est])

    # Sort the predictions for each user and retrieve the highest `n` ones.
    for uid, user_ratings in top_n.items():
        user_ratings.sort(key=lambda x: x[1], reverse=True)
        top_n[uid] = user_ratings[:n]

    return top_n


def run_content_based_recommendation(restaurant):
    index = restaurants[restaurants['restaurant_name'] == restaurant].index[0]
    distances = sorted(list(enumerate(similarity[index])), reverse=True, key=lambda x: x[1])
    recommended_restaurant_names = []
    recommended_restaurant_posters = []
    for i in distances[1:6]:
        # fetch the restaurant poster
        restaurant_id = restaurants.iloc[i[0]].restaurant_id
        recommended_restaurant_posters.append(fetch_restaurant_poster(restaurant_id))

        recommended_restaurant_names.append(restaurants.iloc[i[0]].restaurant_name)

    return recommended_restaurant_names, recommended_restaurant_posters
