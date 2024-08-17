import streamlit as st
import pandas as pd
import os
from config import (
    restaurants,
    run_content_based_recommendation,
    run_collabotive_filtering,
    user_id,
    fetch_restaurant_poster,
)

reviews_file = '../reviews.csv'

# Initialize session state for page navigation
if 'page' not in st.session_state:
    st.session_state.page = 'Home'

# Sidebar Navigation
with st.sidebar:
    st.title("Navigation")
    page = st.selectbox(
        "Select a Page",
        options=['Home', 'Reviews', 'About', 'Contact']
    )
    st.session_state.page = page

# Conditional Rendering Based on Page
if st.session_state.page == 'Home':
    st.title('Restaurants Recommender System')
    st.header(f"Welcome user {user_id}")

    # Filters for collaborative filtering
    st.subheader("Filter Your Recommendations")
    selected_cuisine_type = st.selectbox(
        "Select Cuisine Type",
        options=['All'] + list(restaurants['cuisine_type'].unique())
    )
    selected_location = st.selectbox(
        "Select Location",
        options=['All'] + list(restaurants['location'].unique())
    )
    selected_price_range = st.selectbox(
        "Select Price Range",
        options=['All'] + list(restaurants['price_range'].unique())
    )

    restaurant_names = restaurants['restaurant_name'].values
    restaurant_ids = restaurants['restaurant_id'].values

    # Apply the collaborative filtering with the additional parameters
    filtered_restaurant_ids = restaurant_ids
    if selected_cuisine_type != 'All':
        filtered_restaurant_ids = restaurants[restaurants['cuisine_type'] == selected_cuisine_type]['restaurant_id'].values
    if selected_location != 'All':
        filtered_restaurant_ids = restaurants[restaurants['location'] == selected_location]['restaurant_id'].values
    if selected_price_range != 'All':
        filtered_restaurant_ids = restaurants[restaurants['price_range'] == selected_price_range]['restaurant_id'].values

    cf_restaurants = run_collabotive_filtering(
        user_id=user_id, 
        restaurant_ids=filtered_restaurant_ids,
        cuisine_type=selected_cuisine_type if selected_cuisine_type != 'All' else None,
        location=selected_location if selected_location != 'All' else None,
        price_range=selected_price_range if selected_price_range != 'All' else None
    )

    st.caption("Collaborative Filtering Recommendations")
    cols = st.columns(5)
    
    for idx in range(5):
        if idx < len(cf_restaurants[user_id]):
            with cols[idx]:
                st.text(cf_restaurants[user_id][idx][2])
                restaurant_id = cf_restaurants[user_id][idx][0]
                st.image(fetch_restaurant_poster(restaurant_id))
        else:
            with cols[idx]:
                st.text("No more recommendations available")

    selected_restaurant = st.selectbox(
        "Type or select a restaurant from the dropdown",
        restaurant_names
    )

    if st.button('Show Recommendation'):
        recommended_restaurant_names, recommended_restaurant_posters = run_content_based_recommendation(selected_restaurant)
        st.caption("Content-Based Recommendations")
        cols = st.columns(5)
        
        for idx in range(5):
            if idx < len(recommended_restaurant_names):
                with cols[idx]:
                    st.text(recommended_restaurant_names[idx])
                    image = recommended_restaurant_posters[idx]
                    if image:
                        st.image(image)
            else:
                with cols[idx]:
                    st.text("No more recommendations available")

    # Add a button to navigate to the Reviews page
    if st.button("Leave a Review"):
        st.session_state.page = 'Reviews'

elif st.session_state.page == 'Reviews':
    st.title("User Reviews and Comments")

    st.header("Leave a Review")
    st.subheader("Share your experience with the selected restaurant")

    selected_restaurant = st.selectbox(
        "Type or select a restaurant to review",
        restaurants['restaurant_name'].values
    )

    review_text = st.text_area("Write your review here...", "")
    rating = st.slider("Rate the restaurant", 1, 5, 3)

    if st.button("Submit Review"):
        # Prepare the review data as a DataFrame
        review_data = pd.DataFrame({
            "user_id": [user_id],
            "restaurant_name": [selected_restaurant],
            "review_text": [review_text],
            "rating": [rating],
        })

        # Load existing reviews if the file exists
        if os.path.exists(reviews_file):
            reviews_df = pd.read_csv(reviews_file)
        else:
            reviews_df = pd.DataFrame(columns=["user_id", "restaurant_name", "review_text", "rating"])

        # Append the new review to the dataframe
        reviews_df = pd.concat([reviews_df, review_data], ignore_index=True)

        # Save the updated dataframe back to the CSV file
        reviews_df.to_csv(reviews_file, index=False)

        st.success("Thank you for your review!")

        # Display the review
        st.write(f"**User {user_id}**: {review_text} (Rating: {rating}/5)")

    # Display recent reviews
    st.header("Recent Reviews")
    if os.path.exists(reviews_file):
        reviews_df = pd.read_csv(reviews_file)
        for _, review in reviews_df.iterrows():
            st.write(f"**User {review['user_id']}**: {review['review_text']} (Rating: {review['rating']}/5)")
    else:
        st.write("No reviews yet.")

elif st.session_state.page == 'About':
    st.title("About the Restaurants Recommender System")

    st.markdown("""
    Welcome to the Restaurants Recommender System! Our platform is designed to help you discover new dining experiences tailored to your tastes. Whether you’re in the mood for something familiar or eager to try something new, our system will suggest the perfect restaurant for you.

    **How It Works:**

    - **Collaborative Filtering**: We analyze the dining preferences of other users similar to you and suggest restaurants they loved.
    - **Content-Based Filtering**: Based on the types of restaurants you’ve enjoyed in the past, we recommend places with similar offerings.
    - **User Reviews**: Our platform also allows you to read and write reviews, making it easier to decide where to dine next.

    Our goal is to make dining out as enjoyable and personalized as possible. We hope you enjoy using our system as much as we enjoyed building it!
    """)

elif st.session_state.page == 'Contact':
    st.title("Contact Us")

    st.markdown("""
    We’d love to hear from you! Whether you have questions, feedback, or suggestions, feel free to reach out to us through the form below:
    """)

    st.header("Contact Form")

    # Create the contact form
    name = st.text_input("Your Name")
    email = st.text_input("Your Email")
    message = st.text_area("Your Message")

    if st.button("Submit"):
        if not name or not email or not message:
            st.error("Please fill out all fields before submitting.")
        else:
            # Here, you would typically send the message to your email or store it in a database
            # Since this is a demo, we'll just display the submitted information
            st.success("Thank you for your message! We'll get back to you soon.")
            st.write(f"**Name**: {name}")
            st.write(f"**Email**: {email}")
            st.write(f"**Message**: {message}")
