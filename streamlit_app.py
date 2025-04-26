import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LogisticRegression

# Apply a custom Streamlit theme
st.set_page_config(page_title="FlavourAI: Recipe Recommender", page_icon="🍽️", layout="wide")

# Custom CSS for styling
st.markdown("""
    <style>
    .main {
        background-color: #fdf6f0;
        color: #333333;
    }
    .stButton > button {
        background-color: #ff7f50;
        color: white;
        border-radius: 8px;
        padding: 10px 20px;
        font-size: 16px;
    }
    .stButton > button:hover {
        background-color: #ff6333;
        color: white;
    }
    .stExpander > div > div {
        background-color: #fffaf0;
        border-radius: 10px;
        padding: 10px;
    }
    .stTabs [role="tab"] {
        background-color: #ffdab9;
        color: black;
        font-weight: bold;
        padding: 10px;
        border-radius: 5px 5px 0 0;
    }
    .stTabs [role="tab"][aria-selected="true"] {
        background-color: #ffa07a;
        color: black;
    }
            
    h1, h2, h3, h4, h5, h6, label, .stMarkdown p {
    color: #333333 !important;
    }
    </style>
""", unsafe_allow_html=True)

# Load datasets
df_recipes = pd.read_csv("updated_recipes_dataset_M3.csv")
df_reviews = pd.read_csv("cleaned_reviews_dataset.csv")

# Convert CookTime to timedelta
df_recipes['CookTime'] = pd.to_timedelta(df_recipes['CookTime'], errors='coerce')

nutrient_column_map = {
    'Calories': 'Calories_Level',
    'Fat': 'FatContent_Level',
    'Cholesterol': 'CholesterolContent_Level',
    'Sodium': 'SodiumContent_Level',
    'Carbohydrate': 'CarbohydrateContent_Level',
    'Fiber': 'FiberContent_Level',
    'Sugar': 'SugarContent_Level',
    'Protein': 'ProteinContent_Level'
}

# Streamlit App Title
st.title("🍳 FlavourAI: Personalized Recipe Recommender & Insights")

# Tabs
tab1, tab2, tab3, tab4 = st.tabs(["Nutrient-Based", "Keyword-Based", "Review Visualizations", "Recipe Visualizations"])
# ------------------------
# Tab 1: Nutrient-Based Recommendation
# ------------------------
with tab1:
    st.header(":mag: Nutrient Level-Based Recommendation")

    col1, col2 = st.columns([3, 3])

    with col1:
        nutrient_options = ['Calories', 'Fat', 'Cholesterol', 'Sodium', 'Carbohydrate', 'Fiber', 'Sugar', 'Protein']
        level_options = ['Low', 'Medium', 'High', 'Very High']
        level_mapping = {'Low': 1, 'Medium': 2, 'High': 3, 'Very High': 4}  # For mapping levels

        nutrient_1 = st.selectbox("Select Primary Nutrient:", nutrient_options, key="nutrient1")
        level_1 = st.selectbox(f"Select Level for {nutrient_1}:", level_options, key="level1")

        add_second_filter = st.checkbox("Add another nutrient filter?")
        if add_second_filter:
            nutrient_2 = st.selectbox("Select Secondary Nutrient:", [n for n in nutrient_options if n != nutrient_1], key="nutrient2")
            level_2 = st.selectbox(f"Select Level for {nutrient_2}:", level_options, key="level2")

    with col2:
        st.markdown("""
        **Why Nutrients Matter?**

        - **Calories** 🔥: Energy provided by food, essential for bodily functions.
        - **Fat** 🥑: Vital for energy storage and hormone regulation.
        - **Cholesterol** 🧈: Helps build cells but should be consumed in moderation.
        - **Sodium** 🧂: Important for fluid balance and nerve function.
        - **Carbohydrate** 🥞: Main source of energy, especially for brain and muscles.
        - **Fiber** 🌾: Aids digestion and promotes gut health.
        - **Sugar** 🍬: Quick energy source but should be limited for better health.
        - **Protein** 🍗: Builds and repairs tissues, crucial for muscle and enzyme function.
        """)
    

    with col1:
        if st.button("Get Nutrient-Based Recommendations"):
            # Vectorize Keywords
            vectorizer = TfidfVectorizer(stop_words='english', max_features=500)
            X = vectorizer.fit_transform(df_recipes["Keywords"].astype(str))
            
            # Define target variable
            y = df_recipes[nutrient_column_map[nutrient_1]]

            # Train Logistic Regression model
            model = LogisticRegression(max_iter=1000, class_weight='balanced')
            model.fit(X, y)

            # Predict probabilities
            pred_probs = model.predict_proba(X)
            level_index = model.classes_.tolist().index(level_1)
            df_recipes['probability'] = pred_probs[:, level_index]

            # Filter for exact nutrient level first
            filtered_df = df_recipes[df_recipes[nutrient_column_map[nutrient_1]] == level_1]

            # Fallback: If no exact matches, use top dishes based on prediction probability
            if filtered_df.empty:
                st.info("No exact matches found. Showing top dishes based on predicted probabilities.")
                filtered_df = df_recipes.sort_values(by='probability', ascending=False).head(10)
            else:
                filtered_df = filtered_df.sort_values(by='probability', ascending=False).head(10)

            # Display the recipes
            if not filtered_df.empty:
                st.markdown(f"<p style='color:black; font-weight:bold;'>Top {len(filtered_df)} Recipes matching your preferences:</p>", unsafe_allow_html=True)
                for _, row in filtered_df.iterrows():
                    with st.expander(f"{row['Name']} (ID: {row['RecipeId']})"):
                        st.write(f"**Cook Time:** {row['CookTime']}")
                        st.write(f"**Category:** {row['RecipeCategory']}")
                        st.write(f"**Description:** {row['Description']}")

                        clean_ingredients = row['RecipeIngredientParts'].replace('c(', '').replace(')', '').replace('"', '').replace("'", "")
                        st.write(f"**Ingredients:** {clean_ingredients}")

                        clean_instructions = row['RecipeInstructions'].replace('c(', '').replace(')', '').replace('"', '').replace("'", "")
                        st.write(f"**Instructions:** {clean_instructions}")
            else:
                st.warning("No recipes found even after fallback.")
# Tab 2: Keyword-Based Recommendation
# ------------------------
with tab2:
    st.header("💡 Keyword-Based Recommendation")

    user_input = st.text_input("Describe what you feel like eating (e.g., 'spicy chicken with high protein'):", key="keyword_input")

    if st.button("Get Keyword-Based Recommendations", key="keyword_button"):
        if user_input:
            vectorizer = TfidfVectorizer(stop_words='english')
            keyword_corpus = df_recipes["Keywords"].astype(str)
            tfidf_matrix = vectorizer.fit_transform(keyword_corpus)
            user_vec = vectorizer.transform([user_input])

            similarity_scores = cosine_similarity(user_vec, tfidf_matrix).flatten()

            if similarity_scores.max() == 0:
                st.warning("No matching recipes found. Try using different keywords like 'chicken', 'pasta', 'low carb'.")
            else:
                df_temp = df_recipes.copy()
                df_temp['similarity'] = similarity_scores
                top_keyword_recipes = df_temp.sort_values(by='similarity', ascending=False).head(10)

                st.markdown("<p style='color:black; font-weight:bold;'>Top 10 Recipes matching your description:</p>", unsafe_allow_html=True)

                for _, row in top_keyword_recipes.iterrows():
                    with st.expander(f"{row['Name']} (ID: {row['RecipeId']})"):
                        st.write(f"**Cook Time:** {row['CookTime']}")
                        st.write(f"**Category:** {row['RecipeCategory']}")
                        st.write(f"**Description:** {row['Description']}")

                        clean_ingredients = row['RecipeIngredientParts'].replace('c(', '').replace(')', '').replace('"', '').replace("'", "")
                        st.write(f"**Ingredients:** {clean_ingredients}")

                        clean_instructions = row['RecipeInstructions'].replace('c(', '').replace(')', '').replace('"', '').replace("'", "")
                        st.write(f"**Instructions:** {clean_instructions}")
        else:
            st.warning("Please enter a description to get recommendations.")

# ------------------------
# Tab 3: Review Visualizations (Enhanced Styling)
# ------------------------
with tab3:
    st.header("📊 Review Data Visualizations")

    # Use wide layout for plots
    col1, col2 = st.columns(2)

    # 1. Rating Distribution
    with col1:
        st.subheader("⭐ Rating Distribution")
        fig, ax = plt.subplots(figsize=(8, 5))
        sns.countplot(data=df_reviews, x='Rating', palette='cubehelix', ax=ax)
        ax.set_title('Distribution of Review Ratings', fontsize=14)
        ax.set_xlabel('Rating', fontsize=12)
        ax.set_ylabel('Number of Reviews', fontsize=12)
        sns.despine()
        st.pyplot(fig)

    # 2. Review Trends Over Time
    with col2:
        st.subheader("📈 Review Trends Over Time")
        fig, ax = plt.subplots(figsize=(8, 5))
        df_reviews.groupby('Year').size().plot(kind='line', marker='o', color='#FF6F61', linewidth=2, ax=ax)
        ax.set_title('Number of Reviews per Year', fontsize=14)
        ax.set_xlabel('Year', fontsize=12)
        ax.set_ylabel('Number of Reviews', fontsize=12)
        ax.grid(visible=True, linestyle='--', alpha=0.7)
        sns.despine()
        st.pyplot(fig)

    # 3. Average Rating Over Time
    st.subheader("📊 Average Rating per Year")
    fig, ax = plt.subplots(figsize=(10, 5))
    df_reviews.groupby('Year')['Rating'].mean().plot(kind='line', marker='o', color='lime', linewidth=2, ax=ax)
    ax.set_title('Average Rating per Year', fontsize=14)
    ax.set_xlabel('Year', fontsize=12)
    ax.set_ylabel('Average Rating', fontsize=12)
    ax.grid(visible=True, linestyle='--', alpha=0.7)
    sns.despine()
    st.pyplot(fig)

    # 4. Top 10 Reviewers
    st.subheader("👤 Top 10 Reviewers")
    fig, ax = plt.subplots(figsize=(10, 6))
    df_reviews['AuthorName'].value_counts().head(10).sort_values().plot(kind='barh', color='#6A5ACD', ax=ax)
    ax.set_title('Top 10 Reviewers by Number of Reviews', fontsize=14)
    ax.set_xlabel('Number of Reviews', fontsize=12)
    ax.set_ylabel('Reviewer', fontsize=12)
    sns.despine()
    st.pyplot(fig)


# ------------------------
# Tab 4: Recipe Visualizations (Display Uploaded Images)
# ------------------------
with tab4:
    st.header("🍽️ Recipe Data Visualizations")

    # 1. Top 10 Recipe Categories
    st.subheader("📊 ROC Curve Comparison")
    st.image("Screenshot 2025-04-25 173648.png", caption="ROC Curve Comparison", use_column_width=True)

    # 2. Top 10 Recipe Categories
    st.subheader("📂 Top 10 Recipe Categories")
    st.image("Screenshot 2025-04-25 005303.png", caption="Top 10 Recipe Categories by Count", use_column_width=True)

    # 3. Nutrient Correlation Heatmap (Numeric Nutrients)
    st.subheader("📊 Correlation Heatmap of Nutritional Content")
    st.image("Screenshot 2025-04-25 005413.png", caption="Correlation Heatmap of Nutritional Content", use_column_width=True)


    # 4. Average Cook Time by Recipe Category
    st.subheader("⏱️ Average Cook Time by Recipe Category")
    st.image("Screenshot 2025-04-25 005430.png", caption="Average Cook Time by Recipe Category", use_column_width=True)

    # 5. Top 20 Most Common Ingredients
    st.subheader("🧂 Top 20 Most Common Ingredients")
    st.image("Screenshot 2025-04-25 005449.png", caption="Top 20 Most Common Ingredients (Without Outliers)", use_column_width=True)

    # 6. Correlation Heatmap for All Nutrient Levels and Numeric Features
    st.subheader("🔥 Correlation Heatmap for Nutrient Levels and Numeric Features")
    st.image("Screenshot 2025-04-25 005521.png", caption="Correlation Heatmap for Numerical Features", use_column_width=True)