import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import matplotlib.pyplot as plt
import seaborn as sns

# Load datasets
df_recipes = pd.read_csv("updated_recipes_dataset_M3.csv")
df_reviews = pd.read_csv("cleaned_reviews_dataset.csv")

# Convert CookTime to timedelta
df_recipes['CookTime'] = pd.to_timedelta(df_recipes['CookTime'], errors='coerce')

# Streamlit App Title
st.title("FlavourAI: Personalized Recipe Recommender & Insights")

# Tabs
tab1, tab2, tab3 = st.tabs(["Nutrient-Based", "Keyword-Based", "Review Visualizations"])

# ------------------------
# Tab 1: Nutrient-Based Recommendation
# ------------------------
with tab1:
    st.header("🔍 Nutrient Level-Based Recommendation")

    nutrient_options = ['Calories', 'Fat', 'Cholesterol', 'Sodium', 'Carbohydrate', 'Fiber', 'Sugar', 'Protein']
    level_options = ['Low', 'Medium', 'High', 'Very High']

    nutrient_1 = st.selectbox("Select Primary Nutrient:", nutrient_options, key="nutrient1")
    level_1 = st.selectbox(f"Select Level for {nutrient_1}:", level_options, key="level1")

    add_second_filter = st.checkbox("Add another nutrient filter?")
    if add_second_filter:
        nutrient_2 = st.selectbox("Select Secondary Nutrient:", [n for n in nutrient_options if n != nutrient_1], key="nutrient2")
        level_2 = st.selectbox(f"Select Level for {nutrient_2}:", level_options, key="level2")

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

    if st.button("Get Nutrient-Based Recommendations"):
        filtered_df = df_recipes[df_recipes[nutrient_column_map[nutrient_1]] == level_1]
        if add_second_filter:
            filtered_df = filtered_df[filtered_df[nutrient_column_map[nutrient_2]] == level_2]

        top_recipes = filtered_df[['RecipeId', 'Name', 'CookTime', 'Description', 'RecipeCategory', 'RecipeIngredientParts', 'RecipeInstructions']].head(10)

        if not top_recipes.empty:
            st.success(f"Top {len(top_recipes)} Recipes matching:")
            for _, row in top_recipes.iterrows():
                with st.expander(f"{row['Name']} (ID: {row['RecipeId']})"):
                    st.write(f"**Cook Time:** {row['CookTime']}")
                    st.write(f"**Category:** {row['RecipeCategory']}")
                    st.write(f"**Description:** {row['Description']}")

                    clean_ingredients = row['RecipeIngredientParts'].replace('c(', '').replace(')', '').replace('"', '').replace("'", "")
                    st.write(f"**Ingredients:** {clean_ingredients}")

                    clean_instructions = row['RecipeInstructions'].replace('c(', '').replace(')', '').replace('"', '').replace("'", "")
                    st.write(f"**Instructions:** {clean_instructions}")
        else:
            st.warning("No recipes found for the selected nutrient levels.")

# ------------------------
# Tab 2: Keyword-Based Recommendation
# ------------------------
with tab2:
    st.header("💡 Keyword-Based Recommendation")

    user_input = st.text_input("Describe what you feel like eating (e.g., 'spicy chicken with high protein'):")

    if st.button("Get Keyword-Based Recommendations"):
        if user_input:
            vectorizer = TfidfVectorizer(stop_words='english')
            keyword_corpus = df_recipes["Keywords"].astype(str)
            tfidf_matrix = vectorizer.fit_transform(keyword_corpus)
            user_vec = vectorizer.transform([user_input])

            similarity_scores = cosine_similarity(user_vec, tfidf_matrix).flatten()

            if similarity_scores.max() == 0:
                st.warning("No matching recipes found. Try using different keywords like 'chicken', 'pasta', 'low carb'.")
            else:
                df_recipes['similarity'] = similarity_scores
                top_keyword_recipes = df_recipes.sort_values(by='similarity', ascending=False).head(10)

                st.success("Top 10 Recipes matching your description:")
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


