# Milestone 3

![image](https://github.com/user-attachments/assets/1c8c237f-c447-44f3-ba12-285ef48fcf12)

![image](https://github.com/user-attachments/assets/5e0926f3-251c-4b64-9975-cc0ceb2de842)

# 🍳 FlavourAI: Personalized Recipe Recommender & Insights

Welcome to **Milestone 3** of the Food Recipe Recommendation project!  
This dashboard combines **machine learning**, **NLP**, and **data visualization** to help users explore and discover recipes based on their **nutritional preferences** or **ingredient keywords**.

---

## 📂 Repository Structure

| File/Folder                           | Description                                                             |
|----------------------------------------|-------------------------------------------------------------------------|
| `M3_Tool_Presentation.mp4`             | Demonstration video of the final dashboard.                            |
| `M_2_Food and Recipes Recommender.mp4` | Supplementary video showing keyword-based recommendation features.     |
| `Milestone_1_2_3 Final Report.pdf`     | Comprehensive report covering all project milestones.                  |
| `Milestone_3.ipynb`                    | Jupyter notebook with model training, evaluation, and visualizations.  |
| `streamlit_app.py`                     | Streamlit dashboard source code.                                       |
| `README.md`                            | This documentation file.                                               |

---

Milestone 1: https://github.com/AdityaAdke123/Milestone_Sem2_IDS

Milestone 2: https://github.com/AdityaAdke123/Milestone-2_Project

---

## 🛠️ How to Run

### 1. Clone the repository:

git clone https://github.com/yourusername/Milestone_3.git
cd Milestone_3

### 2. Install dependencies:

pip install -r requirements.txt

### 3. Run the Streamlit app:

streamlit run streamlit_app.py


---

## 🚀 Features

### 🔹 Nutrient-Based Recommendations:
- Select nutrients like **Calories**, **Protein**, **Fat**, etc., and choose desired levels (**Low**, **Medium**, **High**, **Very High**).
- The model predicts and recommends top matching recipes using **Logistic Regression**.

### 🔹 Keyword-Based Recommendations:
- Enter phrases like _"spicy chicken low carb"_ and receive personalized recipe suggestions using **TF-IDF Vectorization** and **Cosine Similarity**.

### 🔹 Review Visualizations:
Explore interactive charts on user reviews, including:
- Rating distributions.
- Review trends over the years.
- Average ratings by year.
- Top reviewers.

### 🔹 Recipe Data Visualizations:
Pre-generated insightful plots:
- Top recipe categories.
- Nutrient correlation heatmaps.
- Common ingredients.
- Average cook times by category.

---

## 📊 Model Integration and Evaluation

- **Logistic Regression** is used for nutrient-level classification to recommend relevant recipes.
- **Fallback Strategy:** If no exact nutrient matches are found, the model suggests recipes based on **probability scores** for best alternatives.
- **Keyword Matching:** Utilizes **TF-IDF** and **Cosine Similarity** for effective text-based recommendations.

---

## 🎥 Demonstration

### 🎬 Main Dashboard Walkthrough:
🎥 `M3_Tool_Presentation.mp4`

### 🎬 Keyword-Based Recommendation Demo:
🎥 `M_2_Food and Recipes Recommender.mp4`

---

## 📝 Final Report

Comprehensive documentation of the project's:
- Problem statement.
- Data preprocessing.
- Model training and evaluation.
- Visualizations.
- Insights and conclusions.

📄 `Milestone_1_2_3 Final Report.pdf`

---

## 🔮 Future Scope

- Integrate **ensemble models** or **deep learning** for better accuracy.
- Deploy the app on **Heroku**, **AWS**, or **Streamlit Cloud**.
- Add **user authentication** for personalized dashboards.
- Connect to **real-time recipe APIs** for dynamic content.
- Implement **feedback loops** to continuously improve recommendations.

---

## 🤝 Acknowledgments

- Part of the **Applied Data Science** coursework.
- Special thanks to mentors, peers, and reviewers for their invaluable feedback.

---

## 📌 Contact

Feel free to reach out via GitHub issues or contact:  
📧 **Aditya Adke**
