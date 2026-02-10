# Spam Email Classifier 📧

A machine learning project that analyzes email data from a CSV file to detect and visualize the difference between **Spam** and **Ham** (normal) messages. The project uses the Naive Bayes algorithm to achieve high accuracy.

## 📊 Dataset Overview (Based on email.csv)
According to the provided dataset, the distribution is as follows:
* **Total Messages:** 5,573
* **Ham (Normal) Messages:** 4,825
* **Spam Messages:** 747

## 🛠️ Installation & Setup
1. **Install required libraries:**

   ```bash
   pip install pandas scikit-learn matplotlib

2.  **Prepare files: Ensure email.csv and the Python script are in the same folder.**

3. Run the project:
   ```bash
   python main.py
   ```

## 🤖 How the Model Works

   This project uses Natural Language Processing (NLP) and the Multinomial Naive Bayes algorithm.

    1. Vectorization: It converts text messages into numbers (word counts).

    2. Training: It learns which words (e.g., "win", "cash", "free") appear most often in spam.

    3. Prediction: It calculates the probability of a new message being spam based on those learned words.


## 🧪 Testing Results

* **The model achieves an accuracy of approximately 98.57%.**

* **Example Spam Detection: "WINNER! You won a £1000 prize" -> Spam 🚩**

* **Example Ham Detection: "Hey, are we still meeting later?" -> Ham (Normal) ✅**



* **Algorithm: Multinomial Naive Bayes (optimized for text classification)**

* **Preprocessing: Text vectorization using CountVectorizer**


* **Visualizations: Automated generation of Confusion Matrices and distribution plots using Matplotlib and Seaborn**

* **Performance: Achieved high precision and recall, effectively minimizing "False Positives"**
**(important for ensuring real emails aren't marked as spam)**


