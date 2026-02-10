import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, ConfusionMatrixDisplay

# 1. Load the dataset
df = pd.read_csv('email.csv')

# --- STEP 1: Data Analysis (Counting Spam vs Ham) ---
counts = df['Category'].value_counts()
print(f"--- Dataset Summary ---")
print(f"Total Emails: {len(df)}")
print(f"Ham Emails (Normal): {counts.get('ham', 0)}")
print(f"Spam Emails: {counts.get('spam', 0)}")

# --- STEP 2: Visualization (Saving Graphs) ---

# Graph 1: Bar chart of Spam vs Ham
plt.figure(figsize=(8, 6))
counts.plot(kind='bar', color=['skyblue', 'salmon'])
plt.title('Distribution of Spam vs Ham Emails')
plt.xlabel('Category')
plt.ylabel('Count')
plt.xticks(rotation=0)
plt.savefig('spam_distribution.png') # Saves the graph
print("\n[Saved: spam_distribution.png]")

# Graph 2: Message Length Distribution
df['Length'] = df['Message'].apply(len)
plt.figure(figsize=(10, 6))
df[df['Category'] == 'ham']['Length'].plot(bins=50, kind='hist', color='blue', label='Ham', alpha=0.6)
df[df['Category'] == 'spam']['Length'].plot(bins=50, kind='hist', color='red', label='Spam', alpha=0.6)
plt.title('Email Length Distribution (Ham vs Spam)')
plt.xlabel('Character Length')
plt.legend()
plt.savefig('length_distribution.png') # Saves the graph
print("[Saved: length_distribution.png]")

# --- STEP 3: Model Training ---

# Preprocessing: Convert labels to numbers
df['spam'] = df['Category'].apply(lambda x: 1 if x == 'spam' else 0)

# Split data (80% Train, 20% Test)
X_train, X_test, y_train, y_test = train_test_split(df['Message'], df['spam'], test_size=0.2, random_state=42)

# Vectorization (Converting text to numbers)
cv = CountVectorizer()
X_train_count = cv.fit_transform(X_train.values)

# Train the Model
model = MultinomialNB()
model.fit(X_train_count, y_train)

# --- STEP 4: Evaluation & Confusion Matrix ---

X_test_count = cv.transform(X_test)
y_pred = model.predict(X_test_count)

print(f"\nModel Accuracy: {accuracy_score(y_test, y_pred):.2%}")
print("\nClassification Report:\n", classification_report(y_test, y_pred))

# Graph 3: Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(8, 6))
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['Ham', 'Spam'])
disp.plot(cmap='Blues', ax=plt.gca())
plt.title('Model Performance Confusion Matrix')
plt.savefig('confusion_matrix.png') # Saves the graph
print("[Saved: confusion_matrix.png]")

# --- STEP 5: Manual Testing Function ---
def predict_spam(text):
    data = cv.transform([text])
    prediction = model.predict(data)
    return "Spam" if prediction[0] == 1 else "Ham (Normal)"

# Example test
print("\n--- Manual Test Results ---")
print(f"Message 1: 'Hey, want to go to the park?' -> {predict_spam('Hey, want to go to the park?')}")
print(f"Message 2: 'FREE entry to win cash!' -> {predict_spam('FREE entry to win cash!')}")