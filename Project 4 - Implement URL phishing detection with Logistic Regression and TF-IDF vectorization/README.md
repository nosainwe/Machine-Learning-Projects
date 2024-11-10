<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    
</head>
<body>

<h1>Phishing URL Detection Project</h1>
<p>This project demonstrates a machine learning approach to identifying phishing URLs. Using a logistic regression model and TF-IDF vectorization, the system analyzes URLs and classifies them as "phishing" or "legitimate." This README provides an overview of the code, setup, and functionality.</p>

<h2>1. Project Structure</h2>
<ul>
    <li><strong>main.py</strong>: Main code file that preprocesses data, trains the model, evaluates accuracy, and saves the model.</li>
    <li><strong>phishing_site_urls.csv</strong>: Dataset file containing URLs labeled as either "phishing" or "legitimate" (replace with your actual dataset).</li>
    <li><strong>model.pkl</strong>: Trained Logistic Regression model saved as a binary file for later use.</li>
    <li><strong>vector.pkl</strong>: TF-IDF vectorizer object saved for re-use during prediction.</li>
</ul>

<h2>2. Code Overview</h2>
<p>The code is split into several key parts for URL cleansing, feature extraction, model training, evaluation, and prediction.</p>

<h3>Importing Libraries</h3>
<p>Libraries like <code>pandas</code> and <code>sklearn</code> are used for data processing, while <code>pickle</code> saves the trained model and vectorizer:</p>
<pre>
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
import pandas as pd
import random
import pickle
</pre>

<h3>Data Preprocessing</h3>
<p>The <code>url_cleanse</code> function is used to clean URLs by converting to lowercase, removing protocols like <code>http://</code>, <code>https://</code>, and <code>www</code>, and then tokenizing:</p>
<pre>
def url_cleanse(url):
    url = url.lower()
    url = re.sub(r'http[s]?://', '', url)
    url = re.sub(r'www\.', '', url)
    url = re.sub(r'[^a-z0-9\s.-]', '', url)
    tokens = re.split(r'[./-]', url)
    return tokens
</pre>

<h3>Dataset Loading and Shuffling</h3>
<p>We load the dataset using <code>pandas</code> and shuffle it to prevent any ordering bias:</p>
<pre>
data_csv = pd.read_csv('/content/phishing_site_urls.csv', sep=',', on_bad_lines='skip')
data_list = data_csv.values.tolist()
random.shuffle(data_list)
</pre>

<h3>Feature Extraction and Model Training</h3>
<p>Using TF-IDF, we vectorize URLs to represent them numerically, and then train a logistic regression model on 80% of the data:</p>
<pre>
url_vectorizer = TfidfVectorizer(tokenizer=url_cleanse)
l_regress = LogisticRegression()
x_train, x_test, y_train, y_test = train_test_split(inputurls, y, test_size=0.2, random_state=42)
x_train_vectorized = url_vectorizer.fit_transform(x_train)
l_regress.fit(x_train_vectorized, y_train)
</pre>

<h3>Model Evaluation</h3>
<p>To evaluate, we vectorize the test URLs and calculate the accuracy score:</p>
<pre>
x_test_vectorized = url_vectorizer.transform(x_test)
l_score = l_regress.score(x_test_vectorized, y_test)
print("Score: {:.2f}%".format(100 * l_score))
</pre>

<h3>Saving the Model and Vectorizer</h3>
<p>Both the trained model and the TF-IDF vectorizer are saved using <code>pickle</code> for future prediction tasks:</p>
<pre>
with open("model.pkl", 'wb') as f:
    pickle.dump(l_regress, f)
with open("vector.pkl", 'wb') as f2:
    pickle.dump(url_vectorizer, f2)
</pre>

<h3>Loading and Using the Model for Predictions</h3>
<p>We load the saved model and vectorizer to predict on new URLs:</p>
<pre>
with open("model.pkl", 'rb') as f1:
    lgr = pickle.load(f1)
with open("vector.pkl", 'rb') as f2:
    url_vectorizer = pickle.load(f2)
inputurls = ['example.com', 'malicioussite.com']
x = url_vectorizer.transform(inputurls)
y_predict = lgr.predict(x)
print(inputurls)
print(y_predict)
</pre>

<h2>3. Setup and Running the Code</h2>
<ol>
    <li><strong>Install required libraries:</strong> Make sure <code>scikit-learn</code>, <code>pandas</code>, and <code>pickle</code> are installed. Use <code>pip install -r requirements.txt</code> if a requirements file is provided.</li>
    <li><strong>Prepare your dataset:</strong> Ensure the <code>phishing_site_urls.csv</code> file is correctly formatted with URL labels.</li>
    <li><strong>Run the main script:</strong> Execute the Python script to preprocess data, train the model, evaluate, and save files.</li>
</ol>

<h2>4. Results and Usage</h2>
<p>Upon successful training, the script will output the model’s accuracy. The saved model and vectorizer can be used for predicting new URLs as phishing or legitimate:</p>
<pre>
# Example usage
inputurls = ['hackthebox.eu', 'facebook.com']
y_predict = lgr.predict(url_vectorizer.transform(inputurls))
</pre>

<h2>5. Troubleshooting</h2>
<ul>
    <li><strong>ConvergenceWarning:</strong> If you receive a convergence warning, consider increasing the max iterations or scaling the data.</li>
    <li><strong>Model Accuracy:</strong> Ensure a balanced dataset for optimal results; uneven data may lead to biased results.</li>
</ul>

<h2>6. Conclusion</h2>
<p>This project demonstrates a fundamental approach to URL classification for phishing detection using logistic regression. It includes custom URL preprocessing, TF-IDF vectorization, and model saving/loading for scalable future use.</p>

</body>
</html>


Link to dataset used https://www.kaggle.com/datasets/taruntiwarihp/phishing-site-urls
