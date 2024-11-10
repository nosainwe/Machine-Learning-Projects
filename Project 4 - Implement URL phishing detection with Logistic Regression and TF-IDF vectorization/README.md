- Defined a custom URL preprocessing function (`url_cleanse`) to clean and tokenize URLs.
- Loaded and shuffled a phishing dataset (URLs and corresponding labels).
- Split the data into training and testing sets for model evaluation.
- Utilized `TfidfVectorizer` with the custom tokenizer to convert URL strings into numerical features.
- Trained a Logistic Regression model on the vectorized training data to classify URLs as phishing or legitimate.
- Evaluated model performance on a test set and printed accuracy.
- Saved the trained model and vectorizer to disk using `pickle` for later use.
- Demonstrated how to load the saved model and vectorizer for URL predictions.

Link to dataset used https://www.kaggle.com/datasets/taruntiwarihp/phishing-site-urls
