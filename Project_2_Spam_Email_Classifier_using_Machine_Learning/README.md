<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Spam Detection Model README</title>
</head>
<body>
    <h1>Spam Detection Model</h1>
    <p>This project is a machine learning model that detects spam messages using a logistic regression algorithm. The model is trained on a dataset of SMS messages, where it learns to classify messages as "spam" or "ham" (not spam).</p>

  <h2>Dependencies</h2>
    <p>Install the following dependencies before running the code:</p>
    <ul>
        <li>NumPy</li>
        <li>Pandas</li>
        <li>Scikit-learn</li>
    </ul>

   <h2>Dataset</h2>
    <p>The dataset used is stored in a CSV file <code>mail_data.csv</code> - https://drive.google.com/file/d/1uzbhec5TW_OjFr4UUZkoMm0rpyvYdhZw/view , with two columns:</p>
    <ul>
        <li><strong>Category:</strong> The classification label ("ham" or "spam").</li>
        <li><strong>Message:</strong> The SMS text message content.</li>
    </ul>

   <h2>Data Preprocessing</h2>
    <ol>
        <li>Load the dataset using Pandas.</li>
        <li>Handle missing data by replacing any <code>null</code> values with an empty string.</li>
        <li>Convert <code>spam</code> messages to label 0 and <code>ham</code> messages to label 1.</li>
        <li>Separate the data into features (messages) and labels (categories).</li>
        <li>Split the data into training and testing sets (80% training, 20% testing).</li>
    </ol>

   <h2>Feature Extraction</h2>
    <p>Text data is converted into numerical features using the <code>TfidfVectorizer</code> from <code>sklearn.feature_extraction.text</code>:</p>
    <ul>
        <li>Lowercase and remove English stop words.</li>
        <li>Transform the text data into a TF-IDF feature matrix for training the model.</li>
    </ul>

  <h2>Model Training</h2>
    <p>A <code>LogisticRegression</code> model is trained on the extracted features from the training set. The labels are converted to integers to work with the model.</p>

   <h2>Model Evaluation</h2>
    <p>The model's accuracy is calculated on both the training and testing datasets to verify its performance:</p>
    <ul>
        <li><strong>Training Accuracy:</strong> <code>96.77%</code></li>
        <li><strong>Testing Accuracy:</strong> <code>96.68%</code></li>
    </ul>

   <h2>Prediction System</h2>
    <p>A sample message can be input to test the model's prediction. The text is transformed into feature vectors, and the model then classifies the message as "Spam" or "Ham."</p>

  <h2>Example Code</h2>
    <pre>
# Example of using the prediction system
input_mail = ["Free entry in 2 a wkly comp to win FA Cup f"]
input_data_features = feature_extraction.transform(input_mail)
prediction = model.predict(input_data_features)

if prediction[0] == 1:
    print("Ham mail")
else:
    print("Spam mail")
    </pre>

   <h2>Accuracy</h2>
    <p>Based on the provided dataset, the model achieved high accuracy, indicating its effectiveness in identifying spam messages.</p>

  <h2>References</h2>
    <p>Scikit-learn documentation and tutorials on text classification were helpful for setting up and training this model.</p>
    <h2>License</h2>
<p>This project is licensed under the MIT License. See the <code>LICENSE</code> file for more details.</p>
</body>
</html>
