import joblib
import pandas as pd
import numpy as np
import re
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.layers import Embedding, Conv1D, GlobalMaxPooling1D, Dense
import flask


# Load the complaints dataset
df = pd.read_csv('customer_complaints.csv')

# Extract features and labels
X = df['Complaint']
y = df['Domain']  # Assuming we're only predicting the 'Domain' here

# Preprocess text data
def preprocess_text(text):
    text = text.lower()  # Convert text to lowercase
    text = re.sub(r'[^a-zA-Z\s]', '', text)  # Remove punctuation
    return text

X = X.apply(preprocess_text)

# Tokenize the text
max_words = 5000
tokenizer = Tokenizer(num_words=max_words)
tokenizer.fit_on_texts(X)
X_sequences = tokenizer.texts_to_sequences(X)

# Pad sequences to ensure uniform length
max_sequence_length = 100  # Adjust according to your data
X_padded = pad_sequences(X_sequences, maxlen=max_sequence_length)

# Encode labels
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_padded, y_encoded, test_size=0.2, random_state=42)

# CNN model architecture
 # Call the OpenAI API to predict severity
    #Embedding Layer:
    #The first layer in the neural network is the Embedding layer.
    #It is responsible for converting integer-encoded words (represented as indices) into dense vectors of fixed size.
    #The size of the embedding vectors (embedding_dim) is set to 100 in this case.
    #The input_length parameter specifies the length of input sequences, which is set to max_sequence_length (100 in this case). 
embedding_dim = 100
filters = 128
kernel_size = 5

model = Sequential()
model.add(Embedding(max_words, embedding_dim, input_length=max_sequence_length))
model.add(Conv1D(filters, kernel_size, activation='relu'))
model.add(GlobalMaxPooling1D())
model.add(Dense(64, activation='relu'))
model.add(Dense(len(label_encoder.classes_), activation='softmax'))  # Softmax for multi-class classification

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',  # Use sparse categorical cross-entropy for integer-encoded labels
              metrics=['accuracy'])

# Train the model
model.fit(X_train, y_train, epochs=10, batch_size=32, validation_split=0.2)

# Evaluate the model on test data
loss, accuracy = model.evaluate(X_test, y_test)
print(f'Test Accuracy: {accuracy}')

# Save the model
model.save('domain_model.h5')

# Save the label encoder
joblib.dump(label_encoder, 'label_encoder.joblib')

# Function to predict labels for input complaint text
# Function to predict labels for input complaint text
def predict_label(complaint_text):
    preprocessed_text = preprocess_text(complaint_text)
    sequence = tokenizer.texts_to_sequences([preprocessed_text])
    padded_sequence = pad_sequences(sequence, maxlen=max_sequence_length)
    predicted_label_index = np.argmax(model.predict(padded_sequence), axis=-1)[0]
    predicted_label = label_encoder.inverse_transform([predicted_label_index])[0]
    return predicted_label

# Example input complaint text
complaint_text = "I got into a fight with a coworker "

# Predict label for the input complaint text
predicted_label = predict_label(complaint_text)

# Print the predicted label
print("Predicted Domain:", predicted_label)
