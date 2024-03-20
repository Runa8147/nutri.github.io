import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Embedding, LSTM
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

# Example data (replace this with your own data)
data = [
    ("What should I eat for breakfast?", "breakfast"),
    ("Can you suggest a healthy lunch?", "lunch"),
    ("I need ideas for dinner tonight.", "dinner"),
    # Add more examples...
]

# Extract input (X) and output (y) from data
X = [text for text, label in data]
y = [label for text, label in data]

# Tokenize input text
tokenizer = Tokenizer()
tokenizer.fit_on_texts(X)
X_sequences = tokenizer.texts_to_sequences(X)

# Pad sequences to ensure uniform length
max_seq_length = max(len(seq) for seq in X_sequences)
X_padded = pad_sequences(X_sequences, maxlen=max_seq_length, padding='post')

# Convert output labels to one-hot encoded vectors
label_set = set(y)
label_to_index = {label: i for i, label in enumerate(label_set)}
index_to_label = {i: label for label, i in label_to_index.items()}
y_encoded = np.array([label_to_index[label] for label in y])

# Define the model architecture
model = Sequential([
    Embedding(input_dim=len(tokenizer.word_index) + 1, output_dim=100, input_length=max_seq_length),
    LSTM(100),
    Dense(len(label_set), activation='softmax')
])

# Compile the model
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Train the model
model.fit(X_padded, y_encoded, epochs=10, batch_size=32)

# Save the trained model
model.save('chatbot_model.h5')

# Save the tokenizer
import pickle
with open('tokenizer.pkl', 'wb') as f:
    pickle.dump(tokenizer, f)

# Define function for generating chatbot response
def generate_response(message):
    if message.lower() in ["hi", "hey"]:
        return "Hey there! Nice to chat with you again. What can I do for you?"
    elif message.lower().startswith("hey"):
        name = message.split(" ")[1]  # Extracting name from message
        return f"Hey {name}! Nice to chat with you again. What can I do for you?"
    elif message.lower() == "what can you do":
        return "I am your personalized AI nutritionist. I can provide you with personalized nutrition recommendations based on your goals and preferences."
    else:
        return "I'm sorry, I didn't understand that. Can you please rephrase?"
