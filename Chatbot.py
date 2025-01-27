import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.utils import pad_sequences
import numpy as np
import json

# Step 1: Load the conversation data from the JSON file
def load_conversations_from_json(json_path):
    with open(json_path, 'r', encoding='utf-8') as file:
        data = json.load(file)

    # Extract patterns and responses from each intent
    conversations = []
    for intent in data['intents']:
        tag = intent.get('tag')  # Safely get the 'tag'
        patterns = intent.get('patterns', [])  # Get 'patterns', default to empty list if not present
        responses = intent.get('responses', [])  # Get 'responses', default to empty list if not present
        
        # Only add conversation if both patterns and responses are available
        if patterns and responses:
            for pattern, response in zip(patterns, responses):
                conversations.append((pattern, response))  # Pair each pattern with a response
    return conversations

# Step 2: Preprocess the text
def preprocess_text(text):
    text = text.lower()
    text = ''.join(char for char in text if char.isalnum() or char.isspace())
    return text

# Step 3: Load and preprocess conversations from the JSON file
file_path = r'C:\Users\amink\Downloads\archive (2)\KB.json'  # Adjust the path if needed
conversations = load_conversations_from_json(file_path)
print(f"Loaded {len(conversations)} conversations.")

# Preprocess the conversations
conversations = [
    (preprocess_text(input_text), preprocess_text(output_text))
    for input_text, output_text in conversations
]
print(f"Sample processed conversations: {conversations[:3]}")

# Step 4: Tokenization and Padding
# Flatten input and output texts
input_texts = [input_text for input_text, _ in conversations]
output_texts = [output_text for _, output_text in conversations]

# Initialize Tokenizer
tokenizer = Tokenizer()
tokenizer.fit_on_texts(input_texts + output_texts)

# Tokenize input and output
input_sequences = tokenizer.texts_to_sequences(input_texts)
output_sequences = tokenizer.texts_to_sequences(output_texts)

# Pad sequences to a uniform length
max_seq_length = max(max(len(seq) for seq in input_sequences), max(len(seq) for seq in output_sequences))
input_padded = pad_sequences(input_sequences, maxlen=max_seq_length, padding='post')
output_padded = pad_sequences(output_sequences, maxlen=max_seq_length, padding='post')

# Shift output for teacher forcing
output_padded_shifted = np.zeros_like(output_padded)
output_padded_shifted[:, :-1] = output_padded[:, 1:]

vocab_size = len(tokenizer.word_index) + 1
print(f"Vocabulary size: {vocab_size}")
print(f"Maximum sequence length: {max_seq_length}")

# Step 5: Define the Model
model = Sequential([
    Embedding(input_dim=vocab_size, output_dim=128, input_length=max_seq_length),
    LSTM(128, return_sequences=True),
    Dropout(0.2),
    LSTM(128, return_sequences=True),
    Dense(vocab_size, activation='softmax')
])

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
print(model.summary())

# Step 6: Train the Model
print("Training the model...")
model.fit(input_padded, output_padded_shifted, epochs=10, batch_size=32, validation_split=0.2)

# Save the trained model
model.save("chatbot_model.h5")
print("Model saved as chatbot_model.h5")

# Step 7: Generate Responses
def generate_response(input_text):
    input_sequence = tokenizer.texts_to_sequences([preprocess_text(input_text)])
    input_padded = pad_sequences(input_sequence, maxlen=max_seq_length, padding='post')
    prediction = model.predict(input_padded, verbose=0)
    predicted_sequence = np.argmax(prediction[0], axis=-1)
    response = ' '.join(tokenizer.index_word[idx] for idx in predicted_sequence if idx != 0)
    return response

# Step 8: Testing the bot
while True:
    user_input = input("You: ")
    if user_input.lower() in ['exit', 'quit']:
        print("Chatbot: Goodbye!")
        break
    response = generate_response(user_input)
    print(f"Chatbot: {response}")
