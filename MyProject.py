from datasets import load_dataset
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import tensorflow as tf
from sklearn.model_selection import train_test_split

dataset = load_dataset("sonos-nlu-benchmark/snips_built_in_intents")


train_texts = dataset['train']['text']
train_labels = dataset['train']['label']

classes = ['ComparePlaces', 'RequestRide', 'GetWeather', 'SearchPlace', 
           'GetPlaceDetails', 'ShareCurrentLocation', 'GetTrafficInformation', 
           'BookRestaurant', 'GetDirections', 'ShareETA']


train_texts = [str(text) for text in train_texts]

train_texts, test_texts, train_labels, test_labels = train_test_split(train_texts, train_labels, test_size=0.3, random_state=0)

tokenizer = Tokenizer(num_words=10000, oov_token='<OOV>')
tokenizer.fit_on_texts(train_texts)
train_sequences = tokenizer.texts_to_sequences(train_texts)
test_sequences = tokenizer.texts_to_sequences(test_texts)

# Pad sequences to the same length
max_length = 100
train_padded = pad_sequences(train_sequences, maxlen=max_length, padding='post', truncating='post')
test_padded = pad_sequences(test_sequences, maxlen=max_length, padding='post', truncating='post')

# Encode the labels
label_encoder = LabelEncoder()
train_encoded_labels = label_encoder.fit_transform(train_labels)
test_encoded_labels = label_encoder.transform(test_labels)


model = tf.keras.Sequential([
    tf.keras.layers.Embedding(input_dim=10000, output_dim=64, input_length=max_length),
    tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(64, return_sequences=True)),
    tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(32)),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dropout(0.5),
    tf.keras.layers.Dense(len(label_encoder.classes_), activation='softmax')
])

model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])



history = model.fit(train_padded, train_encoded_labels, epochs=100, batch_size=32, validation_split=0.2)

test_loss, test_accuracy = model.evaluate(test_padded, test_encoded_labels)
print(f'Test Accuracy: {test_accuracy:.2f}')

new_texts = ["Show me directions to the nearest gas station", "Find me a table for four for dinner tonight"]
new_sequences = tokenizer.texts_to_sequences(new_texts)
new_padded = pad_sequences(new_sequences, maxlen=max_length, padding='post', truncating='post')

predictions = model.predict(new_padded)
predicted_labels = label_encoder.inverse_transform(predictions.argmax(axis=1))



###### MAKING PREDCITIONS ########
print(classes[predicted_labels[0]])
print(classes[predicted_labels[1]])
