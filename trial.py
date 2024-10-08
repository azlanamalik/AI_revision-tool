import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.preprocessing import LabelEncoder

# Example training data
training_data = [
    (['hi'], 'greetings'), (['hello'], 'greetings'), (['hey'], 'greetings'), (['hi'], 'greetings'),
    (['good', 'morning'], 'greetings'), (['good', 'afternoon'], 'greetings'),
    (['good', 'evening'], 'greetings'), (['hello', 'ready', 'study'], 'greetings'),
    (['let', 'start', 'revise'], 'greetings'), (['like', 'study', 'subject'], 'study_topic'),
    (['want', 'revise', 'topic'], 'study_topic'), (['let', 'subject'], 'study_topic'),
    (['study', 'topic'], 'study_topic'), (['need', 'help', 'subject'], 'study_topic'),
    (['want', 'focus', 'topic'], 'study_topic'), (['let', 'review', 'subject'], 'study_topic'),
    (['like', 'study'], 'study_something'), (['want', 'topic'], 'study_something'),
    (['let', 'study'], 'study_something'), (['subject'], 'study_something'),
    (['need', 'study'], 'study_something'), (['want', 'focus'], 'study_something'),
    (['let', 'review', 'material'], 'study_something'), (['mood', 'study'], 'study_something'),
    (['cover'], 'study_something'), (['need', 'learn', 'new'], 'study_something'),
    (['weather', 'like'], 'ask_weather'), (['weather', 'today'], 'ask_weather'),
    (['tell', 'weather'], 'ask_weather'), (['weather', 'forecast'], 'ask_weather'),
    (['go', 'rain', 'today'], 'ask_weather'), (['weather', 'outside'], 'ask_weather'),
    (['tell', 'weather'], 'ask_weather'), (['weather', 'report'], 'ask_weather'),
    (['need', 'umbrella', 'today'], 'ask_weather'), (['temperature', 'outside'], 'ask_weather')
]

# Separate the patterns and intents
patterns, intents = zip(*training_data)
patterns = [' '.join(words) for words in patterns]  # Combine words in each pattern into a single string

# Encode the labels
label_encoder = LabelEncoder()
encoded_intents = label_encoder.fit_transform(intents)

# Tokenize the patterns
tokenizer = Tokenizer(oov_token="<OOV>")
tokenizer.fit_on_texts(patterns)
word_index = tokenizer.word_index
sequences = tokenizer.texts_to_sequences(patterns)

# Pad sequences to ensure uniform input length
max_length = max(len(x) for x in sequences)
padded_sequences = pad_sequences(sequences, maxlen=max_length, padding='post')

print(f"Padded Sequences: {padded_sequences}")
print(f"Word Index: {word_index}")
