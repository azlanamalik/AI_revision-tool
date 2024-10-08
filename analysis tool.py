#import tensorFlow as tf

'''pre processing data'''

import json
import spacy
from nltk import PorterStemmer
import pickle
import tensorflow as tf
import random
import nltk
from nltk.stem import WordNetLemmatizer
import numpy as np
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import os
nlp = spacy.load("en_core_web_sm")
class data_processing:
    def __init__(self,file_path):
        self.words = []
        self.documents = []
        self.classes = []
        json = self.loc_json(file_path)
        X,Y = self.inp_arrays(json)
        print(X,Y)
        self.storing_in_pickle()
    def loc_json(self,file_path):
        #returns the json file but in correct form to 

        with open(file_path, 'r', encoding='utf-8') as file:
            data = json.load(file)
        return data
        


    def inp_arrays(self,data):
        for intent in data['intents']:
            for pattern in intent['patterns']:
                val = self.pre_process(pattern)
                self.words.extend(val)
                self.documents.append((val,intent['tag']))
                if intent['tag'] not in self.classes:
                    self.classes.append(intent['tag'])

        word = set(self.words)
        end_class = set(self.classes)
        return word,end_class
        

    def pre_process(self,inp_data):
        doc = nlp(inp_data)
        filtered_token = []
        for token in doc:
            if token.is_digit or token.is_stop or token.is_punct:
                continue
            filtered_token.append(token.lemma_)
        return filtered_token

    def storing_in_pickle(self):
        pickle.dump(self.words,open('words.pkl','wb'))
        pickle.dump(self.classes,open('classes.pkl','wb'))

class AI:
    def __init__(self,data_processing):
        self.training = []
        self.data_processing = data_processing
        self.classes_outp = [0] * len(self.data_processing.classes)
        ##order of operation
        train_X,train_y,counter,Label_mapping = self.inp_dat_x()##getting in data 
        neural_net_model = self.neural_network(train_X,train_y,counter)
        file_path = r'C:\Users\Azlan\OneDrive\Desktop\programming project for revision\label_txt.txt'
        self.write_to_file(file_path,Label_mapping)


    def write_to_file(self,file_path,data):#will write labels to file
        new_dat = repr(data)
        with open(file_path, 'w') as file:
            file.write(new_dat)
            print(f"Data written to '{file_path}'.")
    def inp_dat_ord(self):
        lemmatizer = WordNetLemmatizer()
        for document in self.data_processing.documents:
            bag = []
            wordPatterns = document[0]
            wordPatterns = [lemmatizer.lemmatize(word.lower()) for word in wordPatterns]
            for word in self.data_processing.words:
                bag.append(1) if word in wordPatterns else bag.append(0)

            outputRow = list(self.classes_outp)
            outputRow[self.data_processing.classes.index(document[1])] = 1
            self.training.append(bag + outputRow)
    
    def inp_dat_x(self):
        training_data = self.data_processing.documents
        # Separate the patterns and intents
        patterns, intents = zip(*training_data)
        patterns = [' '.join(words) for words in patterns]  # Combine words in each pattern into a single string

        # Encode the labels
        label_encoder = LabelEncoder()
        encoded_intents = label_encoder.fit_transform(intents)
        label_mapping = list(zip(label_encoder.classes_, label_encoder.transform(label_encoder.classes_)))
        print(label_mapping)

        # Tokenize the patterns
        tokenizer = Tokenizer(oov_token="<OOV>")
        tokenizer.fit_on_texts(patterns)
        word_index = tokenizer.word_index
        sequences = tokenizer.texts_to_sequences(patterns)

        # Pad sequences to ensure uniform input length
        max_length = 50
        padded_sequences = pad_sequences(sequences, maxlen=max_length, padding='post')
        path = r'C:\Users\Azlan\OneDrive\Desktop\programming project for revision\words_index.txt'
        self.write_to_file(path,word_index)

        print(f"Padded Sequences: {padded_sequences}")
        print(f"Word Index: {word_index}")
        print(self.data_processing.documents)
        classes = []
        counter = 0
        for y_Data in encoded_intents:
            if y_Data not in classes:
                classes.append(y_Data)
                counter = counter + 1
        original_labels = label_encoder.inverse_transform(encoded_intents)
        print(original_labels)
        return padded_sequences,encoded_intents,counter,label_mapping

    def neural_network(self,trainX,trainY,counter):
        model = tf.keras.Sequential([
        tf.keras.layers.Embedding(input_dim=len(trainX) + 1, output_dim=16, input_length=len(trainX[0])),
        tf.keras.layers.GlobalAveragePooling1D(),
        tf.keras.layers.Dense(16, activation='relu'),
        tf.keras.layers.Dropout(0.5),
        tf.keras.layers.Dense(counter, activation='softmax')
        ])

# Compile the model
        model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
        model.fit(trainX, trainY, epochs=100, batch_size=3, verbose=1)
        model.save('chat_box_AI.keras')
        print('Done')
        return model
        






def main():
    file_path = r'C:\Users\Azlan\OneDrive\Desktop\programming project for revision\intents.json'
    data_processing1 =  data_processing(file_path)
    print("hell")
    AI1 = AI(data_processing1)



main()

    

    

    



