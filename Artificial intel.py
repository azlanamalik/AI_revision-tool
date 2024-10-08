import tensorflow as tf
import numpy as np
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import json
import spacy
import requests
from bs4 import BeautifulSoup
import random
from spacy.matcher import PhraseMatcher
nlp = spacy.load("en_core_web_sm")

#give a response related to the input change the json iteration module

class prediction():
    def __init__(self):
        self.model = tf.keras.models.load_model(r'C:\Users\Azlan\OneDrive\Desktop\programming project for revision\chat_box_AI.keras')
        file_path_enc = r"C:\Users\Azlan\OneDrive\Desktop\programming project for revision\label_txt.txt"
        self.label_encoder_val = self.read_list_from_file(file_path_enc)
        file_path_word  = r'C:\Users\Azlan\OneDrive\Desktop\programming project for revision\words_index.txt'
        self.word_list = self.read_dict_from_file(file_path_word)
        #run first time
        raw_inp = self.raw_input_from_user()
        pre_process_1 = self.pre_process(raw_inp)
        pre_process_2 = self.pre_process2(pre_process_1)
        predicted_res = self.predict(pre_process_2)
        self.return_statement(predicted_res)
        #if it hits a terminating loop
        while (self.terminating_condition(raw_inp,predicted_res)):
            raw_inp = self.raw_input_from_user()
            pre_process_1 = self.pre_process(raw_inp)
            pre_process_2 = self.pre_process2(pre_process_1)
            predicted_res = self.predict(pre_process_2)
            self.return_statement(predicted_res)
        if predicted_res == "ask_weather":
            self.get_weather()
        if predicted_res == "study_something":
            self.Study_topic(raw_inp)
    def Study_topic(self,input_dat):
        topic = self.extract_topic(input_dat)
        topic = topic.lower()
        topic_list = list(topic)
        ##manipulating so can use wiki search
        topic_list[0] = topic_list[0].upper()
        for i in range (0,len(topic_list)):
            if topic_list[i] == ' ':
                topic_list[i] = '_'
        ##
        adjusted_topic = "".join(topic_list)
        self.find_information_wiki(adjusted_topic)

    def find_information_wiki(self,topic):
        Url = 'https://en.wikipedia.org/wiki/'  + topic
        response = requests.get(Url).content
        soup = BeautifulSoup(response, 'html.parser')
        paragraphs = soup.find_all('p')
        wiki_text = ''
        for paragraph in paragraphs:
            wiki_text += paragraph.text
    
        print(wiki_text)
    def extract_topic(self,sentence):
        doc = nlp(sentence)
        
        for token in doc:
            if token.dep_ in ("dobj", "pobj"):
                return " ".join([word.text for word in token.subtree])
        if doc.noun_chunks:
            return doc.noun_chunks[0].text
        
        return None  # If no subject found
    def get_weather(self):
        location = input("enter your country ")
        Url = 'https://www.metoffice.gov.uk/weather/world/'  + location.lower()
        response = requests.get(Url).content
        str_arr = ""
        soup = BeautifulSoup(response, 'html.parser')
        soup = soup.find_all('li')
        for div in soup:
            paragraphs = div.find_all('span') 
            for p in paragraphs:
                str_arr = str_arr + p.text
        array_string = str_arr.split("°")
        array_string.pop(len(array_string) - 1)
        print(array_string)


    def terminating_condition(self,raw_inp,predicted_res):
        if raw_inp == "end" or predicted_res == "ask_weather" or "study_something":
            return False
        else:
            return True
    def pre_process2(self, pre_process1):
        # Convert the tokens in pre_process1 to numbers using self.word_list
        num_array = [self.word_list.get(token,1) for token in pre_process1]
        
        # Convert the list to a NumPy array
        return num_array
    
    def raw_input_from_user(self):
        return input("enter a sentence")
        
    def predict(self,new_input):
        #new imp should be a pre processed sequence of numbers using apdded words
        new_input = [list(new_input[i:i+3]) for i in range(0, len(new_input), 50)]
        new_input = np.array(new_input)
        padded_sequences = pad_sequences(new_input, maxlen=50, padding='post')

        # Predict the intent
        predicted_probabilities = self.model.predict(padded_sequences)
        predicted_class = np.argmax(predicted_probabilities, axis=-1)
        for i in range(len(self.label_encoder_val)):
            if (self.label_encoder_val[i][1] == predicted_class):
                predicted_output = self.label_encoder_val[i][0]
        
        print(predicted_output)
        return predicted_output



    def read_list_from_file(self,file_path):
        try:
            with open(file_path, 'r') as file:
                data = file.read()
                # Convert the string data to a list
                data_list = eval(data)
                
                if isinstance(data_list, list):
                    return data_list
                else:
                    raise ValueError("The file does not contain a valid list.")
        
        except FileNotFoundError:
            print(f"File '{file_path}' not found.")
            return None
        except Exception as e:
            print(f"An error occurred: {e}")
            return None
    def read_dict_from_file(self,file_path):
        """Reads the data from the file and returns it as a dictionary."""
        
        try:
            with open(file_path, 'r') as file:
                data = file.read()
                # Convert the string data to a dictionary
                data_dict = eval(data)
                
                if isinstance(data_dict, dict):
                    return data_dict
                else:
                    raise ValueError("The file does not contain a valid dictionary.")
        
        except FileNotFoundError:
            print(f"File '{file_path}' not found.")
            return None
        except Exception as e:
            print(f"An error occurred: {e}")
            return None
    def pre_process(self,inp_data):
        doc = nlp(inp_data)
        filtered_token = []
        for token in doc:
            if token.is_digit or token.is_stop or token.is_punct:
                continue
            filtered_token.append(token.lemma_)
        return filtered_token
    def return_statement(self,predicted_res):
        file_path = r'C:\Users\Azlan\OneDrive\Desktop\programming project for revision\intents.json'
        with open(file_path, 'r', encoding='utf-8') as file:
            data = json.load(file)
        list_of_responses = []
        for intent in data['intents']:
                for pattern in intent['responses']:
                    if (intent['tag'] == predicted_res):
                        list_of_responses.append(pattern)
        
        random_index = random.randint(0,len(list_of_responses) - 1)
        return_statement = list_of_responses[random_index]
        print(return_statement)
                



def main():
    prediction()

main()