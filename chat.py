import json
import time

import numpy as np
import pandas as pd
from tensorflow import keras
from sklearn.preprocessing import LabelEncoder

import colorama

colorama.init()
from colorama import Fore, Style, Back

import random
import pickle
with open("intents.json") as file:
    data = json.load(file)
import warnings
warnings.filterwarnings('ignore')

print(Fore.GREEN + "ChatBot:" + Style.RESET_ALL, "Hi! Welcome to WorldJobs !! \n\t I am your assistant World "
                                                 "Classic!! \n\t I "
                                                 "will help you "
                                                 "in finding your dream job :)\n\n"
                                                 "\t What is your Good Name ? ")
print(Fore.LIGHTBLUE_EX + "User: " + Style.RESET_ALL, end="")
name = input()
print(Fore.GREEN + "ChatBot:" + Style.RESET_ALL + "Hi " + name + "!! \n\n "
                                                                 "Lets Start with me asking you some of your "
                                                                 "preferences "
                                                                 "and then you "
                                                                 "can ask me anything related to the job :) \n\n"
                                                                 "Ok..\n"
                                                                 "What are your Education Qualifications?"
                                                                 "(Tell me in brief)")
print(Fore.LIGHTBLUE_EX + "User: " + Style.RESET_ALL, end="")
s1 = input()
print(Fore.GREEN + "ChatBot:" + Style.RESET_ALL + "That's Great!!\n\n" + "Tell me something about your skills and "
                                                                         "earlier work experience.")
print(Fore.LIGHTBLUE_EX + "User: " + Style.RESET_ALL, end="")
s2 = input()
print(
    Fore.GREEN + "ChatBot:" + Style.RESET_ALL + "Wow!!\n\n" + "So, what type of jobs are looking for? Explain briefly.")
print(Fore.LIGHTBLUE_EX + "User: " + Style.RESET_ALL, end="")
s3 = input()
print(Fore.GREEN + "ChatBot:" + Style.RESET_ALL + "OK!!\n\n" + "What work do you expect from your dream job?")
print(Fore.LIGHTBLUE_EX + "User: " + Style.RESET_ALL, end="")
s4 = input()

s = s1 + s2 + s3 + s4
# load trained model
model = keras.models.load_model('chat_model')

# load tokenizer object
with open('tokenizer.pickle', 'rb') as handle:
    tokenizer = pickle.load(handle)

# load label encoder object
with open('label_encoder.pickle', 'rb') as enc:
    lbl_encoder = pickle.load(enc)

max_len = 4000
result = model.predict(keras.preprocessing.sequence.pad_sequences(tokenizer.texts_to_sequences([s]),
                                                                  truncating='post', maxlen=max_len))
df1 = pd.read_csv('main_df.csv')
tag = lbl_encoder.inverse_transform([np.argmax(result)])
tag = tag[0]
poss=[]
for i in df1.index:
    if df1['Title'][i] == tag:
        poss.append(i)
pos=np.random.choice(poss)

# try:
#     if np.isnan(df1.loc[pos,'AboutC']):
#         print("No Info")
# except:
#     print(df1.loc[pos,'AboutC'])


print(Fore.GREEN + "ChatBot:" + Style.RESET_ALL + "Ok after applying linear transformations and "
                                                  "laplace transformations(lol), i have found the best job based "
                                                  "on your preferences:\n\n")
time.sleep(2)
print("TITLE: " + df1.loc[pos,'Title'] + "\n")
time.sleep(1)
print("COMPANY: " +
      df1.loc[pos,'Company'] + "\n")
time.sleep(1)
print("JOB DESCRIPTION: " + df1.loc[pos,'JobDescription'] + "\n")
time.sleep(3)
print("JOB REQUIREMENT: "
      + df1.loc[pos,'JobRequirment'])
time.sleep(5)
warnings.filterwarnings('ignore')
warnings.warn('DelftStack')
warnings.warn('Do not show this message')
model1 = keras.models.load_model('bot_model')

# load tokenizer object
with open('tokenizer1.pickle', 'rb') as handle:
    tokenizer1 = pickle.load(handle)

# load label encoder object
with open('label_encoder1.pickle', 'rb') as enc:
    lbl_encoder1 = pickle.load(enc)
warnings.filterwarnings('ignore')

# parameters
max_len = 15
warnings.filterwarnings('ignore')
while True:
    print(Fore.LIGHTBLUE_EX + "User: " + Style.RESET_ALL, end="")
    inp = input()
    warnings.filterwarnings('ignore')
    result1 = model1.predict(keras.preprocessing.sequence.pad_sequences(tokenizer1.texts_to_sequences([inp]),
                                                                        truncating='post', maxlen=max_len))
    result2=result1
    result2 = result2.reshape((result2.shape[1], 1))
    maxi=-1
    for i in result2:
        if i>maxi:
            maxi=i
    if maxi<=0.7:
        time.sleep(5)
        print(Fore.GREEN + "ChatBot:" + Style.RESET_ALL +"Sorry :( I don't understand. We are still in development."
                                                         " Try "
                                                         "again with something i can understand ;)")
        continue
    tag = lbl_encoder1.inverse_transform([np.argmax(result1)])
    tag=tag[0]
    if tag=="goodbye":
        print(Fore.GREEN + "ChatBot:" + Style.RESET_ALL +"Thank you for visiting World Jobs !!")
        break
    elif tag=="company":
        print(Fore.GREEN + "ChatBot:" + Style.RESET_ALL +"Here's the details about the company offering the job:\n"
              "COMPANY NAME: "+df1.loc[pos,'Company']+"\n")
        try:
            if np.isnan(df1.loc[pos, 'AboutC']):
                pass
        except:
            print("ABOUT: "+ df1.loc[pos, 'AboutC'])
    elif tag=="location":
        try:
            if np.isnan(df1.loc[pos, 'Location']):
                print(Fore.GREEN + "ChatBot:" + Style.RESET_ALL +" Sorry but the required"
                                                                 " information is not available :(")
        except:
            print(Fore.GREEN + "ChatBot:" + Style.RESET_ALL + " You have to work in " + df1.loc[pos,'Location'])
    elif tag=="application":
        try:
            if np.isnan(df1.loc[pos, 'ApplicationP']):
                print(Fore.GREEN + "ChatBot:" + Style.RESET_ALL + " Sorry but the required"
                                                                  " information is not available :(")
        except:
            print(Fore.GREEN + "ChatBot:" + Style.RESET_ALL +" The application procedure is as follows: \n"
                  +df1.loc[pos,'ApplicationP'])
    elif tag=="salary":
        try:
            if np.isnan(df1.loc[pos, 'Salary']):
                print(Fore.GREEN + "ChatBot:" + Style.RESET_ALL + " Sorry but the required"
                                                                  " information is not available :(")
        except:
            print(Fore.GREEN + "ChatBot:" + Style.RESET_ALL + " The payscale is : \n" + df1.loc[pos,'Salary'])
    elif tag=="duration":
        try:
            if np.isnan(df1.loc[pos, 'Duration']):
                print(Fore.GREEN + "ChatBot:" + Style.RESET_ALL + " Sorry but the required"
                                                                  " information is not available :(")
        except:
            print(Fore.GREEN + "ChatBot:" + Style.RESET_ALL +" The duration of the job is : \n"+df1.loc[pos,'Duration'])
    else:
        for i in data['intents']:
            if i['tag'] == tag:
                print(Fore.GREEN + "ChatBot:" + Style.RESET_ALL, np.random.choice(i['responses']))

