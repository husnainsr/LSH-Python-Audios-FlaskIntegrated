# This is a sample Python script.
from flask import Flask, render_template,request
import pandas as pd
import numpy as np
import os
import librosa
import pickle

app = Flask(__name__)

# This Function is used to extract MFCC of Audios
def extract_mfcc(path):
    audio,sr =librosa.load(path)
    mfccs=(librosa.feature.mfcc(y=audio, sr=sr))
    mean=np.mean(mfccs,axis=1)
    std=np.std(mfccs,axis=1)
    mfcc_feature=(np.concatenate([mean, std]))
    # mfccs_feature=mfccs.T
    # mfccs_feature=mfccs_feature.flatten()
    threshold = np.median(mfcc_feature)
    bin_mfcc = (mfcc_feature > threshold).astype(int)
    return bin_mfcc

# Random Permutation is genrated through this method
def genrate_random_perm(mfcc_length, number_permutation):
    random_permutation = []
    number_permutation = number_permutation
    mfcc_length = 40
    for i in range(0, number_permutation):
        check = True
        while (check):
            temp = np.random.permutation(mfcc_length)
            if len(random_permutation) > 0:
                for j in range(0, len(random_permutation)):
                    if (temp.all() == random_permutation[j].all()):
                        check = False
                        break
                if (check == False):
                    check = True
                break
            else:
                break
        random_permutation.append(temp)
    return random_permutation
#Genrate Hash takes MFCC and Permutation and return minimum Values from HashTable where 1 is peresent
def genrate_hash(mfcc,random_permutation,number_permutation):
    hash_signature=[]
    for i in range(0,number_permutation):
        index=np.where(mfcc== 1)
        values = np.take(random_permutation[i],index)
        min_value = np.min(values)
        hash_signature.append(min_value)
    return hash_signature

#This buckets Function Divide the Hash Signature into Buckets and Assign the key item to value
def buckets(LSH_table,hash_feature,fileName):
    for i in range(0, len(hash_feature), 4):
        # Sum the next 5 values in the array list
        sum_ = sum(hash_feature[i:i+4])
        current_values = LSH_table.get(sum_)  # retrieve the current values associated with the key
        if current_values is None:
            LSH_table[sum_] = fileName  # add a new key-value pair if the key does not exist
        else:
            current_values=current_values+","+fileName
            LSH_table[sum_]= current_values  # update the values associated with the key 1

# Querry Function Take Audio and threshold value to check along side stored values in LSH_table
def query(audio, threshold, random_permutation, number_permutation, LSH_table, hashing_dict):
    mfcc = extract_mfcc(audio)  # querry mfcc
    queryHash_feature = genrate_hash(mfcc, random_permutation, number_permutation)  # querry hash is genrated
    # print(queryHash_feature)
    keys = []
    for i in range(0, len(queryHash_feature), 4):  # this will sum next values
        # Sum the next 5 values in the array list
        sum_ = sum(queryHash_feature[i:i + 4])
        keys.append(sum_)
    # print(keys)
    AllFiles = []
    for i in range(0, len(keys)):  # this will extract all the key items and genrate unqiue files
        if LSH_table.get(keys[i]) is None:
            continue
        files = LSH_table.get(keys[i]).split(",")
        if len(files) > 0:
            for j in range(0, len(files)):
                AllFiles.append(files[j])
    uniqueFiles = list(set(AllFiles))
    similarites = {}
    # threshold=0
    for i in uniqueFiles:
        dataBaseAudio_Hash = hashing_dict.get(i)
        counter = 0
        for j in range(0, len(queryHash_feature)):
            if (dataBaseAudio_Hash[j] == queryHash_feature[j]):
                counter += 1
        jaccardSimilarity = counter / 20
        if jaccardSimilarity >= threshold:
            similarites[i] = jaccardSimilarity
    max_key = max(similarites.items(), key=lambda x: x[1])[0]
    # print(similarites)
    # print("Nearest Audio could be: ",max_key," with Jaccard Similarity ",similarites.get(max_key))
    r=printAnswer(similarites)
    return r

def printAnswer(dic):
    final_output=[]
    if len(dic)!=0:
        k=[]
        val=[]
        for key, value in sorted(dic.items(), key=lambda x: x[1],reverse=True):
            k.append(key)
            val.append(value)
        s="Q1)Nearest Audio "+str(k[0])+" with Similarity "+ str(val[0]*100)+"%-----"
        s=str(s)
        final_output.append(s)
        if(len(k)>2):
            s="Q2)You Will Also Like these Audios: ",str(k[1])," and "+str(k[2])
            final_output.append(s)
        elif(len(k)>1):
            s="Q2)You Will Also Like these Audio: ",str(k[1])
            final_output.append(s)
        else:
            s='Q2)You Have Unique Taste Try this: 002096.mp3'
            final_output.append(s)
    else:
        s='There is no Audio Matched Try changing the Threshold Value'
        final_output.append(s)
    return final_output

@app.route('/', methods = ['GET'])
def upload():
    return render_template('index.html')

@app.route('/', methods = ['POST'])
def answer():
    audio_query = request.files['aud']
    audio_path = './audios/' + audio_query.filename
    audio_query.save(audio_path)

    #Settings
    mfcc_length = 40  # 1Demensional
    number_permutation = 20  # how many permutation columns are required
    LSH_table = {}
    hashing_dict = {}
    random_permutation = []

    with open('./files/random_perm.pickle', 'rb') as file:
        random_permutation = pickle.load(file)
    with open('./files/LSH_table.pickle', 'rb') as file:
        LSH_table = pickle.load(file)
    with open('./files/hashing_dict.pickle', 'rb') as file:
        hashing_dict = pickle.load(file)

    threshold = 0.9  # Jaccard Similarity Threshold to check similarity between test audio and others
    p=query(audio_path, threshold, random_permutation, number_permutation, LSH_table, hashing_dict)
    concatenated_string = '\n'.join([''.join(t) for t in p])
    # concatenated_string = '\n'.join(p)
    print(concatenated_string)
    return render_template('index.html',output=concatenated_string)


if __name__ == '__main__':
    app.run()


