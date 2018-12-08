import json
import pandas as pd
with open('../data/train_annotations.json', 'r') as f:
    train_json = json.load(f)
with open('../data/train_questions/OpenEnded_abstract_v002_train2015_questions.json', 'r') as f:
    questions_json = json.load(f)
# print(train_json[0])
print("Number of files : ",len(train_json))
base_path="abstract_v002_train2015_000000000000"
train=pd.DataFrame()
index=0
questions={}
for entry in questions_json["questions"]:
    questions[entry["question_id"]]=entry["question"]
print('question dictionary made')
print(questions[55360])
for entry in train_json:
    train.loc[index,"image"]=base_path[:-len(str(entry["image_id"]))]+str(entry["image_id"])+".png"
    train.loc[index,"question"]=questions[entry["question_id"]]
    train.loc[index,"answer"]=entry["multiple_choice_answer"]
    s=str(index+1)+"/"+str(len(train_json))+" done "
    print(s,end="\r")
    index+=1
print(train.head())
train.to_csv("../data/train.csv",index=False)

