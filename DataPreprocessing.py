import pandas as pd
from tqdm import tqdm
import re
import nltk
import json
import unidecode
# from textblob import TextBlob

def isWordPresent(sentence):
    s = sentence.split(" ")
    for i in s:
        if (i.lower() == 'who' or i.lower() == 'what' or i.lower() == 'when' or i.lower() == 'where'):
            return True
    return False

def prepro(values):
    max_len = 0
    min_len = 10000
    preprocessed_data = []
    for sentance in tqdm(values):
        sent = expand_sentences(sentance)
        if(max_len<len(sent.strip().split(' '))):
            max_len = len(sent.strip().split(' '))
        if(min_len>len(sent.strip().split(' '))):
            min_len = len(sent.strip().split(' '))
        preprocessed_data.append(sent.strip())
    return (preprocessed_data,max_len, min_len)


def expand_sentences(phrase):
    phrase = phrase.lower()
    phrase = re.sub(r'[àáâãäå]', 'a', phrase)
    phrase = re.sub(r'[èéêë]', 'e', phrase)
    phrase = re.sub(r'[ìíîï]', 'i', phrase)
    phrase = re.sub(r'[òóôõö]', 'o', phrase)
    phrase = re.sub(r'[ùúûü]', 'u', phrase)
    phrase = re.sub(r"won't", "will not", phrase)
    phrase = re.sub(r"won't", "will not", phrase)
    phrase = re.sub(r"can\'t", "can not", phrase)
    phrase = re.sub(r"n\'t", " not", phrase)
    phrase = re.sub(r"\'re", " are", phrase)
    phrase = re.sub(r"\'s", " is", phrase)
    phrase = re.sub(r"\'d", " would", phrase)
    phrase = re.sub(r"\'ll", " will", phrase)
    phrase = re.sub(r"\'t", " not", phrase)
    phrase = re.sub(r"\'ve", " have", phrase)
    phrase = re.sub(r"\'m", " am", phrase)
    phrase = phrase.replace('\\r', ' ')
    phrase = phrase.replace('\\"', ' ')
    phrase = phrase.replace('\\n', ' ')
    # phrase = unidecode.unidecode(phrase)
    # doc = TextBlob(phrase)
    # phrase = str(doc.correct())
    phrase = re.sub('[^A-Za-z0-9]+', ' ', phrase.lower())
    return phrase


def tokenize(sentence):
    return nltk.word_tokenize(sentence)

def answer_span(context,ans):
    ans_token = tokenize(ans)
    con_token = tokenize(context)
    ans_len = len(ans_token)
    
    if ans_len!=0 and ans_token[0] in con_token:
    
        indices = [i for i, x in enumerate(con_token) if x == ans_token[0]]
        try:

            if(len(indices)>1):
                start = [i for i in indices if (con_token[i:i+ans_len] == ans_token) ]
                end = start[0] + ans_len - 1
                return start[0],end

            else:
                start = con_token.index(ans_token[0])
                end = start + ans_len - 1
                return start,end
        except:
            return -1,-1
    else:
        return -1,-1

train = pd.read_json("train.json", encoding='utf-8-sig')
dev = pd.read_json("dev.json", encoding='utf-8-sig')

#train data
contexts = []
questions = []
answers_text = []
answers_start = []
title = []
for i in range(train.shape[0]):
    topic = train.iloc[i,1]['paragraphs']
    title_ = train.iloc[i,1]['title']
    for sub_para in topic:
        for q_a in sub_para['qas']:
            if isWordPresent(q_a['question']):
                questions.append(q_a['question'])
                if len(q_a['answers'])>0 :
                    answers_start.append(q_a['answers'][0]['answer_start']) 
                    answers_text.append(q_a['answers'][0]['text'])
                else:
                    answers_start.append(None)
                    answers_text.append(None)
                contexts.append(sub_para['context'])
                title.append(title_)

# test data
test_contexts = []
test_questions = []
test_answers_text = []
test_answers_start = []
test_title = []

for i in range(dev.shape[0]):
    topic = dev.iloc[i,1]['paragraphs']
    title_ = dev.iloc[i,1]['title']
    for sub_para in topic:
        for q_a in sub_para['qas']:
            if isWordPresent(q_a['question']):
                test_questions.append(q_a['question'])
                if len(q_a['answers'])>0 :
                    test_answers_start.append(q_a['answers'][0]['answer_start']) 
                    test_answers_text.append(q_a['answers'][0]['text'])
                else:
                    test_answers_start.append(None)
                    test_answers_text.append(None)
                test_contexts.append(sub_para['context'])
                test_title.append(title_)


train = pd.DataFrame({"context":contexts, "question": questions, "answer_start": answers_start, "text": answers_text,'title':title})
train.dropna(inplace=True)

dev = pd.DataFrame({"context":test_contexts, "question": test_questions, "answer_start": test_answers_start, "text": test_answers_text,'title':test_title})
dev.dropna(inplace=True)

train_final = pd.DataFrame()
dev_final = pd.DataFrame()

preprocessed_context, max_con_len, min_con_len = prepro(train["context"].values)
train_final["paragraph"] = preprocessed_context

dev_preprocessed_context, max_dev_con_len, min_dev_con_len = prepro(dev["context"].values)
dev_final["paragraph"] = dev_preprocessed_context

preprocessed_question, max_que_len, min_que_len = prepro(train["question"].values)
train_final["question"] = preprocessed_question

dev_preprocessed_question, max_dev_que_len, min_dev_que_len = prepro(dev["question"].values)
dev_final["question"] = dev_preprocessed_question

preprocessed_answer, max_ans_len, min_ans_len = prepro(train["text"].values)
train_final["answer"] = preprocessed_answer

dev_preprocessed_answer, max_dev_ans_len, min_dev_ans_len = prepro(dev["text"].values)
dev_final["answer"] = dev_preprocessed_answer

ans_span = []
for i in range(len(train_final)):
    s,e = answer_span(train_final["paragraph"].iloc[i],train_final["answer"].iloc[i])
    ans_span.append((s,e))

train_final["ans_span"] = ans_span

ans_span = []
for i in range(len(dev_final)):
    s,e = answer_span(dev_final["paragraph"].iloc[i],dev_final["answer"].iloc[i])
    ans_span.append((s,e))
    
dev_final["ans_span"] = ans_span

train_final = train_final[train_final["ans_span"] != (-1,-1)]
dev_final = dev_final[dev_final["ans_span"] != (-1,-1)]

other = pd.DataFrame()
other['max_con_length'] = [max_con_len, max_dev_con_len]
other['min_con_length'] = [min_con_len, min_dev_con_len]
other['max_que_length'] = [max_que_len, max_dev_que_len]
other['min_que_length'] = [min_que_len, min_dev_que_len]
other['max_ans_length'] = [max_ans_len, max_dev_ans_len]
other['min_ans_length'] = [min_ans_len, min_dev_ans_len]

train_final.to_csv('Data/train_final.csv')
dev_final.to_csv('Data/dev_final.csv')
other.to_csv('Data/length_details.csv')

vocabs = {}
count = 0
for k in train_final['paragraph']:
    string = k.split(' ')
    for s in string:
        if s not in vocabs:
            vocabs[s] = count
            count = count + 1

for k in train_final['question']:
    string = k.split(' ')
    for s in string:
        if s not in vocabs:
            vocabs[s] = count
            count = count + 1

for k in train_final['answer']:
    string = k.split(' ')
    for s in string:
        if s not in vocabs:
            vocabs[s] = count
            count = count + 1
            
for k in dev_final['paragraph']:
    string = k.split(' ')
    for s in string:
        if s not in vocabs:
            vocabs[s] = count
            count = count + 1
            
for k in dev_final['question']:
    string = k.split(' ')
    for s in string:
        if s not in vocabs:
            vocabs[s] = count
            count = count + 1
            
for k in dev_final['answer']:
    string = k.split(' ')
    for s in string:
        if s not in vocabs:
            vocabs[s] = count
            count = count + 1
    
with open("Data/word_index.json", "w") as outfile:
    json.dump(vocabs, outfile)