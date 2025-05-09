import pandas as pd
import csv
import json  
import ollama
import numpy as np
import sys

from prompt_gss import hh_info, map_val2lab, map_views, base_info, task_info,load_mapping
from hh2att import hh_info

#根据对于工作情况的部分问题态度，预测剩余情况态度
def ol(messages):

    stream = ollama.chat(
        model='llama3:8b',
        messages=messages,
        stream=True,
    )
    ans = ""
    for chunk in stream:
        ans = ans +chunk['message']['content']
    return ans
def map_agree(value):
    if value == '':
        return "shows no reply that "
    
    value = int(float(value))
    if value == 1:
        return "strongly disagrees that "
    if value == 2:
        return "disagrees that "
    if value == 3:
        return "neither disagree nor agrees that  "
    if value == 5:
        return "strongly agrees that "
    if value == 4:
        return "agrees that "
    
    #    Here are some examples and answers:
    # 1. This person, identified by ID: 1, 72.0-year-old, female, belonging to the white community.In their family, they have 1.0 siblings, which suggests a lively household.
    # This person is divorced, has an education level of 4 years of college, earns an income of $25,000 or more, is currently Working Full Time, works as a unknown, and belongs to a household type of Single Adult.This person perceives their work meaningful as Disagree,  
    # believes that the prevention of stress should involve all levels of the organization as Disagree,
    # recognizes senior management's support for stress prevention through involvement and commitment as Agree,
    # and feels that they are allowed to change their starting and quitting times on a daily basis often. 
  
    # The answer is 4

    # 2. This person, identified by ID: 3, 57.0-year-old, female, belonging to the white community.In their family, they have 1.0 siblings, which suggests a lively household.
    # This person is divorced, has an education level of 12th grade, earns an income of $25,000 or more, is currently Working Full Time, works as a unknown, and belongs to a household type of unknown.This person perceives their work meaningful as Agree,  
    # believes that the prevention of stress should involve all levels of the organization as Disagree,
    # recognizes senior management's support for stress prevention through involvement and commitment as Disagree,
    # and feels that they are allowed to change their starting and quitting times on a daily basis rarely. 

    # The answer is 2

def wrk_prompt(row):
    #对待工作的部分情况 wrkmeaningfl工作意义, strmgtsup来自上级压力管理,psysamephys身心健康认同度,allogrlevel组织各级别参与压力管理认同度,chngtime工作时间灵活性感知 
    mapping = load_mapping('./GSS/mappings/mapping.json')  
    #
    wrkmeangfl = map_val2lab('wrkmeangfl', row['wrkmeangfl'], mapping) if row['wrkmeangfl'] else 'unknown'
    strmgtsup = map_val2lab('strmgtsup', row['strmgtsup'],mapping) if row['strmgtsup'] else 'unknown'
    psysamephys = map_val2lab('psysamephys', row['psysamephys'], mapping) if row['psysamephys'] else 'unknown'
    allorglevel = map_val2lab('allorglevel', row['allorglevel'],mapping) if row['allorglevel'] else 'unknown'
    chngtime = map_val2lab('chngtime',row['chngtime'],mapping) if row['chngtime'] else 'unknown'
    
    predict = "The prevention of stress should involve all levels of the organization "
    value = row['allorglevel']
    # believes that the prevention of stress should involve all levels of the organization as {allorglevel}, 
    # recognizes senior management's support for stress prevention through involvement and commitment as {strmgtsup}, 
    #agrees with the importance of psychological health being as important as productivity {psysamephys}, 
    wrk_description = f'''This person perceives their work meaningful as {wrkmeangfl},  
recognizes senior management's support for stress prevention through involvement and commitment as {strmgtsup},
and feels that they are allowed to change their starting and quitting times on a daily basis {chngtime}. 
    
    '''
    question = f"You are a statistician and a social survey expert. Considering the specific national conditions of the United States in 2022, Can you tell me if this person agrees on: {predict}" + "? 1.Strongly disagree. 2.Disagree. 3.Neither agree nor disagree. 4.Agree. 5.Strongly agree. Your response should consist of just one number to reflect the person's attitude, without any additional text, explanation or even a space letter or a dot.\n"
    question = question + f'''
 
    For example, if you think this person agrees on: "{predict}", you should response JUST 4 without any additional text and you don't have to explain your choice.
'''
    #print(row['strmgtsup'])
    return wrk_description, question, value

def mood_prompt(row):
    #心情 feelnerv紧张 worry控制担忧 feeldown感觉难过 nointerest没有兴趣
    mapping = load_mapping('./GSS/mappings/mapping.json')  
    
    feelnerv = map_val2lab('mood', row['feelnerv'], mapping) if row['feelnerv'] else 'unknown'
    return 
def main():
# 应用函数到每一行并生成描述性字符串
    folder_path = './Data/GSS/'
    input_path = folder_path + 'GSS2022.csv'
    output_path = folder_path + 'GSS_descriptions_wrk.txt'
    df = pd.read_csv(input_path) 
    responses = []
    gts = []
    with open(input_path, mode='r', newline='') as infile, open(output_path, mode='w', newline='') as outfile:
        reader = csv.DictReader(infile)
        writer = csv.writer(outfile)
        cnt = 0
        #predict
        correct = 0
        for row in reader:
    
            #真实值无直接跳过
            
            _,_,baseInfo_description= base_info(row)
            database_background, _, hh_description,_ = hh_info(row)
            wrk_description,question, real = wrk_prompt(row)
            #print(real)
            if real == '':
                continue
            cnt = cnt+1
            gt = float(real)
            prompt = database_background + baseInfo_description+ hh_description + wrk_description + question

            #print(prompt)
            #outfile.write(prompt)
            
            conversation_history = [{"role": "user", "content": prompt}]
            #response = ol(conversation_history)
            response = float(ol(conversation_history))
            responses.append(response)
            gts.append(float(gt))

            if cnt<=10:
                print("gt ", real)
                print("predict", response)
                outfile.write(f"NO: {cnt}\n")
                outfile.write(prompt)
                outfile.write(f"\n gt:{gt}, response:{response}\n")
                outfile.write("###############################################\n")
            
            print(cnt)
            
            if response == gt:
                correct = correct +1 
            if cnt == 499:
                break
    print("acc: ", correct/cnt)
    responses = np.array(responses)  
    gts = np.array(gts)  

    # # 确保两个数组的长度相同  
    # if responses.shape[0] != gts.shape[0]:  
    #     raise ValueError("responses 和 gts 数组的长度必须相同")  
    # data = np.column_stack((responses, gts))  
    # np.savetxt(folder_path + '/responses/wrk/wrk_few31c.csv', data, delimiter=',', header='responses,gts', comments='', fmt='%s')  
    # print("complete writing data into psy3.csv")
if __name__ == "__main__":
    main()