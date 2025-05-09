#选择题
from prompt_generate import generate_description,generate_usageinfo,map_commentNum,map_creation,map_slot,map_weekdays_trails,map_weekends_trails,map_contentViews
import csv
import pandas as pd
import ollama
import sys
import re
import math
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import entropy 
from scipy.interpolate import make_interp_spline  
# from plot import num_plot1

def info_generate(row):
    numPerAction = map_commentNum(row['number_of_words_per_action'])
    WeekendsWatch = map_weekends_trails(row['weekends_trails_watched_per_day'])
    
    conView = map_contentViews(row['content_views'])
    #prediction
    
    creation = map_score2option(row['creations'])
    WeekdaysWatch = map_score2option(row['weekdays_trails_watched_per_day'])

    slot1 = map_slot(row['slot1_trails_watched_per_day'],1)
    slot2 = map_slot(row['slot2_trails_watched_per_day'],2)
    slot3 = map_slot(row['slot3_trails_watched_per_day'],3)
    slot4 = map_slot(row['slot4_trails_watched_per_day'],4)
    given_info = f"This one {numPerAction}, while at the same time {conView}. Besides, this person {WeekendsWatch}\n"
    question = '''According to the information, can you help me choose the video playback habits of this person? 
                    Your response should consist of just the option A,B,C,D,E or F without any additional text, explanation or even a space letter(Here is an example of a required reponse that you should follow: [1.A,2.B]) 
                    1.  uploading video habit:
                        A. Watch very few videos [0-0.2].
                        B. Watch fewer videos [0.2-0.4].
                        C. Watch some videos [0.4-0.6].
                        D. Watch more videos [0.6,0.8].
                        E. Watch lots of videos [0.8,1.0].
                        F. Watch massive videos [1.0,...].
                    
                    2.  Videos watching on weekdays (per day):
                        A. Watch very few videos [0-0.2].
                        B. Watch fewer videos [0.2-0.4].
                        C. Watch some videos [0.4-0.6].
                        D. Watch more videos [0.6,0.8].
                        E. Watch lots of videos [0.8,1.0].
                        F. Watch massive videos [1.0,...].
                    '''
    return given_info, question ,creation + WeekdaysWatch

def map_score2option(a): #turn to options
    a = float(a)
    if 0 <= a < 0.2:
        return 'A'
    elif 0.2 <= a < 0.4:
        return 'B'
    elif 0.4 <= a < 0.6:
        return 'C'
    elif 0.6 <= a < 0.8:
        return 'D'
    elif 0.8 <= a <= 1.0:
        return 'E'
    else:
        return 'F'

ollama_session = ollama.chat(
    model='llama3.1',
    messages=[],
    stream=True,
)
def safe_float_conversion(s):  
    # 使用正则表达式匹配合法的浮点数  
    match = re.search(r'[-+]?\d*\.?\d+', s)  
    if match:  
        return float(match.group())  
    else:  
        raise ValueError(f"Invalid float format: {s}")  


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
def  generate_numerical(row):
    #given numPeraction,creation,weekdayswatch   predict conview,weekendswatch
    numPerAction = map_commentNum(row['number_of_words_per_action'])
    creation = map_creation(row['creations'])
    WeekdaysWatch = map_weekdays_trails(row['weekdays_trails_watched_per_day'])
    conView = map_contentViews(row['content_views'])
    WeekendsWatch = map_weekends_trails(row['weekends_trails_watched_per_day'])
    #prediction
    
    task = "total num of videos watched(normalized)"
    pred = row['content_views']

    slot1 = map_slot(row['slot1_trails_watched_per_day'],1)
    slot2 = map_slot(row['slot2_trails_watched_per_day'],2)
    slot3 = map_slot(row['slot3_trails_watched_per_day'],3)
    slot4 = map_slot(row['slot4_trails_watched_per_day'],4)

    given_info = f"This one {numPerAction}, while at the same time {creation}. Besides, this person {WeekdaysWatch} and {WeekendsWatch}.\n"
    # Some examples and answers are as follows:

    #             1. Average duration of the videos that this person has watched till date is 93.912. This one uses more words in comments(normalized score 0.714285714, in range of [0.6,0.8]), while at the same time uploads very few videos(normalized score 0.0234375, in range of [0,0.2]). Besides, this person watches very few videos on weekdays per day(normalized score 0.0046875, in range of [0,0.2]) and watches very few videos(normalized score 0.0234375, in range of [0,0.2]).
    #             The answer is 0.0234375

    #             2.  Average duration of the videos that this person has watched till date is 237.921.This one comments with very few words(normalized score 0.0, in range of [0,0.2]), while at the same time uploads very few videos(normalized score 0.0, in range of [0,0.2]). Besides, this person watches very few videos on weekdays per day(normalized score 0.0, in range of [0,0.2]) and watches very few videos(normalized score 0.001814882, in range of [0,0.2]).
    #             The answer is 0.001814882
    question = f'''According to the information, can you help me predict the {task} of this person?  
                Your response should consist of just one normalized number to reflect the person's habit, without any additional text, explanation or even a space letter.

                Here is an example of a required reponse that you should follow: 
                    if you think this person has weak {task}, you should response JUST a float number like 0.032. (or more you think, the number may be bigger)
                So your response should be like 0.032(Without any additional text! Just the number, NOT 0.032. but 0.032)
                '''
    
    return given_info, question ,pred
def main():
    #/home/cyyuan/Data/Trell social media usage/random_features.csv
    folder_path = './Data/Trell social media usage/'
    input_path = folder_path + 'random_features.csv'
    output_path = folder_path + 'responses/numerical.txt'
    fig_path = folder_path + 'pic/'
    responses = []
    preds = []
    introduction_prompt = "Trell is an Indian social media and content creation app that allows users to create and share video content, focusing on travel, lifestyle, and various experiences. Now you are role-playing this person based on the above information."
    
    cnt = 0
    t = 0
    df = pd.read_csv(input_path) 
    with open(input_path, mode='r', newline='') as infile, open(output_path, mode='w', newline='') as outfile:
        reader = csv.DictReader(infile)
        for i,row in enumerate(reader):

            description = generate_description(row)
            given_info, question, pred = generate_numerical(row)
            
            #以防0太多干扰判断
            if float(pred) == 0 or float(pred) >= 1:
                continue
            prompt = introduction_prompt + description + given_info + question + '\n'

            #print(prompt)
            conversation_history = [{"role": "user", "content": prompt}]
            response = safe_float_conversion(ol(conversation_history))
            responses.append(response)
            preds.append(float(pred))
            cnt = cnt+1

            if cnt<=10:
                
                outfile.write(prompt)
                outfile.write(f"response: {response}")
                outfile.write('\n'+pred)
            print(f"No:{i+1}: response:{response} gt:{pred}")
            if cnt>=1000:
                print("total num:",cnt)
                break
    # acc = t/(2*cnt)   #两道题
    # print("acc: ",acc)
    # plt.figure(figsize=(10, 5))  
    responses = np.array(responses)  
    preds = np.array(preds)  

    # 确保两个数组的长度相同  
    if responses.shape[0] != preds.shape[0]:  
        raise ValueError("responses 和 preds 数组的长度必须相同")  
    data = np.column_stack((responses, preds))  
    np.savetxt(folder_path + '/responses/zero_num30.csv', data, delimiter=',', header='responses,preds', comments='', fmt='%s')  
    print("complete writing data into few31.csv")


if __name__ == "__main__":
    main()