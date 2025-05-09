import pandas as pd
import csv
import json
import ollama
import numpy as np
def load_mapping(file_path):  
    with open(file_path, 'r') as file:  
        return json.load(file)  
    
def ol(messages):
    stream = ollama.chat(
        model='llama3',
        messages=messages,
        stream=True,
    )
    ans = ""
    for chunk in stream:
        ans = ans +chunk['message']['content']
    return ans

def generate_baseinfo(row):
    mapping = load_mapping('./Income/mappings/mapping.json')  
    uuid = row['uuid']
    age = row['age']
    gender = mapping["gender"].get(row["gender"])  
    rural = mapping["rural"].get(row["rural"])  
    dem_education_level = mapping['dem_education_level'].get(row['dem_education_level'])
    dem_full_time_job = mapping['dem_full_time_job'].get(row['dem_full_time_job'])
    dem_has_children = mapping['dem_has_children'].get(row['dem_has_children'])

    base_info = f'''
This person, identified by {uuid}, is a {age}-year-old {gender}, {rural} {dem_education_level}, {dem_full_time_job}, and {dem_has_children}. 
'''
    return base_info

def generate_description(row):
    mapping = load_mapping('./Income/mappings/mapping.json')  

    question_bbi_2016wave4_basicincome_awareness = mapping['question_bbi_2016wave4_basicincome_awareness'].get(row['question_bbi_2016wave4_basicincome_awareness'])
    question_bbi_2016wave4_basicincome_vote = mapping['question_bbi_2016wave4_basicincome_vote'].get(row['question_bbi_2016wave4_basicincome_vote'])
    question_bbi_2016wave4_basicincome_effect = mapping['question_bbi_2016wave4_basicincome_effect'].get(row['question_bbi_2016wave4_basicincome_effect'])
    question_bbi_2016wave4_basicincome_argumentsfor = mapping['question_bbi_2016wave4_basicincome_argumentsfor'].get(row['question_bbi_2016wave4_basicincome_argumentsfor'])
    question_bbi_2016wave4_basicincome_argumentsagainst = mapping['question_bbi_2016wave4_basicincome_argumentsagainst'].get(row['question_bbi_2016wave4_basicincome_argumentsagainst'])

    # Question 3: What could be the most likely effect of basic income on your work choices? I would...
    # The individual's answer is {question_bbi_2016wave4_basicincome_effect}
    # Question 2: Which of the following arguments FOR the basic income do you find convincing?
    # The individual's answer is {question_bbi_2016wave4_basicincome_argumentsfor}
    que = f'''how would this person vote if there would be a referendum on introducing basic income today
'''
    part_responses = f'''
Their responses to the survey questions are as follows:
Question 1: How familiar are you with the concept known as "basic income"
The individual's answer is {question_bbi_2016wave4_basicincome_awareness}
additional information about basic income: "A basic income is an income unconditionally paid by the government to every individual regardless of whether they work and irrespective of any other sources of income. It replaces other social security payments and is high enough to cover all basic needs (food, housing etc.)."

Question 2: Which of the following arguments FOR the basic income do you find convincing?
The individual's answer is {question_bbi_2016wave4_basicincome_argumentsfor}

Question 3: What could be the most likely effect of basic income on your work choices?
The individual's answer is {question_bbi_2016wave4_basicincome_effect}
'''

    roleplay_setting = f'''
You are a statistician and a social survey expert. I will give you some information about this person's demographic background and their responses to several questions regarding basic income in the basic_income dataset, 
which requires you to accurately analyze this person's perspective and predict {que}.
'''
    background = '''
This dataset is based on a European public opinion study on basic income conducted by Dalia Research in April 2016 across 28 EU Member States. 
The survey included 9,649 participants aged 14-65, with the sample designed to reflect population distributions for age, gender, region/country, education level (ISCED 2011), and urbanization (rural vs. urban).

'''

    return background, part_responses, roleplay_setting

def generate_questions(row):
    mapping = load_mapping('./Income/mappings/mapping.json')  
    question_bbi_2016wave4_basicincome_argumentsfor = mapping['question_bbi_2016wave4_basicincome_argumentsfor'].get(row['question_bbi_2016wave4_basicincome_argumentsfor'])
    
    
    #更改pred处
    #pred = mapping['answer_bbi_2016wave4_basicincome_argumentsagainst'].get(row['question_bbi_2016wave4_basicincome_argumentsagainst'])
    pred = mapping['answer_bbi_2016wave4_basicincome_vote'].get(row['question_bbi_2016wave4_basicincome_vote'])
    #pred = mapping['answer_bbi_2016wave4_basicincome_effect'].get(row['question_bbi_2016wave4_basicincome_effect'])
    #pred = mapping['answer_bbi_2016wave4_basicincome_argumentsfor'].get(row['question_bbi_2016wave4_basicincome_argumentsfor'])
    que = f'''how would this person vote if there would be a referendum on introducing basic income today
'''
    prediction_task = f'''
    Predict {que}, using the predefined answer options below.
    
    Answer options:
    1. I would vote for it
    2. I would probably vote for it
    3. I would probably vote against it
    4. I would vote against it
    5. I would not vote

Here are some examples and answers:

1. This person, identified by f6e7ee00-deac-0133-4de8-0a81e8b09a82, is a 61-year-old male., This person lives in a rural area. This person doesn't have a formal education., This person does not have a full-time job., and There are no children in the individual's current household.. 
The answer is 5

2. This person, identified by 54f0f1c0-dda1-0133-a559-0a81e8b09a82, is a 57-year-old male., This person lives in an urban area. This person has a high level of formal education., This person has a full-time job., and There are children in the individual's current household.. 
The answer is 2

    Your response should consist of just the option (1/2/3/4/5) to reflect the person's opinion, without any additional text, explanation or even a space letter. You are not allowed to explain anything of your response. Your entire output should not exceed 1 character.
    If your answer is 1, your response should just be 1, without any additional text or explanation.
    If your answer is 3, your response should just be 3, without any additional text or explanation.    

    '''
    return prediction_task, pred
def main():
# 应用函数到每一行并生成描述性字符串
    folder_path = './Data/Income/'
    input_path = folder_path + 'basic_income_dataset_dalia.csv'
    output_path = folder_path + 'income_descriptions_vote.txt'
    df = pd.read_csv(input_path) 
    responses = []
    gts = []
    with open(input_path, mode='r', newline='') as infile, open(output_path, mode='w', newline='') as outfile:
        reader = csv.DictReader(infile)
        writer = csv.writer(outfile)
        cnt = 0
        correct = 0
        for i,row in enumerate(reader):
            background, part_response, roleplay_setting = generate_description(row)
            baseinfo = generate_baseinfo(row)
            pred_task,gt = generate_questions(row)

            if gt == None:
                cnt = cnt +1
                continue
            gt = float(gt)
            
            print("ground truth: ", gt)
            prompt = background + baseinfo + part_response + roleplay_setting+ pred_task
            
            conversation_history = [{"role": "user", "content": prompt}]
            #print(ol(conversation_history))

            response = float(ol(conversation_history)[0])
            
            responses.append(response)
            gts.append(gt)
            
            if response == gt:
                correct = correct+1

            if i<=10:
                print(f"No:{i}")
                print("response: ",response)
                print("gt: ",gt)
                outfile.write(prompt)
                outfile.write(f"response: {response}\n")
                outfile.write(f"gt: {gt}\n")
            else:
                print(f"No:{i}")
            
            if i>=1000:
                break
    cnt = i - cnt
    acc = correct/cnt
    print("total num: ", cnt, "\nacc: ", acc)

    responses = np.array(responses)  
    gts = np.array(gts)  
    
    # 确保两个数组的长度相同  
    if responses.shape[0] != gts.shape[0]:  
        raise ValueError("responses 和 gts 数组的长度必须相同")  
    data = np.column_stack((responses, gts))  
    np.savetxt(folder_path + 'responses/baseincome_p3.csv', data, delimiter=',', header='responses,gts', comments='', fmt='%s')  
    print("complete writing data into few.csv")
            


if __name__ == "__main__":
    main()