import pandas as pd
import csv
import json  
import ollama
import numpy as np
import sys
from prompt_gss import load_mapping, ol,map_val2lab,map_sex

def base_info(row):
    mapping = load_mapping('./GSS/mappings/mapping.json')  
    # id = row['id']
    # age = row['age']
    # sibs = row['sibs']
    # race = map_val2lab('race', float(row['racecen1']), mapping)  
    # sex = map_sex(float(row['sex']), mapping)  

    #deal with empty values
    id = row.get('id', 'Unknown ID')  # Default to 'Unknown ID' if not present  
    age = row.get('age', '')  # Default to empty string if not present  
    sibs = row.get('sibs', '0')  # Default to '0' if not present  
    racecen1 = row.get('racecen1', '')  # Get racecen1 value  
    sex_value = row.get('sex', '')  # Get sex value  

    # Handling racecen1 conversion safely  
    try:  
        race = map_val2lab('race', float(racecen1), mapping) if racecen1 else 'Unknown race'  
    except ValueError:  
        race = 'Invalid race value'  

    # Handling sex conversion safely  
    try:  
        sex = map_sex(float(sex_value), mapping) if sex_value else 'Unknown sex'  
    except ValueError:  
        sex = 'Invalid sex value'  
    baseInfo_parts = []  
    if id:  
        baseInfo_parts.append(f"identified by ID: {id}")  
    if age:  
        baseInfo_parts.append(f"{age}-year-old")  
    if sex:  
        baseInfo_parts.append(sex)  
    if race:  
        baseInfo_parts.append(f"belonging to the {race} community, which is known for its cultural heritage")  
    # Join the parts, ensuring proper sentence structure  
    if baseInfo_parts:  
        baseInfo_description = "This person, " + ", ".join(baseInfo_parts) + ".\n"  
    else:  
        baseInfo_description = "This person has not provided sufficient information.\n"  
    # Constructing the description  
    description_parts = []  
    if sibs:  
        description_parts.append(f"they have {sibs} siblings, which suggests a lively household filled with shared experiences and memories.")  
    else:  
        description_parts.append("they have no siblings, suggesting a quieter household.")  

    description = "In their family, " + " ".join(description_parts) + " This person's background and family dynamics contribute to their unique perspective on life.\n\n"  
    #print(baseInfo_description+description)
    




    database_background = "The General Social Survey (GSS), conducted since 1972 by NORC at the University of Chicago, collects data on American society to track trends in opinions, attitudes, and behaviors. Funded by the NSF, it covers topics like civil liberties, crime, and social mobility, enabling researchers to analyze societal changes over decades and compare the U.S. to other nations.\n"
    role_play_info = "You are a statistician and a social survey expert. I will give you some information about this person's response and attitude toward different policies in GSS2022 dataset, which requires you to accurately analyze this person's behavior and make predictions about other policy responses of this person.\n\n "
    
    return database_background, role_play_info, baseInfo_description, description

def hh_info(row):
    mapping = load_mapping('./GSS/mappings/mapping.json')  
    #婚姻，教育，收入，工作状态，工作职务，家庭构成
    marital = map_val2lab('marital', row['marital'], mapping) if row['marital'] else 'unknown'
    educ = map_val2lab('educ', row['educ'], mapping) if row['educ'] else 'unknown'
    income = map_val2lab('income', row['income'], mapping) if row['income'] else 'unknown'
    wrkstat = map_val2lab('wrkstat', row['wrkstat'], mapping) if row['wrkstat'] else 'unknown'
    occ = map_val2lab('occ10', row['occ10'], mapping) if row['occ10'] else 'unknown'
    hhtype = map_val2lab('hhtype1',row['hhtype1'], mapping) if row['hhtype1'] else 'unknown'

    hh_description = "This person is " + marital +  ", has an education level of " + educ + ", "\
                 "earns an income of " + income + ", is currently " + wrkstat + ", " \
                 "works as a " + occ + ", and belongs to a household type of " + hhtype + "."
    # 
    #prediction
    polviews = row['educ']

    database_background = "The General Social Survey (GSS), conducted since 1972 by NORC at the University of Chicago, collects data on American society to track trends in opinions, attitudes, and behaviors. Funded by the NSF, it covers topics like civil liberties, crime, and social mobility, enabling researchers to analyze societal changes over decades and compare the U.S. to other nations.\n"
    role_play_info = "You are a statistician and a social survey expert. I will give you some information about this person's household information in GSS2022 dataset, which requires you to accurately analyze this person's political views. It's a sevenseven-point scale arranging from extremely liberal--point 1--to extremely conservative--point 7.\n "
    role_play_info = role_play_info + "Your response should consist of just one number (from 1 to 7) to reflect the person's attitude, without any additional text, explanation or even a space letter or a dot. For example, if you think this person has an extremely liberal view, you should response JUST 1 \n"
  
    #print(hh_description)
    # mapping['martial'].get(str(int(row['martial'])), 'unknown')
    


    return database_background, role_play_info, hh_description, polviews

def main():
    folder_path = './Data/GSS/'
    input_path = folder_path + 'selected_GSS.csv'
    output_path = folder_path + 'GSS_descriptions.txt'
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

            #hh_description,gts = hh_info(row)
            database_background, role_play_info, hh_description, gt = hh_info(row)
            #无预测值则继续
            if gt == "":
                continue
            cnt = cnt+1
            prompt = database_background+role_play_info+hh_description

            conversation_history = [{"role": "user", "content": prompt}]
            response = float(ol(conversation_history))

            gt = float(gt)
            gts.append(gt)
            responses.append(response)

            print("cnt: ", cnt, " response: ",response," gts: ",gt)
            
    responses = np.array(responses)  
    gts = np.array(gts) 
    if responses.shape[0] != gts.shape[0]:  
        raise ValueError("responses 和 gts 数组的长度必须相同")  
    data = np.column_stack((responses, gts))  
    np.savetxt(folder_path + '/responses/GSSzero_educ30.csv', data, delimiter=',', header='responses,gts', comments='', fmt='%s')  
    print("complete writing data into choose_educ30.csv")

if __name__ == "__main__":
    main()

