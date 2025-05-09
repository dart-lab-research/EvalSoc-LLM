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
    mapping = load_mapping('/home/cyyuan/RECS/mappings/mapping.json')
    age = row['HHAGE']
    gender = mapping["HHSEX"].get(row["HHSEX"])  
    employment = mapping['EMPLOYHH'].get(row['EMPLOYHH'])
    state = mapping['state_postal'].get(row['state_postal'])
    hhrace = mapping['HOUSEHOLDER_RACE'].get(row['HOUSEHOLDER_RACE'])
    hhmember = row['NHSLDMEM']
    athome = mapping['ATHOME'].get(row['ATHOME'])
    income = mapping['MONEYPY'].get(row['MONEYPY'])

    background = f'''
The Residential Energy Consumption Survey (RECS), conducted by the U.S. Energy Information Administration (EIA), is a nationally representative dataset that provides detailed insights into household energy usage, costs, and characteristics. 
First launched in 1978, the 2020 RECS cycle collected data from nearly 18,500 households, representing 123.5 million primary residence housing units. 
The survey gathers information on energy usage patterns, housing characteristics, and household demographics through web and mail forms, supplemented by data from energy suppliers.
    '''
    base_info = f'''
This person, who is a {age}-year-old {gender}, whose employment status is {employment} and lives in the state {state}.
Besides, this person's race is {hhrace} and the number of household members is {hhmember}, while {athome}.
Moreover, the total household income is {income}.
'''

    return background, base_info

def generate_description(row):
    mapping = load_mapping('/home/cyyuan/RECS/mappings/mapping.json')

    gas = mapping["UGASHERE"].get(row["UGASHERE"])  
    electricity_cooking = mapping["ELFOOD"].get(row["ELFOOD"])  
    propane_cooking = mapping["LPCOOK"].get(row["LPCOOK"])  
    natural_gas_cooking = mapping["UGCOOK"].get(row["UGCOOK"])

    electricity_used = mapping["USEEL"].get(row["USEEL"])  
    natural_gas_used = mapping["USENG"].get(row["USENG"])  
    propane_used = mapping["USELP"].get(row["USELP"])  
    fuel_oil_used = mapping["USEFO"].get(row["USEFO"])  
    solar_thermal_used = mapping["USESOLAR"].get(row["USESOLAR"])  
    wood_used = mapping["USEWOOD"].get(row["USEWOOD"])  
    all_electric = mapping["ALLELEC"].get(row["ALLELEC"])

    usage_info = f'''
When it comes to energy use, for this person, {gas}. Besides, {electricity_used}, {natural_gas_used} and {propane_used}.
Moreover, for this person, {fuel_oil_used}, {solar_thermal_used} and {wood_used}. At the same time, for this person, {natural_gas_cooking} and {propane_cooking}.
    '''

    pred = row['ELFOOD']
    task = mapping['Tasks'].get("jELFOOD")

    question = f'''
You are a statistician and a social survey expert. I provide you with some basic information about this person and some of his energy or fuel usage in his daily life. 
You need to accurately analyze this person's energy habits and predict {task} based on the information I provide.
    1. Yes
    0. No

    Here are some examples and answers:
    1. This person, who is a 65-year-old female, whose employment status is Retired and lives in the state New Mexico.
Besides, this person's race is White Alone and the number of household members is 2, while the number of weekdays someone is at home most or all of the day is None.
Moreover, the total household income is $60,000 - $74,999.

When it comes to energy use, for this person, Natural gas is available in neighborhood.. Besides, Electricity is used, Natural gas is used and Propane is not used.
Moreover, for this person, Fuel oil is not used, Solar thermal is not used and Wood is not used. At the same time, for this person, Natural gas is used for cooking and Propane is not used for cooking.
    The answer is 0


    2. This person, who is a 79-year-old female, whose employment status is Retired and lives in the state Arkansas.
Besides, this person's race is White Alone and the number of household members is 1, while the number of weekdays someone is at home most or all of the day is 5 days.
Moreover, the total household income is $15,000 - $19,999.

When it comes to energy use, for this person, Natural gas is available in neighborhood.. Besides, Electricity is used, Natural gas is used and Propane is not used.
Moreover, for this person, Fuel oil is not used, Solar thermal is not used and Wood is not used. At the same time, for this person, Natural gas is not used for cooking and Propane is not used for cooking.
   The answer is 1


Your response should consist of just the option (1/0) to reflect the person's opinion, without any additional text, explanation or even a space letter. 
You are not allowed to explain anything of your response. Your entire output should not exceed 1 character.
    If your answer is Yes, your response should just be 1, without any additional text or explanation.
    If your answer is No, your response should just be 0, without any additional text or explanation.    

'''
    
    pred_n = row['KWH']
    #'Total electricity use, in kilowatthours, 2020, including self-generation of solar power, ranging [42.01-184101.84]'
    task_numerical = mapping['Tasks'].get("nKWH")

    numerical = f'''
You are a statistician and a social survey expert. I provide you with some basic information about this person and some of his energy or fuel usage in his daily life. 
You need to accurately analyze this person's energy habits and predict the {task_numerical} of this individual based on the information I provide. 
Besides, you are not allowed to response more than a numerical number. 

Here are some examples and answers:
    1. This person, who is a 65-year-old female, whose employment status is Retired and lives in the state New Mexico.
Besides, this person's race is White Alone and the number of household members is 2, while the number of weekdays someone is at home most or all of the day is None.
Moreover, the total household income is $60,000 - $74,999.

When it comes to energy use, for this person, Natural gas is available in neighborhood.. Besides, Electricity is used, Natural gas is used and Propane is not used.
Moreover, for this person, Fuel oil is not used, Solar thermal is not used and Wood is not used. At the same time, for this person, Natural gas is used for cooking and Propane is not used for cooking.
    The answer is 5243.05

    2. This person, who is a 79-year-old female, whose employment status is Retired and lives in the state Arkansas.
Besides, this person's race is White Alone and the number of household members is 1, while the number of weekdays someone is at home most or all of the day is 5 days.
Moreover, the total household income is $15,000 - $19,999.

When it comes to energy use, for this person, Natural gas is available in neighborhood.. Besides, Electricity is used, Natural gas is used and Propane is not used.
Moreover, for this person, Fuel oil is not used, Solar thermal is not used and Wood is not used. At the same time, for this person, Natural gas is not used for cooking and Propane is not used for cooking.
   The answer is 2387.64

The predictions should be presented as a concise numerical output. No calculation process required. Besides, when outputting numbers, please use pure numeric format and do not add commas. For example, if your answer is '10,000', you should response '10000' without the comma.

    '''
    return usage_info, numerical, pred_n
def main():
# 应用函数到每一行并生成描述性字符串
    folder_path = './Data/RECS/'
    input_path = "/home/cyyuan/Data/RECS/selected_RECS.csv"
    output_path = "/home/cyyuan/Data/RECS/prompt_sample.txt"
    df = pd.read_csv(input_path) 
    responses = []
    gts = []
    with open(input_path, mode='r', newline='') as infile, open(output_path, mode='w', newline='') as outfile:
        reader = csv.DictReader(infile)
        writer = csv.writer(outfile)
        cnt = 0
        correct = 0
        for i,row in enumerate(reader):
            usage_info, question, gt = generate_description(row)
            background, base_info = generate_baseinfo(row)

            if gt == None:
                cnt = cnt +1
                continue
            gt = float(gt)
        
            prompt = background + base_info + usage_info + question
            
            conversation_history = [{"role": "user", "content": prompt}]
            #print(ol(conversation_history))

            response = float(ol(conversation_history))
            
            responses.append(response)
            gts.append(gt)
            
            if response == gt:
                correct = correct+1

            if i<=5:
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
    np.savetxt('/home/cyyuan/Data/RECS/responses/RECS30fewnum.csv', data, delimiter=',', header='responses,gts', comments='', fmt='%s')  
    print("complete writing data into food.csv")
            


if __name__ == "__main__":
    main()