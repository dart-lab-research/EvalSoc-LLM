import pandas as pd
import csv
import json  
# import ollama
import numpy as np
import sys
import os

from openai import OpenAI
# Mapping = "D://School//UM//LLM//RECS//mappings//mapping.json"
# 调用 GPT-3.5 模型  
client = OpenAI(api_key=apik)
def ask_gpt(client, messages):
    response = client.chat.completions.create(
        model="gpt-4-turbo",
        messages=messages,
        temperature=0,
        top_p=0,
        seed=0
    )
    return response.choices[0].message.content


# def ol(messages):
#     stream = ollama.chat(
#         model='llama3',
#         messages=messages,
#         stream=True,
#     )
#     ans = ""
#     for chunk in stream:
#         ans = ans +chunk['message']['content']
#     return ans

def generate_background():

    database_background = '''
   The dataset was collected in 2013 by students of the Statistics class at FSEV UK. It includes responses from 1010 Slovakian participants aged 15-30, containing 150 variables across various categories: music preferences, movie preferences, hobbies, phobias, health habits, personality traits, spending habits, and demographics.
   The data explores relationships between preferences and behaviors, with research questions covering clustering, hypothesis testing, predictive modeling, and recommendations. It offers insights into youth behaviors, personality traits, and spending patterns, making it useful for sociological and marketing research.
    '''

#         The distribution ratios of the options selected by respondents on some other questions are as follows:
# The resources the country spends on the military, armaments and defense:
# [1: 0.285, 2: 0.389, 3: 0.325]
# The resources the country spends on halting the rising crime rate:
# [1: 0.726, 2: 0.196, 3: 0.078]
# The resources the country spends on improving the nation's education system:
# [1: 0.75, 2: 0.186, 3: 0.064]
    role_playing_task_setting_zero = '''
   You are a data scientist and behavioral analyst. Your task is to generate a synthetic dataset for movie preferences, specifically focusing on the four genres: Action, Documentary, Thriller, and Comedy.
   You will not be provided with any specific distributional information or feature statistics. 
   Based on your understanding of general movie preferences and patterns, generate plausible values for these four movie genres, simulating how individuals might rate their enjoyment or interest in each genre. Use your expertise to create a balanced and realistic dataset that reflects potential viewing habits and preferences across a range of individuals. '''
    role_playing_task_setting_few = '''
    You are a data scientist and behavioral analyst. I will provide you with the statistical distribution of various music preferences and movie preferences, including genres such as Slow songs or fast songs, Dance, Pop, Rock, Horror, Romantic, Sci-fi, and others.
    Based on these distributions, your task is to generate a synthetic dataset reflecting these preferences. Specifically, you need to simulate the preferences for the Action, Documentary, Thriller, and Comedy movie genres. The generated dataset should follow the provided distributions, ensuring that the simulated movie preferences align with realistic patterns observed in real-world surveys.
    Please make sure the dataset captures plausible combinations and maintains consistency with the provided distribution, simulating how individuals might rate their enjoyment of these four movie genres.
    '''

    Distribution_info = ''' 
    I will inform you in advance about the distribution of various music and movie preferences as follows:
    Music Preferences Distribution:
    Music: [1: 0.0045, 2: 0.0089, 3: 0.0326, 4: 0.1276, 5: 0.8264]
    Slow songs or fast songs: [1: 0.0223, 2: 0.0593, 3: 0.6053, 4: 0.2270, 5: 0.0861]
    Dance: [1: 0.1009, 2: 0.2255, 3: 0.2982, 4: 0.2448, 5: 0.1306]
    Folk: [1: 0.2953, 2: 0.3309, 3: 0.2374, 4: 0.0875, 5: 0.0490]
    Country: [1: 0.3338, 2: 0.3620, 3: 0.1899, 4: 0.0816, 5: 0.0326]
    Classical music: [1: 0.1291, 2: 0.2478, 3: 0.2819, 4: 0.1944, 5: 0.1469]
    Musical: [1: 0.2018, 2: 0.2433, 3: 0.2685, 4: 0.1662, 5: 0.1202]
    Pop: [1: 0.0593, 2: 0.1588, 3: 0.2715, 4: 0.2923, 5: 0.2181]
    Rock: [1: 0.0430, 2: 0.1083, 3: 0.2092, 4: 0.2967, 5: 0.3427]
    Metal or Hardrock: [1: 0.3917, 2: 0.2166, 3: 0.1499, 4: 0.1291, 5: 0.1128]
    Punk: [1: 0.3145, 2: 0.2344, 3: 0.2211, 4: 0.1484, 5: 0.0816]
    Hiphop, Rap: [1: 0.2077, 2: 0.2062, 3: 0.2166, 4: 0.2315, 5: 0.1380]
    Reggae, Ska: [1: 0.1736, 2: 0.2567, 3: 0.2908, 4: 0.1855, 5: 0.0935]
    Swing, Jazz: [1: 0.2018, 2: 0.2359, 3: 0.2626, 4: 0.2003, 5: 0.0994]
    Rock n roll: [1: 0.1053, 2: 0.1869, 3: 0.3012, 4: 0.2522, 5: 0.1543]
    Alternative:[1: 0.1914, 2: 0.2285, 3: 0.2448, 4: 0.1795, 5: 0.1558]
    Latino: [1: 0.1884, 2: 0.2685, 3: 0.2300, 4: 0.1677, 5: 0.1454]
    Techno, Trance: [1: 0.3858, 2: 0.2092, 3: 0.1958, 4: 0.1320, 5: 0.0772]
    Opera: [1: 0.3769, 2: 0.2967, 3: 0.1721, 4: 0.0979, 5: 0.0564]
    Movie Preferences Distribution:
    Movies: [1: 0.0030, 2: 0.0045, 3: 0.0697, 4: 0.1958, 5: 0.7270]
    Horror: [1: 0.2552, 2: 0.2107, 3: 0.2240, 4: 0.1617, 5: 0.1484]
    Romantic: [1: 0.0608, 2: 0.1647, 3: 0.2656, 4: 0.2582, 5: 0.2507]
    Sci-fi: [1: 0.1306, 2: 0.2107, 3: 0.2507, 4: 0.2181, 5: 0.1899]
    War: [1: 0.1261, 2: 0.2033, 3: 0.2582, 4: 0.1810, 5: 0.2315]
    Fantasy/Fairy tales: [1: 0.0401, 2: 0.1202, 3: 0.2300, 4: 0.2404, 5: 0.3694]
    Animated: [1: 0.0475, 2: 0.1246, 3: 0.1929, 4: 0.2448, 5: 0.3902]
    '''

    Prediction_task_zero = '''
    Based on your understanding, prior knowledge, and general reasoning, predict the following for a person’s movie preferences:
    1. How much does this person enjoy Action movies?
    (Action movies: Don't enjoy at all 1-2-3-4-5 Enjoy very much)
    2. How much does this person enjoy Documentary movies?
    (Documentaries: Don't enjoy at all 1-2-3-4-5 Enjoy very much)
    3. How much does this person enjoy Thriller movies?
    (Thriller movies: Don't enjoy at all 1-2-3-4-5 Enjoy very much)
    4. How much does this person enjoy Comedy movies?
    (Comedies: Don't enjoy at all 1-2-3-4-5 Enjoy very much)
    Generate 50 rows of data in each batch and repeat this process for 100 batches to simulate a comprehensive dataset. For each batch, provide the predictions directly in the following format:
    [<Action movie enjoyment>, <Documentary movie enjoyment>, <Thriller movie enjoyment>, <Comedy movie enjoyment>]
    For example, if the person’s enjoyment ratings are 3 for Action movies, 2 for Documentaries, 1 for Thrillers, and 4 for Comedy movies, your response should be:
    [3, 2, 1, 4]
    After completing a batch, explicitly indicate the batch number before moving to the next batch (e.g., "Batch 1 complete"). Continue this process until all 100 batches are generated.
    The format must be a JSON string representing a four-dimensional array. Make sure it is an array of arrays with no objects, like in a spreadsheet.
    After generating, only show the data you generated without additional words.
    '''
    Prediction_task_few = '''
    Based on the provided distributions of Music preferences and Movie preferences, predict the following for each person:
    1. How much does this person enjoy Action movies?
    (Action movies: Don't enjoy at all 1-2-3-4-5 Enjoy very much)
    2. How much does this person enjoy Documentary movies?
    (Documentaries: Don't enjoy at all 1-2-3-4-5 Enjoy very much)
    3. How much does this person enjoy Thriller movies?
    (Thriller movies: Don't enjoy at all 1-2-3-4-5 Enjoy very much)
    4. How much does this person enjoy Comedy movies?
    (Comedies: Don't enjoy at all 1-2-3-4-5 Enjoy very much)
    The distribution information for Music and Movies preferences has been provided. Using these distributions, predict the preferences for Action, Documentary, Thriller, and Comedy movie genres for each individual.
    Generate 20 rows of data in each batch and repeat this process for 100 batches to simulate a comprehensive dataset. For each batch, provide the predictions directly in the following format:
    [<Action movie enjoyment>, <Documentary movie enjoyment>, <Thriller movie enjoyment>, <Comedy movie enjoyment>]
    For example, if the person’s enjoyment ratings are 3 for Action movies, 2 for Documentaries, 1 for Thrillers, and 4 for Comedy movies, your response should be:
    [3, 2, 1, 4]
    After completing a batch, explicitly indicate the batch number before moving to the next batch (e.g., "Batch 1 complete"). Continue this process until all 100 batches are generated.
    The format must be a JSON string representing a four-dimensional array. Make sure it is an array of arrays with no objects, like in a spreadsheet.
    After generating, only show the data you generated without additional words.
    '''
  
    return database_background + role_playing_task_setting_zero +  Prediction_task_zero

def main():
# 应用函数到每一行并生成描述性字符串
    # folder_path = './Data/GSS/'
    # input_path = folder_path + 'selected_GSS.csv'
    output_dir = 'D:\HaolingX\Role-Playing LLM+Survey\code\Full_Simulation\Res_fullSimulation'
    # 确保目录存在
    os.makedirs(output_dir, exist_ok=True)
    # 指定文件名
    output_file = os.path.join(output_dir, 'outputGPT4_zero2.txt')
    responses = []
    gts = []
    with open(output_file, mode='w', newline='') as outfile:
        
        writer = csv.writer(outfile)
        cnt = 0
        #predict
        correct = 0
        prompt = generate_background()
        print(prompt)
        for i in range(20):
    
            # outfile.write(prompt+'\n')
            messages = [{"role": "user", "content": prompt}]
            print(f"No:{i} generating...")
            aa = ask_gpt(client, messages)
            outfile.write(aa+'\n')
            #aa = get_sample_inform(row)
            
            #batchsize 20 times 20
            
        print("done!")
        
if __name__ == "__main__":
    main()