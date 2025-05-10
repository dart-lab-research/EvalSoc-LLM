>📋  A template README.md for code accompanying a Machine Learning paper

# Large Language Models as Virtual Survey Respondents: Evaluating Sociodemographic Response Generation

This repository is the official implementation of [My Paper Title](https://openreview.net/forum?id=1bvK5RG0MS). 

>📋  Optional: include a graphic explaining your approach/main result, bibtex entry, link to demos, blog posts and tutorials

An overview of our benchmark workflow is illustrated in the following figure, showcasing Partial Attribute Simulation (left) and Full Attribute Simulation (right). 
![Benchmark Workflow](https://github.com/dart-lab-research/EvalSoc-LLM/blob/main/figs/workflow.png)

## Requirements

### Install Ollama
To install the Ollama platform, execute the following command:

```bash
curl -fsSL https://ollama.com/install.sh | sh
```

After installation, you can run different versions of Llama models using the following commands:

```bash
# For Llama 3.0 8B
ollama run llama3

# For Llama 3.1 8B
ollama run llama3.1:8b
```

### Install Dependencies
Create and activate a new conda environment with all required dependencies:

```bash
# Create environment from environment.yml
conda env create -f environment.yml -n myenv

# Activate the environment
conda activate myenv
```

### Configure OpenAI API Key
Create a configuration file for your OpenAI API key:

```bash
echo "YOUR_API_KEY_HERE" > config/api_key.txt
```

### Dataset Preparation

Our dataset is publicly available on Hugging Face. You can access it through the following link: [SurveyEval-LLM](https://huggingface.co/datasets/BrainCompiler/SurveyEval-LLM)

## Evaluation

### PAS
To evaluate my model on ImageNet, run:

For example, EMP...

```eval
python ./evvaluation_pas/EMP/rpla.py --llm [llm name]
```

llm name: gpt-3.5-turbo, gpt-4-turbo, llama3:8b, llama3.1:8b


### FAS

1. Simulation and Sample Generating

we get ...(input file and output file)
2. Manually filter dirty samples referring to section n in paper (several failure cases described in section n):
For example :
fig 
Due to .... different cases (unrelated info): script x
, we manually...

we get ...(input file and output file)

3. Random Sampling and Metrics Computing
```eval
python ... --llm [llm name]
```
we get ...(input file and output file)

>📋  Describe how to evaluate the trained models on benchmarks reported in the paper, give commands that produce the results (section below).


## Results

Our model achieves the following performance on :

### [Image Classification on ImageNet](https://paperswithcode.com/sota/image-classification-on-imagenet)

| Model name         | Top 1 Accuracy  | Top 5 Accuracy |
| ------------------ |---------------- | -------------- |
| My awesome model   |     85%         |      95%       |

>📋  Include a table of results from your paper, and link back to the leaderboard for clarity and context. If your main result is a figure, include that figure and link to the command or notebook to reproduce it. 


## Contributing

>📋  Pick a licence and describe how to contribute to your code repository. 
