# Question-Answering-Project
## Description: 
This is a simple Extractive Question-Answering Chatbot. This involves posing questions about a document and identifying the answers as spans of text in the document itself. You provide it a context and a question, it will response you with an answer.
I finetuned a Bert-base model on Squad Dataset. You can find it here in [my Github:](https://github.com/Huynguyen610/Squad-Dataset)

## End to End Process
  Setup a Github Repository for the project
  Create template structure 
  Project setup & Requirements installation
  Setup my logging, exception and utils Modules
  Experiment with notebooks
  Create a full pipeline including: Data Ingestion, Data Validation, Data Transformation, Model Training and Model Evaluation
  Manage my Pipeline with the help of DVC package(MLOps tool) for Pipeline tracking & implementation
  Create a Prediction pipeline and User App Interface with Chainlit - an open-source Python package that makes it incredibly fast to build Chat GPT like applications
  Dockerize my application
  Deploy my application on AWS with CI/CD using Github Actions


# How to run?
### STEPS:

Clone the repository

```bash
https://github.com/Huynguyen610/Question-Answering-Project
```

### STEP 01- Create a conda environment after opening the repository

```bash
conda create -n summary python=3.8 -y
```

```bash
conda activate summary
```


### STEP 02- install the requirements
```bash
pip install -r requirements.txt
```


```bash
# Finally run the following command
python app.py
```

Now,
```bash
open up you local host and port
```
