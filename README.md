# Question-Answering-Project
## Description: 
This is a simple Extractive Question-Answering Chatbot. This involves posing questions about a document and identifying the answers as spans of text in the document itself. You provide it a context and a question, it will response you with an answer.  
  
I finetuned a Bert-base model on Squad Dataset. You can find the dataset here in [my Github:](https://github.com/Huynguyen610/Squad-Dataset)

And this is my sample bot that I've deployed on Streamlit: [My sample Bot](https://extractive-app-bot-fxajpxvw6lqqjeycie4xrq.streamlit.app/)

<br></br>

## End to End Process
<br></br>
1. Setup a Github Repository for the project  
    
2. Create template structure   
    
3. Project setup & Requirements installation  
    
4. Setup my logging, exception and utils Modules  
    
5. Experiment with notebooks  
    
6. Create a full pipeline including: Data Ingestion, Data Validation, Data Transformation, Model Training and Model Evaluation  
   
7. Manage my Pipeline with the help of DVC package(MLOps tool) for Pipeline tracking & implementation  
   
8. Create a Prediction pipeline and User App Interface with Streamlit - an open-source Python package that makes it incredibly fast to build Chat GPT like applications  
    
9. Deploy it on [Streamlit](https://share.streamlit.io/)
<br></br>


(With limited resources, but for deployment illustrative purpose I upload my model on [huggingface hub](https://huggingface.co/huynguyen61098/Bert-Base-Cased-Squad-Extractive-QA), and call the model directly from it)
<br></br>

# How to run the app in your local?
<br></br>

### STEPS 01:

Clone the repository using git cmd

```bash
git clone https://github.com/Huynguyen610/Question-Answering-Project.git
```
<br></br>

### STEP 02- Create a conda environment after opening the repository

```bash
conda create -n qa_project python=3.9 -y
```

```bash
conda activate qa_project
```
<br></br>


### STEP 03- install the requirements
```bash
pip install -r requirements.txt
```
<br></br>


### STEP 04 - initial your dvc in terminal:
```bash
dvc init
```
<br></br>


### STEP 04 - reproduce your artifacts
Now, run this command to automate the pipeline (data ingestion, data validation, data transformation, model training and evaluation). It'll execute the dvc.yaml file:
```bash
dvc repro
```
<br></br>


### STEP 05 - Finally run the following command to test it on your local host
```bash
streamlit run app.py
```


