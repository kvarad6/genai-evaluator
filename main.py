
import os
from google.cloud import aiplatform
from dotenv import load_dotenv
from PyPDF2 import PdfReader
from typing_extensions import Concatenate
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import VertexAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains.question_answering import load_qa_chain
from langchain.llms import VertexAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain.evaluation import load_evaluator
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import json
from langchain.chains.summarize import load_summarize_chain


from fastapi import FastAPI, Request, Form
from fastapi.templating import Jinja2Templates


app = FastAPI()
templates = Jinja2Templates(directory="templates")

app.counter = 0
app.unknown = 0
app.dict = {}

load_dotenv()

def getContent(path):
    pdfreader = PdfReader(path)

    rawText = ''
    for i, page in enumerate(pdfreader.pages):
        content = page.extract_text()
        if content:
            rawText += content

    text_splitter = CharacterTextSplitter(        
    separator = "\n",
    chunk_size = 1000,
    #chunk_overlap: as chunk size is 1000, first sentance will have 0-1000 characters, and second sentence will start from 800th character.
    # chunk_overlap  = 200,
    chunk_overlap  = 0,
    length_function = len)
    texts = text_splitter.split_text(rawText)
    return texts



os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "./credentials.json"

aiplatform.init(
    project='',
    location='us-central1',
)

llm = VertexAI()

def cosine_similarity_score(s1, s2):
    vectorizer = CountVectorizer().fit_transform([s1, s2])
    vectors = vectorizer.toarray()
    cosine_sim = cosine_similarity(vectors)
    return cosine_sim[0][1]

def generateQuestion(input_content):

    prompt_template = PromptTemplate.from_template(
        """Generate few question and their answers from the {input} in the format:
        Q: 
        A: """
        # "Generate few questions from the {content}."
    )
    chain = LLMChain(llm=VertexAI(), prompt=prompt_template)
    result = chain.run(input_content)
    lines = result.strip().split('\n')

    questions = []
    answers = []

    # Iterate through the lines and extract questions and answers
    for line in lines:
        # Split each line at the ':' to separate the question/answer marker and the content
        parts = line.split(': ', 1)
        
        # Check if the line is formatted as a question or answer
        if len(parts) == 2:
            marker, content = parts
            if marker == 'Q':
                questions.append(content)
            elif marker == 'A':
                answers.append(content)

    qaList = []
    qaList.append(questions[0])
    qaList.append(answers[0])
    return qaList
@app.get("/")
def read_form(request: Request):
    app.counter = 0
    app.dict = {}
    app.unknown = 0
    return templates.TemplateResponse("index.html", {"request": request})


@app.post("/get_next_questions")
async def getNextQuestions(request: Request, userAnswer: str = Form(...)):
    updatedDict = "ansDict" + str(app.counter)
    initial_answer = app.dict[updatedDict]["Expected Answer"]
    initial_question = app.dict[updatedDict]["Question"]
    app.dict[updatedDict]["User Input"] = userAnswer 
    similarity_score = cosine_similarity_score(userAnswer, initial_answer)
    app.dict[updatedDict]["Score"] = similarity_score         
    
    if(similarity_score<0.4):
        app.counter = app.counter + 1
        updatedDict = "ansDict" + str(app.counter)
        updatedDict = {updatedDict : {
        "Question": initial_question,
        "Expected Answer": initial_answer,
        }}
        app.dict.update(updatedDict)
        app.unknown = app.unknown + 1
        # if(len(app.dict)==4):
        if(app.unknown>3):
            app.dict.popitem()
            jsonResponse = json.dumps(app.dict)
            return templates.TemplateResponse("result.html", {"request": request, "jsonResponse": jsonResponse})
        return templates.TemplateResponse("getUserResponse.html", {"request": request, "userAnswer": userAnswer, "initial_answer": initial_answer, "initial_question": initial_question})
        
    app.counter = app.counter + 1
    updatedDict = "ansDict" + str(app.counter)
    qaList = generateQuestion(userAnswer)
    initial_question = qaList[0]
    initial_answer = qaList[1]
    updatedDict = {updatedDict : {
        "Question": initial_question,
        "Expected Answer": initial_answer,
    }}
    app.dict.update(updatedDict)

    # if(len(app.dict)>3):
    if(app.unknown+1>3 or len(app.dict)>4):
        app.dict.popitem()
        jsonResponse = json.dumps(app.dict)
        return templates.TemplateResponse("result.html", {"request": request, "jsonResponse": jsonResponse})
    
    return templates.TemplateResponse("getFirstQuestion.html", {"request": request, "initial_question": initial_question, "initial_answer": initial_answer})

@app.post("/get_first_question")
async def getQuestions(request: Request, path: str = Form(...)):
    rawText = getContent(path)
    qaList = generateQuestion(rawText)
    initial_question = qaList[0]
    initial_answer = qaList[1]
    app.counter = app.counter + 1
    updatedDict = "ansDict" + str(app.counter)
    updatedDict = {updatedDict : {
        "Question": initial_question,
        "Expected Answer": initial_answer,
    }}
    app.dict.update(updatedDict)
    return templates.TemplateResponse("getFirstQuestion.html", {"request": request, "initial_question": initial_question, "initial_answer": initial_answer, "path": path})




