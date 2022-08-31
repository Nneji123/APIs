
import io
import os
import sys
import uvicorn
import pandas as pd
from ast import literal_eval
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import PlainTextResponse
from pydantic import BaseModel
from functions import get_recommendation

sys.path.append(os.path.abspath(os.path.join("..", "config")))

app = FastAPI(
    title="Course Recommender API",
    description="""An API for getting course recommendations.""",
)

origins = ["*"]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

with open("joblists.txt") as file:
    lines = file.readlines()
    jobs = [line.rstrip() for line in lines]

DB = pd.read_csv("JDs_final.csv").dropna()
DB.details = DB.details.apply(lambda x: literal_eval(x))
DB.tokenized = DB.tokenized.apply(lambda x: literal_eval(x))
DB = DB.drop_duplicates(["description"])
datas= pd.read_csv("processed_courses_data.csv")


@app.get("/", response_class=PlainTextResponse, tags=["home"])
async def home():
    note = """
    Course Recommender API
    Note: add "/redoc" to get the complete documentation.
    """
    return note


class Course(BaseModel):
    job: str


# endpoint for just enhancing the image
@app.post("/movie", response_class=PlainTextResponse)
async def get_course(data: Course):
    text = get_recommendation(DB=DB, data=datas, jobname=data.job)
    return text

if __name__=="__main__":
    uvicorn.run(app, port=8000) 

