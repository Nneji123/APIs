import os
from fastapi import FastAPI
import pandas as pd
import pickle
from difflib import get_close_matches
from download import main

if os.path.exists("similarity.pkl"):
    print("model exists")
else:
    main()

app = FastAPI()


courses_list = pickle.load(open('courses.pkl','rb'))
similarity = pickle.load(open('similarity.pkl','rb'))
df_courses = pd.read_pickle('courses.pkl')
course_list = list(df_courses['course_name'])


#creating a home route
@app.get('/')
async def Home():
    return {'text': 'Welcome!'}


# Creating a welcome message
@app.get('/{name}')
def info(name:str):
    return {'Welcome to the Course Recommender System, {}!'.format(name)}

# a route for course recommendation
@app.post('/recommend')
async def recommend(course:str):
    try:
        course = course.lower()
        correct_course = get_close_matches(
        course, course_list, n=3, cutoff=0.3)[0]
        index = courses_list[courses_list['course_name'] == correct_course].index[0]
        distances = sorted(list(enumerate(similarity[index])), reverse=True, key=lambda x: x[1])
        recommended_course_names = []
        for i in distances[1:7]:
            course_name = courses_list.iloc[i[0]].course_name
            recommended_course_names.append(course_name)
        return {"courses":recommended_course_names}
    except IndexError:
        return {'message': 'No course found!'}

    