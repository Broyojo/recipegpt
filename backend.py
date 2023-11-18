import os

import openai
import pinecone
from fastapi import FastAPI
from fastapi.responses import HTMLResponse
from transformers import (AutoModel, AutoModelForSequenceClassification,
                          AutoTokenizer)

app = FastAPI()

@app.get("/")
async def root():
    with open("static/main.html", "r") as f:
        return HTMLResponse(f.read())

@app.get("/get_recipe_from_image_ingredients/{filepath}")
async def get_recipe_from_image(filepath: str):
    pass
    
@app.get("/get_recipe_from_text_ingredients/{text}")
async def get_recipe_from_text(text: str):
    pass

@app.get("/get_recipe_from_food_image/{filepath}")
async def get_recipe_from_food_image(filepath: str):
    pass