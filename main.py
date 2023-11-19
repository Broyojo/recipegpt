import base64
import json
import os
from typing import List

import openai
import pinecone
from fastapi import FastAPI, File, Form, Query, UploadFile
from fastapi.responses import HTMLResponse
from openai import OpenAI
from tqdm import tqdm
from transformers import (AutoModel, AutoModelForSequenceClassification,
                          AutoTokenizer)

app = FastAPI()
client = OpenAI(api_key="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")

PINECONE_API_KEY = "35abfaa7-5beb-403b-a878-bbcc51742cc4"
pinecone.init(api_key=PINECONE_API_KEY, environment="gcp-starter")

index = pinecone.Index("recipes")

with open("Epicurious/full_format_recipes.json", "r") as f:
    recipes = json.load(f)

def make_document(recipe):
    s = ""
    
    if "title" in recipe and recipe["title"] is not None:
        s += "Title: " + recipe["title"] + "\n"
    
    if "desc" in recipe and recipe["desc"] is not None:
        s += "Description: " + recipe["desc"] + "\n"
    
    if "ingredients" in recipe and recipe["ingredients"] is not None:
        s += "Ingredients:\n"
        for ingredient in recipe["ingredients"]:
            s += "- " + ingredient + "\n"
    
    if "directions" in recipe and recipe["directions"] is not None:
        s += "Directions:\n" + "\n".join(recipe["directions"]) + "\n"
    
    metadata = {}
    
    if "fat" in recipe and recipe["fat"] is not None:
        metadata["fat"] = recipe["fat"]
    
    if "calories" in recipe and recipe["calories"] is not None:
        metadata["calories"] = recipe["calories"]
    
    if "protein" in recipe and recipe["protein"] is not None:
        metadata["protein"] = recipe["protein"]
    
    if "rating" in recipe and recipe["rating"] is not None:
        metadata["rating"] = recipe["rating"]
        
    if "sodium" in recipe and recipe["sodium"] is not None:
        metadata["sodium"] = recipe["sodium"]
    
    return s[:-1], metadata

def dedup(recipe_documents):
    deduped_recipe_documents = []
    for recipe_document in recipe_documents:
        if recipe_document not in deduped_recipe_documents:
            deduped_recipe_documents.append(recipe_document)
    return deduped_recipe_documents

recipe_documents = dedup([make_document(recipe) for recipe in recipes])

@app.get("/")
async def home():
    with open("static/home.html", "r") as f:
        return HTMLResponse(f.read())

@app.get("/home.html")
async def home():
    with open("static/home.html", "r") as f:
        return HTMLResponse(f.read())

@app.get("/main.html")
async def main():
    with open("static/main.html", "r") as f:
        return HTMLResponse(f.read())

@app.get("/page.html")
async def page():
    with open("static/page.html", "r") as f:
        return HTMLResponse(f.read())

with open("Epicurious/full_format_recipes.json", "r") as f:
    recipes = json.load(f)

model = AutoModel.from_pretrained(
    'jinaai/jina-embeddings-v2-base-en', 
    trust_remote_code=True, 
    low_cpu_mem_usage=True,
    device_map="auto",
).to("cpu")

@app.post("/get_recipe")
async def get_recipe(
    photos: List[UploadFile] = Form(...),
    target_calorie_range: str = Form(...),
    # dietary_restrictions: str = Form(...),
    # meal_type: str = Form(...),
    # manual_ingredients: str = Form(...)
):
    encoded_files = []
    for file in photos:
        content = await file.read()
        encoded_files.append(base64.b64encode(content).decode("utf-8"))
    
    ingredients_over_rounds = []
    
    for _ in tqdm(range(5)):
        ingredients = client.chat.completions.create(
            model="gpt-4-vision-preview",
            messages=[
                {
                    "role": "system",
                    "content": "You are a helpful ingredient identifier."
                },
                {
                    "role": "user",
                    "content": [
                        "Output bulleted list of edible ingredients in the image along with their quantities.",
                        *[{"image": file} for file in encoded_files],
                    ],
                }
            ],
            temperature=0.7,
            max_tokens=1024,
        ).choices[0].message.content
        
        ingredients_over_rounds.append(ingredients)

    ingredient_prompt = "You are a helpful ingredient identifier. Here are the ingredients found by several experts:"
    
    for i, ingredients in enumerate(ingredients_over_rounds):
        ingredient_prompt += f"\n{i+1}. {ingredients}\n\n"

    ingredient_prompt += "Please make a refined list of ingredients which is composed of the ingredients which are agreed upon by the experts:"

    ingredients = client.chat.completions.create(
        model="gpt-4-vision-preview",
        messages=[
            {
                "role": "system",
                "content": ingredient_prompt
            },
        ],
        max_tokens=1024,
    ).choices[0].message.content
    print(target_calorie_range)
    
    calorie_range = target_calorie_range.split("-")

    results = index.query(
        vector=model.encode(ingredients).tolist(),
        filter={
            "$and": [{"calories": {"$gte": int(calorie_range[0])}}, {"calories": {"$lte": int(calorie_range[1])}}],
        },
        top_k=10,
        include_metadata=True,
    )

    print(results)
    
    matched_recipes = []
    
    for result in results.matches:
        matched_recipes.append(recipe_documents[int(result["id"])][0])
        
    prompt = f"You are a helpful recipe creator/suggester. Any recipe you suggest or create must be possible to make given all the ingredients available. The recipe must have a calorie count within the range \n{target_calorie_range}\n\n. Here are the ingredients available:\n{ingredients}\n\n. Here are a few potentially relevant recipes that relate to the available ingredients:\n"
    
    for matched_recipe in matched_recipes:
        prompt += f"{matched_recipe}\n\n"
    
    prompt += "Your job is to synthesize a new recipe that is possible to make given the available ingredients and taking some inspiration from the potentially relevant recipes. Ensure that the calorie count of the recipe is within the provided range. Please make your recipe detailed and easy to follow:"
    
    print(prompt)
    
    print("#"*100)
    
    created_recipe = client.chat.completions.create(
        model="gpt-4-vision-preview",
        messages=[
            {
                "role": "system",
                "content": prompt
            }
        ],
        max_tokens=1024,
    ).choices[0].message.content
    
    print(created_recipe)
    return {"recipe": created_recipe}