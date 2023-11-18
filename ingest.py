import os
import random

import pinecone
import torch
from tqdm import tqdm
from transformers import (AutoModel, AutoModelForSequenceClassification,
                          AutoTokenizer)

PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
pinecone.init(api_key=PINECONE_API_KEY, environment="gcp-starter")

index = pinecone.Index("recipes")

import json

with open("Epicurious/full_format_recipes.json", "r") as f:
    recipes = json.load(f)

model = AutoModel.from_pretrained(
    'jinaai/jina-embeddings-v2-base-en', 
    trust_remote_code=True, 
    low_cpu_mem_usage=True,
    device_map="auto",
).to("cuda")

# model = model.to_bettertransformer()

BATCH_SIZE = 8

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

print(len(recipe_documents))

# for i in tqdm(range(0, len(recipe_documents), BATCH_SIZE)):
#     documents = recipe_documents[i:i+BATCH_SIZE]
#     with torch.no_grad():
#         embeddings = model.encode([document[0] for document in documents])
#     vectors = [
#         (str(i+j), embedding.tolist(), metadata) for j, (embedding, metadata) in enumerate(zip(embeddings, [document[1] for document in documents]))
#     ]
#     index.upsert(vectors)

results = index.query(
    vector=model.encode('organic milk, orange juice, eggs, lettuce, yogurts').tolist(),
    # filter={
    #     "calories": {"$lte": 100},
    # },
    top_k=3,
    include_metadata=True,
)

print(results)

for result in results["matches"]:
    print(recipe_documents[int(result["id"])][0])