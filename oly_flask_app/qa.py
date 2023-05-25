import pandas as pd
import ast
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd
import openai
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from flask import Flask, request, render_template

openai.api_key = "YOUR-KEY_HERE"

## fixing formatting issues
embedding_df = pd.read_csv("/Users/sumiran.singh.thakur/Desktop/Olympic Nonsense/brazil_food_clean.csv")
embedding_df["Embedding"] = embedding_df["Embedding"].apply(lambda x: ast.literal_eval(x))
# replace "ofcoffee" with "of coffee"
embedding_df['Sentence'] = embedding_df['Sentence'].str.replace('ofcoffee', 'of coffee')

def distances_from_embeddings(q_embeddings, s_embeddings):
    # Calculate the cosine similarity between the question embedding and the sentence embeddings
    similarities = cosine_similarity(np.array(q_embeddings).reshape(1, -1), s_embeddings)

    # Convert similarities to distances
    distances = 1 - similarities

    return distances[0]

def create_context(question, df, max_len, size="ada"):
    """
    Create a context for a question by finding the most similar context from the dataframe
    """

    # Get the embeddings for the question
    q_embeddings = openai.Embedding.create(input=question, engine='text-embedding-ada-002')['data'][0]['embedding']

    # Get the distances from the embeddings
    df['distances'] = distances_from_embeddings(q_embeddings, df['Embedding'].tolist())

    returns = []
    cur_len = 0

    # Sort by distance and add the text to the context until the context is too long
    for i, row in df.sort_values('distances', ascending=True).iterrows():
        
        # Add the length of the text to the current length
        cur_len += row['Token Length'] + 4
        
        # If the context is too long, break
        if cur_len > max_len:
            break
        
        # Else add it to the text that is being returned
        returns.append(row["Sentence"])

    # Return the context
    return "\n\n###\n\n".join(returns)

def answer_question(
    df,
    model="text-davinci-003",
    question="What is love?",
    max_len=2000,
    size="ada",
    debug=False,
    max_tokens=200,
    stop_sequence=None
):
    """
    Answer a question based on the most similar context from the dataframe texts
    """
    context = create_context(question, df, max_len=max_len, size=size)

    try:
        # Create a completions using the question and context
        prompt = f"Answer the question based on the context below, and if the question can't be answered based on the context, say 'I don't know, my knowledge is limited to the 2016 Olympic games snack information '\n\nContext: {context}\n\n---\n\nQuestion: {question}\nAnswer: "
        response = openai.Completion.create(
            prompt=prompt,
            temperature=0,
            max_tokens=max_tokens,
            top_p=1,
            frequency_penalty=0,
            presence_penalty=0,
            stop=stop_sequence,
            model=model,
        )
        return response["choices"][0]["text"].strip()
    except Exception as e:
        print(e)
        return ""

app = Flask(__name__, static_url_path='/static')

# Initialize list to store questions and answers
convo = []

# Define function to pass enumerate to Jinja2 template
def jinja2_enumerate(iterable, start=0):
    return enumerate(iterable, start=start)

# Add jinja2_enumerate function to Jinja2 globals
app.jinja_env.globals.update(enumerate=jinja2_enumerate)

# Define routes
@app.route("/")
def home():
    return render_template("home.html", convo=convo)

@app.route("/get_answer", methods=["POST"])
def get_answer():
    question = request.form.get("question")
    answer = answer_question(embedding_df, question=question)
    convo.append((question, answer))  # Store question and answer in the list
    return render_template("home.html", convo=convo)

if __name__ == "__main__":
    app.run(debug=True, port=27000)
