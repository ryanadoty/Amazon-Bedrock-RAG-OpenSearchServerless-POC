import boto3
import json
import botocore
from dotenv import load_dotenv
import os
import sys
from opensearchpy import OpenSearch
from opensearchpy import OpenSearch, RequestsHttpConnection, AWSV4SignerAuth
import time
from langchain.embeddings import BedrockEmbeddings
from langchain.llms.bedrock import Bedrock
import numpy as np
from langchain.text_splitter import CharacterTextSplitter, RecursiveCharacterTextSplitter
from langchain.document_loaders import PyPDFLoader, PyPDFDirectoryLoader
from IPython.display import display_markdown, Markdown, clear_output
load_dotenv()

boto3.setup_default_session(profile_name='bedrock')
bedrock = boto3.client('bedrock', 'us-east-1', endpoint_url='https://bedrock.us-east-1.amazonaws.com')
opensearch = boto3.client("opensearchserverless")

# OpenSearch Client
host = os.getenv('opensearch_host')  # cluster endpoint, for example: my-test-domain.us-east-1.aoss.amazonaws.com
region = 'us-east-1'
service = 'aoss'
credentials = boto3.Session(profile_name='bedrock').get_credentials()
auth = AWSV4SignerAuth(credentials, region, service)

client = OpenSearch(
    hosts=[{'host': host, 'port': 443}],
    http_auth=auth,
    use_ssl=True,
    verify_certs=True,
    connection_class=RequestsHttpConnection,
    pool_maxsize=20
)

def get_embedding(body):
    modelId = 'amazon.titan-e1t-medium'
    accept = 'application/json'
    contentType = 'application/json'
    response = bedrock.invoke_model(body=body, modelId=modelId, accept=accept, contentType=contentType)
    response_body = json.loads(response.get('body').read())
    embedding = response_body.get('embedding')
    return embedding

def answer_query(user_input):
    userQuery = user_input
    userQueryBody = json.dumps({"inputText": userQuery})
    userVectors = get_embedding(userQueryBody)

    query = {
        "size": 3,
        "query": {
            "knn": {
                "vectors": {
                    "vector": userVectors, "k": 3
                }
            }
        },
        "_source": True,
        "fields": ["text"],
    }

    response = client.search(
        body=query,
        index='hp-final'
    )

    # Format Json responses into text

    similaritysearchResponse = ""

    for i in response["hits"]["hits"]:
        outputtext = i["fields"]["text"]
        similaritysearchResponse = similaritysearchResponse + "Info = " + str(outputtext)

        similaritysearchResponse = similaritysearchResponse

    prompt_data = f"""
    Human: You are an AI assistant that will help people answer questions they have about Harry Potter. Answer the provided question to the best of your ability using the information provided in the Context. 
    Summarize the answer and provide sources to where the relevant information can be found. 
    Include this at the end of the response.
    Provide information based on the context provided.
    Format the output in human readable format - use paragraphs and bullet lists when applicable
    Answer in detail with no preamble
    If you are unable to answer accurately, please say so.
    Please mention the books and page numbers of those books where any excerpts are taken from.

    Question: {userQuery}

    Here is the text you should use as context: {similaritysearchResponse}

    Assistant:

    """

    body = json.dumps({"prompt": prompt_data,
                       "max_tokens_to_sample": 4096,
                       "temperature": 0.5,
                       "top_k": 250,
                       "top_p": 0.5,
                       "stop_sequences": []
                       })

    # Run infernce on the LLM

    modelId = "anthropic.claude-v2"  # change this to use a different version from the model provider
    accept = "application/json"
    contentType = "application/json"

    response = bedrock.invoke_model_with_response_stream(body=body, modelId=modelId, accept=accept,
                                                         contentType=contentType)
    stream = response.get('body')
    output = []
    i = 1
    answer = ''
    if stream:
        for event in stream:
            chunk = event.get('chunk')
            if chunk:
                chunk_obj = json.loads(chunk.get('bytes').decode())
                text = chunk_obj['completion']
                clear_output(wait=True)
                output.append(text)
                answer = ''.join(output)
                i += 1
        return (answer)