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

# loading in text data
load_dotenv()

# bedrock client
boto3.setup_default_session(profile_name=os.getenv('profile_name'))
bedrock = boto3.client('bedrock', 'us-east-1', endpoint_url='https://bedrock.us-east-1.amazonaws.com')
opensearch = boto3.client("opensearchserverless")

# OpenSearch Client
host = os.getenv('opensearch_host')  # cluster endpoint, for example: my-test-domain.us-east-1.aoss.amazonaws.com
region = 'us-east-1'
service = 'aoss'
credentials = boto3.Session(profile_name=os.getenv('profile_name')).get_credentials()
auth = AWSV4SignerAuth(credentials, region, service)

client = OpenSearch(
    hosts=[{'host': host, 'port': 443}],
    http_auth=auth,
    use_ssl=True,
    verify_certs=True,
    connection_class=RequestsHttpConnection,
    pool_maxsize=20
)

loader = PyPDFLoader("PDF NAME")
documents = loader.load()

text_splitter = RecursiveCharacterTextSplitter(
    # Play with Chunk Size
    chunk_size=600,
    chunk_overlap=100,
)

doc = text_splitter.split_documents(documents)

avg_doc_length = lambda documents: sum([len(doc.page_content) for doc in documents]) // len(documents)
avg_char_count_pre = avg_doc_length(documents)
avg_char_count_post = avg_doc_length(doc)
print(f'Average length among {len(documents)} documents loaded is {avg_char_count_pre} characters.')
print(f'After the split we have {len(doc)} documents more than the original {len(documents)}.')
print(f'Average length among {len(doc)} documents (after split) is {avg_char_count_post} characters.')

def get_embedding(body):
    """
    This function is used to generate the embeddings for a specific chunk of text
    :param body: This is the example content passed in to generate an embedding
    :return: A vector containing the embeddings of the passed in content
    """
    modelId = 'amazon.titan-e1t-medium'
    accept = 'application/json'
    contentType = 'application/json'
    response = bedrock.invoke_model(body=body, modelId=modelId, accept=accept, contentType=contentType)
    response_body = json.loads(response.get('body').read())
    embedding = response_body.get('embedding')
    return embedding


def indexDoc(client, vectors, text):
    """
    This function indexing the documents and vectors into Amazon OpenSearch Serverless.
    :param client: The instatiation of your OpenSearch Serverless instance.
    :param vectors: The vector you generated with the get_embeddings function that is going to be indexed.
    :param text: The actual text of the document you are storing along with the vector of that text.
    :return: The confirmation that the document was indexed successfully.
    """
    indexDocument = {
        'vectors': vectors,
        'text': text

    }

    response = client.index(
        index='INDEX NAME',
        body=indexDocument,
        # id = '1',
        refresh=False
    )
    return response

# The process of iterating through each chunk of the document you are trying to index, and generate embeddings for.
for i in doc:
    # The text data of each chunk
    exampleContent = i.page_content
    # Generating the embeddings for each chunk of text data
    exampleInput = json.dumps({"inputText": exampleContent})
    exampleVectors = get_embedding(exampleInput)
    # setting the text data as the text variable, and generated vector to a vector variable
    text = exampleContent
    vectors = exampleVectors
    # calling the indexDoc function, passing in the OpenSearch Client, the created vector, and corresponding text data
    indexDoc(client, vectors, text)
