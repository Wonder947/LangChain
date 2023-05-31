# this file defines different tools that could be used by the chatbot
import config
from langchain import OpenAI
from langchain.agents import Tool
from langchain.tools import BaseTool
from langchain.chains import LLMMathChain
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain.agents import initialize_agent

from math import pi
from typing import Union

text_llm = OpenAI(
    model_name="text-davinci-003",
    temperature=0,
    openai_api_key=config.my_openai_api_key
)

my_tools=[]

#example 1:
llm_math = LLMMathChain(llm=text_llm)
math_tool = Tool(
    name='Calculator',
    func=llm_math.run,
    description='Useful for when you need to answer questions about math.'
)
my_tools.append(math_tool)

#example 2:
prompt_ex2 = PromptTemplate(
    input_variables=["query"],
    template="{query}"
)
llm_chain = LLMChain(
    llm=text_llm,
    prompt=prompt_ex2
)
llm_tool = Tool(
    name='Language Model',
    func=llm_chain.run,
    description='use this tool for general purpose queries and logic'
)
my_tools.append(llm_tool)

#example 3:
class CircumferenceTool(BaseTool):
    name = "Circumference calculator"
    description = "use this tool when you need to calculate a circumference using the radius of a circle"

def _run(self, radius: Union[int, float]):
    return float(radius) * 2.0 * pi

def _arun(self, radius: int):
    raise NotImplementedError("This tool does not support async")

# s = CircumferenceTool()
# print(s)
# my_tools.append(CircumferenceTool)

# example 4:
from langchain import Wikipedia
from langchain.agents.react.base import DocstoreExplorer

docstore=DocstoreExplorer(Wikipedia())
tools = [
    Tool(
        name="Search",
        func=docstore.search,
        description='search wikipedia'
    ),
    Tool(
        name="Lookup",
        func=docstore.lookup,
        description='lookup a term in wikipedia'
    )
]
docstore_agent = initialize_agent(
    tools,
    text_llm,
    agent="react-docstore",
    verbose=True,
    max_iterations=3
)

docstore_tool = Tool(
    name="WikiPedia",
    func=docstore_agent,
    description="use this tool when you need to inquire WikiPedia"
)
# my_tools.append(docstore_tool)


# #example 5:
# from datasets import load_dataset
# data = load_dataset("wikipedia", "20220301.simple", split='train[:10000]')
# import tiktoken
# tokenizer = tiktoken.get_encoding('p50k_base')
# # create the length function
# def tiktoken_len(text):
#     tokens = tokenizer.encode(
#         text,
#         disallowed_special=()
#     )
#     return len(tokens)
# from langchain.text_splitter import RecursiveCharacterTextSplitter
# text_splitter = RecursiveCharacterTextSplitter(
#     chunk_size=400,
#     chunk_overlap=20,
#     length_function=tiktoken_len,
#     separators=["\n\n", "\n", " ", ""]
# )
# from langchain.embeddings.openai import OpenAIEmbeddings
# model_name = 'text-embedding-ada-002'
# embed = OpenAIEmbeddings(
#     # document_model_name=model_name,
#     # query_model_name=model_name,
#     openai_api_key=config.my_openai_api_key
# )
# texts = [
#     'this is the first chunk of text',
#     'then another second chunk of text is here'
# ]
# res = embed.embed_documents(texts)
# import pinecone
# index_name = 'langchain-retrieval-augmentation'
# pinecone.init(
#         api_key=config.my_pinecone_api_key,  # find api key in console at app.pinecone.io
#         environment=config.my_pinecone_environment  # find next to api key in console
# )
# # we create a new index
# pinecone.create_index(
#         name=index_name,
#         metric='dotproduct',
#         dimension=len(res[0]) # 1536 dim of text-embedding-ada-002
# )
# index = pinecone.GRPCIndex(index_name)
# from tqdm.auto import tqdm
# from uuid import uuid4
# batch_limit = 100
# texts = []
# metadatas = []
# for i, record in enumerate(tqdm(data)):
#     # first get metadata fields for this record
#     metadata = {
#         'wiki-id': str(record['id']),
#         'source': record['url'],
#         'title': record['title']
#     }
#     # now we create chunks from the record text
#     record_texts = text_splitter.split_text(record['text'])
#     # create individual metadata dicts for each chunk
#     record_metadatas = [{
#         "chunk": j, "text": text, **metadata
#     } for j, text in enumerate(record_texts)]
#     # append these to current batches
#     texts.extend(record_texts)
#     metadatas.extend(record_metadatas)
#     # if we have reached the batch_limit we can add texts
#     if len(texts) >= batch_limit:
#         ids = [str(uuid4()) for _ in range(len(texts))]
#         embeds = embed.embed_documents(texts)
#         index.upsert(vectors=zip(ids, embeds, metadatas))
#         texts = []
#         metadatas = []
# print(index.describe_index_stats())
# from langchain.vectorstores import Pinecone
#
# text_field = "text"
#
# # switch back to normal index for langchain
# index = pinecone.Index(index_name)
#
# vectorstore = Pinecone(
#     index, embed.embed_query, text_field
# )
# from langchain.chat_models import ChatOpenAI
# from langchain.chains import RetrievalQA
#
# # completion llm
# llm = ChatOpenAI(
#     openai_api_key=config.my_openai_api_key,
#     model_name='gpt-3.5-turbo',
#     temperature=0.0
# )
#
# qa = RetrievalQA.from_chain_type(
#     llm=llm,
#     chain_type="stuff",
#     retriever=vectorstore.as_retriever()
# )
#
# database_tool = Tool(
#     name="GQA Model",
#     func=qa.run,
#     description="use this tool when you need to answer questions with specific knowledge"
# )
# # my_tools.append(database_tool)












