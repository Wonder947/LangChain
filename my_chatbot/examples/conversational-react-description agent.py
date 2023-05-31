#example from https://www.pinecone.io/learn/langchain-agents/
#here we will give an example of how to implement a simple conversational-react-description agent
import config
from langchain.agents import initialize_agent
from langchain import OpenAI
from langchain.agents import Tool

llm = OpenAI(
    model_name="text-davinci-003",
    temperature=0,
    openai_api_key=config.my_openai_api_key
)

from langchain.agents import load_tools

tools = load_tools(
    ['llm-math'],
    llm=llm
)

from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain

prompt = PromptTemplate(
    input_variables=["query"],
    template="{query}"
)

llm_chain = LLMChain(llm=llm, prompt=prompt)

# initialize the LLM tool
llm_tool = Tool(
    name='Language Model',
    func=llm_chain.run,
    description='use this tool for general purpose queries and logic'
)

tools.append(llm_tool)

#first initialize the memory we would like to use, we pick ConversationBufferMemory here
from langchain.memory import ConversationBufferMemory

memory = ConversationBufferMemory(
    memory_key="chat_history"
)

#pass this to memory parameter when initializing the agent
conversational_agent = initialize_agent(
    agent='conversational-react-description',
    tools=tools,
    llm=llm,
    verbose=True,
    max_iterations=3,
    memory=memory,
)

conversational_agent("what is the largest animal")
conversational_agent("how to compute 1.6+2*3.5")
conversational_agent("what is the sum of the length of the largest animal and the number you just computed")
conversational_agent("how about compute the sum with weight instead of length")














