#main project
import config
import myTools
import string
from langchain import OpenAI
from langchain.chat_models import ChatOpenAI
from langchain.agents import initialize_agent
from langchain.chains.conversation.memory import ConversationBufferWindowMemory
from langchain.chains.conversation.memory import ConversationSummaryBufferMemory
from langchain.chains import ConversationChain



llm_davinci = OpenAI(
    model_name="text-davinci-003",
    temperature=0,
    openai_api_key=config.my_openai_api_key
)

llm_chat = ChatOpenAI(
    model_name='gpt-3.5-turbo',
    temperature=0,
    openai_api_key=config.my_openai_api_key
)
my_tools=myTools.my_tools
# my_tools = [myTools.CircumferenceTool()]
my_memory = ConversationBufferWindowMemory(
        memory_key='chat_history',
        k=5,
        return_messages=True
)
# my_memory = ConversationSummaryBufferMemory(
#     llm=llm_davinci,
#     max_token_limit=650,
# )

#initialize the agent
# my_agent = initialize_agent(
#     agent='chat-conversational-react-description',
#     tools=my_tools,
#     llm=llm_chat,
#     verbose=True,
#     max_iterations=3,
#     early_stopping_method='generate',
#     memory=my_memory
# )


conversational_agent = initialize_agent(
    agent='conversational-react-description',
    tools=my_tools,
    llm=llm_davinci,
    verbose=True,
    max_iterations=3,
    memory=my_memory,
)
my_agent = conversational_agent

# #rewrite the prompt
# sys_msg = """Assistant is a large language model trained by OpenAI.
#
# Assistant is designed to be able to assist with a wide range of tasks, from answering simple questions to providing in-depth explanations and discussions on a wide range of topics. As a language model, Assistant is able to generate human-like text based on the input it receives, allowing it to engage in natural-sounding conversations and provide responses that are coherent and relevant to the topic at hand.
#
# Assistant is constantly learning and improving, and its capabilities are constantly evolving. It is able to process and understand large amounts of text, and can use this knowledge to provide accurate and informative responses to a wide range of questions. Additionally, Assistant is able to generate its own text based on the input it receives, allowing it to engage in discussions and provide explanations and descriptions on a wide range of topics.
#
# Unfortunately, Assistant is terrible at maths. When provided with math questions, no matter how simple, assistant always refers to it's trusty tools and absolutely does NOT try to answer math questions by itself
#
# Overall, Assistant is a powerful system that can help with a wide range of tasks and provide valuable insights and information on a wide range of topics. Whether you need help with a specific question or just want to have a conversation about a particular topic, Assistant is here to assist.
# """
# new_prompt = my_agent.agent.create_prompt(
#     system_message=sys_msg,
#     tools=my_tools
# )
# my_agent.agent.llm_chain.prompt = new_prompt



my_agent("can you calculate 1+3.5*2 ?")
my_agent("what is the largest animal")
my_agent("what is the sum of the length of the largest animal and the number you just computed")

while(True):
    user_input = input()
    if user_input == 'quit':
        break
    my_agent(user_input)





