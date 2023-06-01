# chatbot with memory using ConversationSummaryBufferMemory
import config
from langchain import OpenAI
from langchain.chains import ConversationChain
from langchain.chains.conversation.memory import ConversationSummaryBufferMemory

from langchain.callbacks import get_openai_callback

def count_tokens(chain, query):
    with get_openai_callback() as cb:
        result = chain.run(query)
        print(f'Spent a total of {cb.total_tokens} tokens')

    return result

myLlm = OpenAI(
    temperature=0,
    openai_api_key=config.my_openai_api_key,
    model_name='text-davinci-003'
)


conversation_sum_bufw = ConversationChain(
    llm=myLlm, memory=ConversationSummaryBufferMemory(
        llm=myLlm,
        max_token_limit=650
    )
)


# res = count_tokens(
#     conversation_sum_bufw,
#     "Good morning AI!"
# )
# print(res)
# print(conversation_sum_bufw.memory.prompt.template)
# print(conversation_sum_bufw.prompt.template)

print(conversation_sum_bufw.run("good morning AI!"))
while(True):
    user_input = input()
    if user_input == 'quit':
        break
    conversation_sum_bufw.run(user_input)


