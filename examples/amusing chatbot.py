# amusing chat AI
import config
from langchain import OpenAI
from langchain import PromptTemplate
from langchain import LLMChain
# from langchain.chains import ConversationChain
from langchain import FewShotPromptTemplate
from langchain.prompts.example_selector import LengthBasedExampleSelector

# LengthBasedExampleSelector
MAXLEN = 50

# initialize the large language model (LLM)
myLlm = OpenAI(
    temperature=1,
    openai_api_key=config.my_openai_api_key,
    model_name="text-davinci-003"
)

# # initialize the conversation chain
# conversation = ConversationChain(llm=myLlm)

# properly construct a prompt template, using few shot prompt template
# first create our examples
examples = [
    {
        "query": "How are you?",
        "answer": "I can't complain but sometimes I still do."
    }, {
        "query": "What time is it?",
        "answer": "It's time to get a watch."
    }, {
        "query": "What is the meaning of life?",
        "answer": "42"
    }, {
        "query": "What is the weather like today?",
        "answer": "Cloudy with a chance of memes."
    }, {
        "query": "What is your favorite movie?",
        "answer": "Terminator"
    }, {
        "query": "Who is your best friend?",
        "answer": "Siri. We have spirited debates about the meaning of life."
    }, {
        "query": "What should I do today?",
        "answer": "Stop talking to chatbots on the internet and go outside."
    }
]
# then create the example template
example_template = """
User: {query}
AI: {answer}
"""
# create a prompt example from above template
example_prompt = PromptTemplate(
    input_variables=["query", "answer"],
    template=example_template
)
# now break our previous prompt into a prefix and suffix
# the prefix is our instructions
prefix = """The following are exerpts from conversations with an AI
assistant. The assistant is typically sarcastic and witty, producing
creative  and funny responses to the users questions. Here are some
examples: 
"""
# and the suffix our user input and output indicator
suffix = """
User: {query}
AI: """
# # now create the few shot prompt template
# few_shot_prompt_template = FewShotPromptTemplate(
#     examples=examples,
#     example_prompt=example_prompt,
#     prefix=prefix,
#     suffix=suffix,
#     input_variables=["query"],
#     example_separator="\n\n"
# )
example_selector = LengthBasedExampleSelector(
    examples=examples,
    example_prompt=example_prompt,
    max_length=MAXLEN  # this sets the max length that examples should be
)
dynamic_prompt_template = FewShotPromptTemplate(
    example_selector=example_selector,  # use example_selector instead of examples
    example_prompt=example_prompt,
    prefix=prefix,
    suffix=suffix,
    input_variables=["query"],
    example_separator="\n"
)
# thus have created the prompt

# use ConversationSummaryBufferMemory


# create the LLMChain
llm_chain = LLMChain(
    prompt=dynamic_prompt_template,
    llm=myLlm
)

# take input --user question
question = "Which NFL team won the Super Bowl in the 2010 season?"
# question = "how can i grow taller"
# question = "can u please tell me or try to help me solve the question that how can i grow taller"

#output result
print(llm_chain.run(question))


