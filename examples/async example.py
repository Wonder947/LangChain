#from https://python.langchain.com/en/latest/modules/models/llms/examples/async_llm.html
import config
import time
import asyncio

from langchain.llms import OpenAI

def generate_serially():
    llm = OpenAI(temperature=0.9, openai_api_key=config.my_openai_api_key)
    for _ in range(10):
        resp = llm.generate(["Hello, how are you?"])
        print(resp.generations[0][0].text)


async def async_generate(llm):
    resp = await llm.agenerate(["Hello, how are you?"])
    print(resp.generations[0][0].text)


async def generate_concurrently():
    llm = OpenAI(temperature=0.9, openai_api_key=config.my_openai_api_key)
    tasks = [async_generate(llm) for _ in range(10)]
    await asyncio.gather(*tasks)


s = time.perf_counter()
# # If running this outside of Jupyter, use asyncio.run(generate_concurrently())
# await generate_concurrently()
asyncio.run((generate_concurrently()))
elapsed = time.perf_counter() - s
print('\033[1m' + f"Concurrent executed in {elapsed:0.2f} seconds." + '\033[0m')

s = time.perf_counter()
generate_serially()
elapsed = time.perf_counter() - s
print('\033[1m' + f"Serial executed in {elapsed:0.2f} seconds." + '\033[0m')




