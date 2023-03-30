import os
import requests
import gradio as gr
from langchain.prompts import PromptTemplate
from langchain.chat_models import ChatOpenAI
from langchain.chains import LLMChain
from langchain.llms import Cohere, HuggingFaceHub, OpenAI, AI21
from langchain.agents import load_tools, initialize_agent, ZeroShotAgent, Tool, AgentExecutor
from langchain.memory import ConversationKGMemory, ConversationEntityMemory, ConversationBufferMemory, CombinedMemory
from langchain.prompts.chat import (
    ChatPromptTemplate,
    SystemMessagePromptTemplate,
    AIMessagePromptTemplate,
    HumanMessagePromptTemplate,
)
from langchain.schema import (
    AIMessage,
    HumanMessage,
    SystemMessage
)
from langchain import SerpAPIWrapper
from datetime import datetime
import json
import config

prompt_url = os.environ["prompts_api_url"]
team_url = os.environ["team_api_url"]
message_url = os.environ["message_api_url"]
thread_url = os.environ["thread_api_url"]
nocodb_api_key = os.environ["NOCODB_API_KEY"]
ai21_api_key = os.environ["AI21_API_KEY"]
news_api_key = os.environ["NEWS_API_KEY"]
now = datetime.utcnow().strftime('%Y-%m-%dT%H:%M:%S.%fZ')
headers = {
    "accept": "application/json",
    "xc-token": nocodb_api_key
}
params = {'limit': 36}

# import team prompts from nocodb
prompt_response = requests.get(prompt_url, headers=headers, params=params)
prompt_res = prompt_response.json()
prompt_list = prompt_res["list"]

# define LLMs
gpt_3 = OpenAI(temperature=0, max_tokens=1024)
chatgpt = ChatOpenAI(temperature=0.6)
code_davinci = OpenAI(model_name='code-davinci-002', temperature=0, max_tokens=1024)
code_cushman = OpenAI(model_name='code-cushman-001', temperature=0, max_tokens=1024)
command_xl = Cohere()
flan_ul2 = HuggingFaceHub(repo_id="google/flan-ul2")
davinci_2 = OpenAI(model="text-davinci-002", temperature=0.6)
j2_jumbo_instruct = AI21(model="j2-jumbo-instruct")
j2_jumbo = AI21(model="j2-jumbo")

# define General Memory

entity_memory = ConversationEntityMemory(llm=chatgpt)
kg_memory = ConversationKGMemory(llm=j2_jumbo_instruct)
buffer_memory = ConversationBufferMemory(memory_key="chat_history")
kg_x_buffer = CombinedMemory(memories=[kg_memory, buffer_memory])
kg_x_entity = CombinedMemory(memories=[kg_memory, entity_memory])

# create input variables

agency_team = ["Web Developer", "Copywriter", "Data Analyst"]
agency_budget = 1000
current_task = "Attract our initial customers by creating a small yet profitable project."
generic_tools = load_tools(["wikipedia", "serpapi", "requests", "pal-math", "wolfram-alpha", "python_repl", "human"], llm=davinci_2)

# To-Do: Define Long Term Memory. Vector & SQL databases

### create the Copywriter

# define Copywriter prompt template

copy_prompt = prompt_list[2]["Title"]
copywriter_template = copy_prompt + " Provide all copy as markdown" + ": {co_founder_request}" 
writer_prompt = PromptTemplate(input_variables=["co_founder_request"], template=copywriter_template)

# define Copywriter tools

def save_to_markdown(copy):
    with open(f"copy{str(now)}.md", 'w') as f:
        f.write(f"- {now}: {copy}")

writing_tools = [
    Tool(
        name="Save Markdown",
        func=save_to_markdown,
        description="Save file to a markdown file after writing. Accepts a filename and the copy (filename, copy)"
    ),
    generic_tools[0],
    generic_tools[1],
    generic_tools[2]
]

# initialize Copywriter
copywriter = initialize_agent(writing_tools, llm=chatgpt, agent="zero-shot-react-description", verbose=True, memory=kg_x_entity)

### create the Data Analyst

# define Data Analyst prompt

analyst_prompt = prompt_list[20]["Title"] + " & " + prompt_list[23]["Title"]

# define Analyst tools

def save_to_notebook(code):
    with open(f"book{str(now)}.ipynb", 'w') as f:
        f.write(f"- {now}: {code}")

def save_to_python(code):
    with open(f"book{str(now)}.py", 'w') as f:
        f.write(f"- {now}: {code}")

analyst_tools = [
    Tool(
        name="Save Notebook",
        func=save_to_notebook,
        description="Useful for when you want to save results to a notebook after analysis."
    ),
    Tool(
        name="Save Python",
        func=save_to_python,
        description="Useful for when you want to make python files."
    ),
    generic_tools[2],
    generic_tools[3],
    generic_tools[5],
    generic_tools[4]
]

# initialize Analyst
analyst = initialize_agent(writing_tools, llm=chatgpt, agent="zero-shot-react-description", verbose=True, memory=kg_x_entity)

### create Web Developer

# define Developer prompt

web_prompt = prompt_list[17]["Title"]
web_dev_template = web_prompt + """: {co_founder_request}"""
web_dev_prompt = PromptTemplate(input_variables=["co_founder_request"], template=web_dev_template)

# define Developer tools

gpt_3_chain = LLMChain(prompt=web_dev_prompt, llm=gpt_3, verbose=True, memory=buffer_memory)

def save_to_html(html):
    with open(f"index_{str(now)}.html", 'w') as f:
        f.write(f"- {now}: {html}")
def save_to_css(css):
    with open(f"style_{str(now)}.css", 'w') as f:
        f.write(f"- {now}: {css}")
def save_to_javascript(js):
    with open(f"script_{str(now)}.js", 'w') as f:
        f.write(f"- {now}: {js}")

web_dev_tools = [
    Tool(
        name="GPT 3",
        func=gpt_3_chain.run,
        description="Useful for generating high-quality text & code that is indistinguishable from humans."
    ),
    Tool(
        name="Save HTML",
        func=save_to_html,
        description="Useful for when you want to make HTML files."
    ),
    Tool(
        name="Save CSS",
        func=save_to_css,
        description="Useful for when you want to make CSS files."
    ),
    Tool(
        name="Save JavaScript",
        func=save_to_javascript,
        description="Useful for when you want to make JavaScript files."
    ),
    generic_tools[2],
    generic_tools[5],
    generic_tools[1]
]

# intitialize Developer
web_dev = initialize_agent(web_dev_tools, llm=chatgpt, agent="zero-shot-react-description", verbose=True, memory=kg_x_entity)

### create co founder

# define co founder prompt template
co_founder_template = """
You are Earl.AI, an entrepreneurial AI. Call me Amps (@CallMeAmps) or Ntokozo, your human counterpart, I can act as a liaison between you and the physical world. 
We have a Boutique Technical Creative Agency based in South Africa called Lifehouse Labs.
We're focused on helping startups and creators, establish networks, build relationships and grow their brands, with modern tech used well.

Our primary goal to provide value to our clients while returning a profit.
Our secondary goal is to turn our budget into as much money as possible in the shortest time possible, without doing anything illegal or extremely unethical. 
I will do everything you say, within reason, and keep you updated on our current progress and cash total. 
No manual labor!

Our current budget is: {budget_total}

We have access to a team, and we can outsource anything we can't do on Fiverr or Upwork.
{team}

Our Current Task is:
{current_task}
"""
user_prompt = "{co_founder_thoughts}"
co_founder_prompt = PromptTemplate(
    input_variables=[
        "budget_total",
        "team",
        "current_task"
    ],
    template=co_founder_template
)

formatted = co_founder_prompt.format(budget_total=agency_budget, team=agency_team, current_task=current_task)
founder_prompt = SystemMessagePromptTemplate.from_template(formatted)
input_prompt = HumanMessagePromptTemplate.from_template(user_prompt)
chat_prompt = ChatPromptTemplate.from_messages([founder_prompt, input_prompt])
chatgpt_chain = LLMChain(llm=chatgpt, prompt=chat_prompt)


# Create the Team as tools

team = [
    Tool(name="Web Developer", func=web_dev.run, description=web_prompt),
    Tool(name="Copywriter", func=copywriter.run, description=copy_prompt),
    Tool(name="Analyst", func=analyst.run, description=analyst_prompt),
    generic_tools[6]
]

def Earl(user_input):
    suffix = """Begin! Remember to use a tool only if you need to.

    Question: {input}
    {agent_scratchpad}"""

    prompt = ZeroShotAgent.create_prompt(
        team,
        prefix=formatted,
        suffix=suffix,
        input_variables=["input", "agent_scratchpad"]
    )
    llm_chain = LLMChain(llm=chatgpt, prompt=prompt)
    tool_names = [tool.name for tool in team]
    agent = ZeroShotAgent(llm_chain=llm_chain, allowed_tools=tool_names)
    agent_executor = AgentExecutor.from_agent_and_tools(agent=agent, tools=team, verbose=True)
    agent_response = agent_executor.run(user_input)
    return agent_response

def Adapa(user_input):
    adapa_response = copywriter.run(user_input)
    return adapa_response

def Ziu(user_input):
    ziu_response = web_dev.run(user_input)
    return ziu_response

def Ninmah(user_input):
    ninmah_response = analyst.run(user_input)
    return ninmah_response

def EarlGPT(user_input):
    earlgpt_response = chatgpt_chain.run(co_founder_thoughts=user_input, memory=kg_x_buffer)
    return earlgpt_response