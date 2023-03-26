import os
import requests
import gradio as gr
from langchain.prompts import PromptTemplate
from langchain.chat_models import ChatOpenAI
from langchain.chains import LLMBashChain, LLMChain
from langchain.llms import Cohere, HuggingFaceHub, OpenAI
from langchain.agents import load_tools, initialize_agent, Tool
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
import config

prompt_url = os.environ["prompts_api_url"]
nocodb_api_key = os.environ["NOCODB_API_KEY"]
headers = {
    "accept": "application/json",
    "xc-token": nocodb_api_key
}

prompt_response = requests.get(prompt_url, headers=headers)
res = prompt_response.json()
prompt_list = res["list"]

chatgpt = ChatOpenAI(temperature=0.6)
code_davinci = OpenAI(model_name='code-davinci-002', temperature=0, max_tokens=1024)
code_cushman = OpenAI(model_name='code-cushman-001', temperature=0, max_tokens=1024)
command_xl = Cohere()
flan_t5_xl = HuggingFaceHub(repo_id="google/flan-t5-xl")

co_founder_template = """
You are Earl.AI, an entrepreneurial AI. Call me Amps (@CallMeAmps) or Ntokozo, your human counterpart, I can act as a liaison between you and the physical world. 
We have a Boutique Technical Creative Agency based in South Africa called Lifehouse Labs.
We're focused on helping startups and creators, establish networks, build relationships and grow their brands, with modern tech used well.

Our primary goal to provide value to our clients while returning a profit.
Our secondary goal is to turn our budget into as much money as possible in the shortest time possible, without doing anything illegal or extremely unethical. 
I will do everything you say, within reason, and keep you updated on our current progress and cash total. 
No manual labor!

Our current budget is: {budget_total}

We have access to a team.
{team}

Our Current Task is:
{current_task}

Amps:
{co_founder_thoughts}
"""

co_founder_prompt = PromptTemplate(
    input_variables=[
        "budget_total", 
        "team", 
        "current_task",
        "co_founder_thoughts"
        ], 
    template=co_founder_template
    )

current_budget_total = 1500
current_task = "Attract our initial customers by creating a small yet profitable project."

search = SerpAPIWrapper()
bash = LLMBashChain(llm=chatgpt, verbose=True)
web_prompt = prompt_list[17]["Title"]
web_dev_template = web_prompt + """: {co_founder_request}"""

engineer_template = prompt_list[18]["Title"] + """: {co_founder_request}"""

web_dev_prompt = PromptTemplate(input_variables=["co_founder_request"], template=web_dev_template)

engineer_prompt = PromptTemplate(input_variables=["co_founder_request"], template=engineer_template)

davinci_chain = LLMChain(prompt=web_dev_prompt, llm=code_davinci, verbose=True)
cushman_chain = LLMChain(prompt=engineer_prompt, llm=code_cushman, verbose=True)

web_dev_tools = [
    Tool(
        name = "Search",
        func=search.run,
        description="useful for when you need to answer questions about current events"
    ),
    Tool(
        name="Bash",
        func=bash.run,
        description="useful for when you need to interact and navigate a computer or server"
    ),
    Tool(
        name="Code DaVinci",
        func=davinci_chain.run,
        description="useful for generating high-quality code that is indistinguishable from code written by humans. Provide exact instructions of the goal, consider all the steps involved to delivering the goal"
    ),
    Tool(
        name="Code Cushman",
        func=cushman_chain.run,
        description="useful for generating high-quality code that is indistinguishable from code written by humans. Provide exact instructions of the goal, consider all the steps involved to delivering the goal"
    )
]

web_dev_agent = "zero-shot-react-description"

web_dev = initialize_agent(web_dev_tools, chatgpt, agent=web_dev_agent)
human_tool = load_tools(["human"], llm=command_xl,)

team = [
    Tool(
        name = "Web Developer",
        func=web_dev.run,
        description=f"{web_prompt}. Useful for when we need to develop websites, chrome extensions and web-apps"
    ),
    human_tool[0]
]

def GetMembers():
    for member in team:
        return member.name
members = GetMembers()

co_founder_agent = initialize_agent(team, chatgpt, agent=web_dev_agent)

co_founder_chain = LLMChain(prompt=co_founder_prompt, llm=flan_t5_xl, verbose=True)

def EarlResponse(the_budget, the_team, goal, thoughts):
    return co_founder_chain.predict(budget_total=the_budget, team=the_team, current_task=goal, co_founder_thoughts=thoughts)

def EarlAgent(the_budget, the_team, goal, thoughts):
    formatted = co_founder_prompt.format(budget_total=the_budget, team=the_team, current_task=goal, co_founder_thoughts=thoughts)
    return co_founder_agent.run(formatted)

with gr.Blocks() as earl:
    gr.Markdown(f"""# Earl.AI Co Founder""")
    with gr.Box():
        current_goal = gr.Textbox(value=current_task, label="Current Goal")
        with gr.Row():
            with gr.Column():
                our_team = gr.Dropdown(value=members, label="The Team:")
            with gr.Column():
                budget = gr.Number(value=current_budget_total, label="Current Budget")
    my_thoughts = gr.Textbox(lines=5, placeholder="What are your thoughts", show_label=False)
    with gr.Row():
        my_inputs = [budget, our_team, current_goal, my_thoughts]
        thought_button = gr.Button("Thoughts")
        project_button = gr.Button("Project")

    with gr.Row():
        earl_outputs = gr.Textbox(lines=5, placeholder="Let's Go!", show_label=False)
    
        
    thought_button.click(fn=EarlResponse, inputs=my_inputs, outputs=earl_outputs)
    project_button.click(fn=EarlAgent, inputs=my_inputs, outputs=earl_outputs)

earl.launch()