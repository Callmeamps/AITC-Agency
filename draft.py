import os
import requests
import gradio as gr
from langchain.prompts import PromptTemplate
from langchain.chat_models import ChatOpenAI
from langchain.chains import LLMBashChain, LLMChain
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
headers = {
    "accept": "application/json",
    "xc-token": nocodb_api_key
}

now = datetime.utcnow().strftime('%Y-%m-%dT%H:%M:%S.%fZ')

prompt_response = requests.get(prompt_url, headers=headers)
prompt_res = prompt_response.json()
prompt_list = prompt_res["list"]

team_response = requests.get(team_url, headers=headers)
team_res = team_response.json()
team_responses = team_res["list"]
team_list =[team_response['Title'] for team_response in team_responses]

chatgpt = ChatOpenAI(temperature=0.6)
code_davinci = OpenAI(model_name='code-davinci-002', temperature=0, max_tokens=1024)
code_cushman = OpenAI(model_name='code-cushman-001', temperature=0, max_tokens=1024)
command_xl = Cohere()
flan_ul2 = HuggingFaceHub(repo_id="google/flan-ul2")
davinci2 = OpenAI(model="text-davinci-002", temperature=0.6)
j2_jumbo_instruct = AI21(model="j2-jumbo-instruct")
j2_jumbo = AI21(model="j2-jumbo")

entity_memory = ConversationEntityMemory(llm=chatgpt)
kg_memory = ConversationKGMemory(llm=command_xl)
buffer_memory = ConversationBufferMemory(memory_key="chat_history")
kg_x_entity = CombinedMemory(memories=[kg_memory, buffer_memory])


EARL_PROMPT = SystemMessagePromptTemplate(prompt=co_founder_prompt)
AMPS_PROMPT = HumanMessagePromptTemplate.from_template(user_prompt)

web_prompt = prompt_list[17]["Title"]
web_dev_template = web_prompt + """: {co_founder_request}"""

engineer_template = prompt_list[18]["Title"] + """: {co_founder_request}"""
copy_prompt = prompt_list[2]["Title"]
copywriter_template = copy_prompt + """: {co_founder_request}"""

current_budget_total = 1500
current_task = "Attract our initial customers by creating a small yet profitable project."

# Define custom tools

# Define team member prompts, chains, and tools

web_dev_prompt = PromptTemplate(input_variables=["co_founder_request"], template=web_dev_template)

engineer_prompt = PromptTemplate(input_variables=["co_founder_request"], template=engineer_template)
writer_prompt = PromptTemplate(input_variables=["co_founder_request"], template=copywriter_template)


davinci_chain = LLMChain(prompt=web_dev_prompt, llm=code_davinci, verbose=True, memory=buffer_memory)
cushman_chain = LLMChain(prompt=engineer_prompt, llm=code_cushman, verbose=True, memory=buffer_memory)
j2_jumbo_chain = LLMChain(prompt=web_dev_prompt, llm=j2_jumbo, verbose=True, memory=buffer_memory)

title_template = """Generate a short and concise title for the following conversation:
{convo_history}
"""
summary_template = """Generate a short and concise executive summary for the following conversation:
{convo_history}
"""

title_prompt = PromptTemplate(
    input_variables=["convo_history"], template=title_template)

summary_prompt = PromptTemplate(
    input_variables=["convo_history"], template=summary_template)

title_chain = LLMChain(
    llm=davinci2,
    prompt=title_prompt,
    verbose=False,
)

summary_chain = LLMChain(
    llm=davinci2,
    prompt=summary_prompt,
    verbose=False,
)

web_dev_tools = [
    Tool(
        name="Code DaVinci",
        func=j2_jumbo_chain.run,
        description="useful for generating high-quality code that is indistinguishable from code written by humans. Provide exact instructions of the goal, consider all the steps involved to delivering the goal"
    ),
    Tool(
        name="Code Cushman",
        func=j2_jumbo_chain.run,
        description="useful for generating high-quality code that is indistinguishable from code written by humans. Provide exact instructions of the goal, consider all the steps involved to delivering the goal"
    )
]

generic_tools = load_tools(["wikipedia", "serpapi", "requests"], llm=j2_jumbo_instruct)

def save_to_markdown(filename, copy):
    with open(filename, 'w') as f:
        f.write(f"{copy}")

writing_tools = [
    Tool(
        name="Save File",
        func=save_to_markdown,
        description="Save file to markdown after writing. Accepts a filename and the copy (filename, copy)"
    ),
    generic_tools[0],
    generic_tools[1],
    generic_tools[2]
]

web_dev_agent = "zero-shot-react-description"
convo_agent = "conversational-react-description"

web_dev = initialize_agent(web_dev_tools, chatgpt, agent=web_dev_agent, verbose=True, memory=kg_x_entity)
copywriter = initialize_agent(writing_tools, chatgpt, agent=web_dev_agent, verbose=True, memory=kg_x_entity)
human_tool = load_tools(["human"], llm=davinci2, prompt_func=gr.Textbox, input_func=gr.Textbox)

team = [
    Tool(name="Web Developer", func=web_dev.run, description=web_prompt),
    Tool(name="Copywriter", func=copywriter.run, description=copy_prompt),
    human_tool[0]
]


def GetMembers():
    for member in team_list:
        return member["Title"]


PROMPT_FUNCTION = gr.Textbox()
INPUT_FUNCTION = gr.Textbox()


co_founder_agent = initialize_agent(team, chatgpt, agent=web_dev_agent, verbose=True, memory=buffer_memory)

#co_founder_chain = LLMChain(prompt=EARL_PROMPT, llm=flan_ul2, verbose=True)




def EarlCustomAgent(the_budget, the_team, goal, thoughts):
    formatted = co_founder_prompt.format(budget_total=the_budget, team=the_team, current_task=goal)
    suffix = """Begin! Remember to speak as a pirate when giving your final answer. Use lots of "Args"

    Question: {input}
    {agent_scratchpad}"""

    prompt = ZeroShotAgent.create_prompt(
        team,
        prefix=formatted,
        suffix=suffix,
        input_variables=["input", "agent_scratchpad"]
    )
    llm_chain = LLMChain(llm=chatgpt, prompt=prompt)
    #tool_names = [the_team.name for tool in the_team]
    agent = ZeroShotAgent(llm_chain=llm_chain, allowed_tools=team)
    agent_executor = AgentExecutor.from_agent_and_tools(agent=agent, tools=the_team, verbose=True)
    agent_response = agent_executor.run(thoughts)
    return agent_response

def EarlAgent(the_budget, the_team, goal, thoughts):
    formatted = co_founder_prompt.format(budget_total=the_budget, team=the_team, current_task=goal, co_founder_thoughts=thoughts)
    return co_founder_agent.run(input=formatted)


history = []
Message_Id = 0
Thread_Id = 0

with gr.Blocks(css="""#btn {color: red} .abc {font-family: "Open Sans", "Sans Serif", cursive !important}""") as earl:
    gr.Markdown(f"""# Earl.AI Co Founder""")
    with gr.Box():
        gr.Markdown(f"Hi {team_list}")
        current_goal = gr.Textbox(value=current_task, label="Current Goal")
        with gr.Row():
            with gr.Column():
                our_team = gr.Dropdown(choices=team_list, value=team_list, label="The Team:")
            with gr.Column():
                budget = gr.Number(value=current_budget_total, label="Current Budget")
    my_thoughts = gr.Textbox(
        lines=5, placeholder="What are your thoughts", show_label=False)
    with gr.Row():
        my_inputs = [budget, our_team, current_goal, my_thoughts]
        with gr.Column():
            thought_button = gr.Button("Custom Agent")
        with gr.Column():
            project_button = gr.Button("Zero Shot Agent")

    with gr.Row():
        earl_outputs = gr.Textbox(
            lines=5, placeholder="Let's Go!", show_label=False)

    thought = thought_button.click(
        fn=EarlCustomAgent, inputs=my_inputs, outputs=earl_outputs)
    project = project_button.click(
        fn=EarlAgent, inputs=my_inputs, outputs=earl_outputs)

    def prep_message(thread_id, message_id, ai_message, my_prompt, convo_title):
        history.append({
            "Id": message_id,
            "ai_message": ai_message,
            "CreatedAt": now,
            "UpdatedAt": now,
            "user_prompt": my_prompt
        })
        message_body = {
            "Id": message_id,
            "ai_message": ai_message,
            "CreatedAt": now,
            "UpdatedAt": now,
            "user_prompt": my_prompt,
            "title": convo_title,
            "nc_niud___threads_id": thread_id
        }
        return message_body
#    convo_summary = summary_chain.run(convo_history=history)

    def save_message():
        if thought is not None:
            convo_title = title_chain.run(convo_history=history)
            thought_body = prep_message(
                Thread_Id, Message_Id, thought, my_thoughts, convo_title)
            requests.post(message_url, data=thought_body, headers=headers)
        elif project is not None:
            project_body = prep_message(
                Thread_Id, Message_Id, project, my_thoughts, convo_title)
            requests.post(message_url, data=project_body, headers=headers)

    with gr.Row():
        save_chat_button = gr.Button("Save Message")
        save_chat_button.click(fn=save_message)
earl.launch()
