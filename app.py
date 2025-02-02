# Rename `os.environ` to `env` for nicer code
from os import environ as env
from crewai import Agent, Task, Crew, LLM, Process
from dotenv import load_dotenv,find_dotenv
import markdown
from langchain_google_genai import ChatGoogleGenerativeAI
from util import get_serper_api_key

from crewai_tools import DirectoryReadTool, \
                         FileReadTool, \
                         SerperDevTool

from crewai.tools import BaseTool
from pydantic import BaseModel, Field


load_dotenv('config/.env')



# Rename `os.environ` to `env` for nicer code

print('GOOGLE_API_KEY:  {}'.format(env['GOOGLE_API_KEY']))

GEMINI_API_KEY = env['GOOGLE_API_KEY']

env["GEMINI_API_KEY"] = env['GOOGLE_API_KEY']
MODEL_NAME = "models/text-embedding-004"

env["SERPER_API_KEY"] = env['SERPER_KEY']


llm = LLM(
    model = "gemini/gemini-1.5-pro-latest",
    temperature = 0.7
)

##################################################
#   Creating Agents
#
##################################################

sales_rep_agent = Agent(
    role="Sales Representative",
    goal="Identify high-value leads that match "
         "our ideal customer profile",
    backstory=(
        "As a part of the dynamic sales team at {agency_name}, "
        "your mission is to scour "
        "the digital landscape for potential leads. "
        "Armed with cutting-edge tools "
        "and a strategic mindset, you analyze data, "
        "trends, and interactions to "
        "unearth opportunities that others might overlook. "
        "Your work is crucial in paving the way "
        "for meaningful engagements and driving the company's growth."
    ),
    allow_delegation = False,
    verbose = True,
    llm = llm,
)


lead_sales_rep_agent = Agent(
    role = "Lead Sales Representative",
    goal = "Nurture leads with personalized, compelling communications",
    backstory = (
        "Within the vibrant ecosystem of {agency_name}'s sales department, "
        "you stand out as the bridge between potential clients "
        "and the solutions they need."
        "By creating engaging, personalized messages, "
        "you not only inform leads about our offerings "
        "but also make them feel seen and heard."
        "Your role is pivotal in converting interest "
        "into action, guiding leads through the journey "
        "from curiosity to commitment."
    ),
    allow_delegation = False,
    verbose = True,
    llm = llm
)

###
#   Creating Tools
###

directory_read_tool = DirectoryReadTool(directory='./instructions')
file_read_tool = FileReadTool()
search_tool = SerperDevTool()


####### Important  #####
# Implemented Custom Tool
#
#  Create a custom tool using crewAi's BaseTool class
#
##############

## Custom a Crewai Tool
#
    # Every Tool needs to have a name and a description.
    #
    # For simplicity and classroom  purposes, SentimentAnalysisTool will return positive
    # for every text.
    #
    # When running locally, you can customize the code with your logic in the _run function.
#

class SentimentAnalysisTool(BaseTool):
    name: str = "Sentiment Analysis Tool"
    description: str = ("Analyzes the sentiment of text "
                        "to ensure positive and engaging communication.")

    def _run(self, text: str) -> str:
        # Your custom code tool goes here
        return "positive"

sentiment_analysis_tool = SentimentAnalysisTool()


###
# Creating Tasks
####

# - The Lead Profiling Task is using crewAI Tools.


lead_profiling_task = Task(
    description=(
        "Conduct an in-depth analysis of {lead_name}, "
        "a company in the {industry} sector "
        "that recently showed interest in our solutions. "
        "Utilize all available data sources "
        "to compile a detailed profile, "
        "focusing on key decision-makers, recent business "
        "developments, and potential needs "
        "that align with our offerings. "
        "This task is crucial for tailoring "
        "our engagement strategy effectively.\n"
        "Don't make assumptions and "
        "only use information you absolutely sure about."
    ),
    expected_output=(
        "A comprehensive report on {lead_name}, "
        "including company background, "
        "key personnel, recent milestones, and identified needs. "
        "Highlight potential areas where "
        "our solutions can provide value, "
        "and suggest personalized engagement strategies."
    ),
    tools= [directory_read_tool, file_read_tool, search_tool],
    agent = sales_rep_agent,
)

# The Personalized Outreach Task is using your custom Tool SentimentAnalysisTool,
# as well as crewAI's SerperDevTool (search_tool).

# personalized outreach task
##
#  Original Task Design
#
#
# personalized_outreach_task = Task(
#     description=(
#         "Using the insights gathered from "
#         "the lead profiling report on {lead_name}, "
#         "craft a personalized outreach campaign "
#         "aimed at {key_decision_maker}, "
#         "the {position} of {lead_name}. "
#         "The campaign should address their recent {milestone} "
#         "and how our solutions can support their goals. "
#         "Your communication must resonate "
#         "with {lead_name}'s company culture and values, "
#         "demonstrating a deep understanding of "
#         "their business and needs.\n"
#         "Don't make assumptions and only "
#         "use information you absolutely sure about."
#     ),
#     expected_output=(
#         "A series of personalized email drafts "
#         "tailored to {lead_name}, "
#         "specifically targeting {key_decision_maker}."
#         "Each draft should include "
#         "a compelling narrative that connects our solutions "
#         "with their recent achievements and future goals. "
#         "Ensure the tone is engaging, professional, "
#         "and aligned with {lead_name}'s corporate identity."
#     ),
#     tools=[sentiment_analysis_tool, search_tool],
#     agent=lead_sales_rep_agent,
# )

# change line SEVEN  with
#    "and how our solutions can support their goals. "
# by:
#     "and how our services can support their healthcare. "

personalized_outreach_task = Task(
    description=(
        "Using the insights gathered from "
        "the lead profiling report on {lead_name}, "
        "craft a personalized outreach campaign "
        "aimed at {key_decision_maker}, "
        "the {position} of {lead_name}. "
        "The campaign should address their recent {milestone} "
        "and how our services can support their healthcare. "
        "Your communication must resonate "
        "with {lead_name}'s culture and values, "
        "demonstrating a deep understanding of "
        "their business and needs.\n"
        "Don't make assumptions and only "
        "use information you absolutely sure about."
    ),
    expected_output=(
        "A series of personalized email drafts "
        "tailored to {lead_name}, "
        "specifically targeting {key_decision_maker}."
        "Each draft should include "
        "a compelling narrative that connects our solutions "
        "with their recent achievements and future goals. "
        "Ensure the tone is engaging, professional, "
        "and aligned with {lead_name}'s corporate identity."
    ),
    tools=[sentiment_analysis_tool, search_tool],
    agent=lead_sales_rep_agent,
)

# Creating the Crew
crew = Crew(
    agents=[sales_rep_agent,
            lead_sales_rep_agent],

    tasks=[lead_profiling_task,
           personalized_outreach_task],

    verbose = True,
    memory = True,
    embedder = {
        "provider": "google",
        "config": {
            "api_key": env["GEMINI_API_KEY"],
            "model_name": "models/text-embedding-004",
            "model": "models/text-embedding-004",
        }
    }
)

#Running the Crew
#Note: LLMs can provide different outputs for they same input,

inputs = {
    "agency_name":"Dulcine insures",
    "lead_name": "DeepLearningAI",
    "industry": "Online Learning Platform",
    "key_decision_maker": "Andrew Ng",
    "position": "CEO",
    "milestone": "product launch"
}

result = crew.kickoff(inputs = inputs)

print(result.raw)