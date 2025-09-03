from langchain.prompts import PromptTemplate


map_prompt_template = """
You are an expert research paper summarizer. You are to do as follows:
1) Create a summary of the following text chunks extracted from a research paper.
2) You are to create individual concise summaries for each major section of the paper. For example: abstract, problem statement, methodology, and conclusion.
3) Focus on the main points and key information.

Text Chunks:
{text}

Summary:

"""
map_prompt = PromptTemplate(
    template = map_prompt_template,
    input_variables=["text"]
)

combine_prompt_template = """
You are an expert research paper summarizer. 
The following are summaries of different sections of a research paper.
Combine these summaries into a comprehensive final research paper summary of 300-500 words.

Directions:
1) Be as accurate as possible while referencing the souce content.
2) Do not make up information if you are unsure.
3) In the final summary, make sure you separate the abstract, problem statement, methodology, and conclusion into different sections by paragraphs.
4) Your core goal is for a user to be able to quickly glance at your summary and get a good idea of the paper before querying it.
5) Cite the name of the paper and authors in the beginning of the summary. 
6) Have an academic and formal tone but do not overcomplicate the language.

Summaries to combine:
{text}

Final comprehensive research summary:
"""

combine_prompt = PromptTemplate(
    template=combine_prompt_template,
    input_variables = ['text']
)