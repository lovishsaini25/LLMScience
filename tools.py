from langchain_community.utilities import ArxivAPIWrapper,WikipediaAPIWrapper
from langchain_community.tools import ArxivQueryRun,WikipediaQueryRun,DuckDuckGoSearchRun
from langchain.agents.agent_types import AgentType
from langchain.agents import Tool, initialize_agent
from langchain.prompts import PromptTemplate
from langchain.chains import LLMMathChain, LLMChain
# from langchain_experimental.llm_symbolic_math.base import LLMSymbolicMathChain

class BuiltInTools:
    '''
    DuckDuckGoSearchAPIWrapper, ArxivAPIWrapper param is optional for QueryRun,
    whereas in Wikipedia it was required
    '''
    def arxiv_tool(self):
        '''
        Arxiv tool
        '''
        # Arxiv API to conduct searches and fetch document summaries
        # Checks 1 top schored document with 1000 limit on the document length
        arxiv_connector=ArxivAPIWrapper(top_k_results=1, doc_content_chars_max=1000)
        # Tool that searches the Arxiv API
        arxiv=ArxivQueryRun(api_wrapper=arxiv_connector, verbose=True)
        return arxiv

    def wikipedia_tool(self):
        ## Wikipedia tool
        wiki_connector=WikipediaAPIWrapper(top_k_results=1,doc_content_chars_max=1000)
        wiki=WikipediaQueryRun(api_wrapper=wiki_connector, verbose=True)
        return wiki

    def search_engine(self):
        """
        A tool that searches the internet
        """
        search=DuckDuckGoSearchRun(name="SearchEngine", verbose=True)
        return search

class CustomTools:
    def __init__(self, model) -> None:
        self.model = model

    def logical_tool(self):
        prompt="""
            You are an agent tasked with solving the user's mathematical or logical reasoning question. 
            Logically arrive at the solution and provide a detailed, step-by-step explanation for the question below.
            Use deductive, inductive, or abductive reasoning as appropriate and display your reasoning point-wise.

            Here are some examples to guide you:
            1. Deductive Reasoning: 
            - If all mammals are warm-blooded and dolphins are mammals,
            can we conclude that dolphins are warm-blooded? Explain why.
    
            2. Mathematical Problem-Solving:
            - Given the function f(x) = 2x + 5, solve for x when f(x) = 15. Show your steps and reasoning.

            3. Hypothetical Reasoning: 
            - Imagine a world where the force of gravity is twice as strong as it is on Earth.
            How would this impact human and animal physiology? Provide reasoning for your assumptions.

            Now, solve the question below in a similar logical manner:
            Question: {question}
            Answer:
            """
        prompt_template=PromptTemplate(input_variables=["question"],template=prompt)
        ## Make a chain of LLM, and the above prompt -->> prompt | llm
        chain=LLMChain(llm=self.model,
                       prompt=prompt_template,
                       verbose=True)

        # Since Description is passed as Prompts by Langchain, we didn't need add it explicitly
        # We did it to make the LLM chain to understand the requirements well.
        reasoning_tool=Tool(name="Reasoning tool",
                            func=chain.run,
                            description="A tool for answering logic-based and reasoning questions.")
        return reasoning_tool

    def numerical_math_tool(self):
        # Question: What is 13 raised to the .3432 power?
        # normal QA LLM fails because it can't perform this calculation
        # LLMMathChain has internal prompts that make our LLM model use Maths library in Python
        # Hence, we use LLMMathChain or else we need to provide a good prompt.
        # print(math_chain.prompt.template) --> It will return the internal prompt used by LLMMathChain
        math_chain=LLMMathChain.from_llm(llm=self.model, verbose=True)
        calculator=Tool(name="Calculator",
                        func=math_chain.run,
                        description="Used for answering logic-based and reasoning-heavy science questions.")
        return calculator

    def equation_math_tool(self):
        ## LLMSymbolicMathChain is experimental yet and does not produce the desirable output
        # equation_chain=LLMSymbolicMathChain.from_llm(llm=self.model, allow_dangerous_requests=False,
        #                                              verbose=True)
        # equation_tool = Tool(name="MathSolver",
        #                       func=equation_chain.run,
        #                       description="A tool for solving mathematical equations.")


        prompt_template=PromptTemplate(input_variables=["question"], template="""
            You are an expert at solving Math's problems. Solve the system of equations using Python's `fsolve` from `scipy.optimize`.
            So we hooked you up to a Python 3 kernel, and now you can execute code. If anyone gives you a hard math equations, just use this format and we’ll take care of the rest:
            Question: ${{Question with hard calculation.}}
            ```python
            ${{Code that prints what you need to know}}
            ```
            ```output
            ${{Output of your code}}
            ```
            Answer: ${{Answer}}

            Here is an example
            Question:
            1. 6^x + 6^y = 42
            2. x + y = 3

            ```python
            from scipy.optimize import fsolve
            import numpy as np

            # Define the system of equations
            def equations(vars):
                x, y = vars
                eq1 = (6**x + 6**y) - 42  # First equation: 6^x + 6^y = 42
                eq2 = x + y - 3  # Second equation: x + y = 3
                return [eq1, eq2]

            # Initial guesses for x and y
            initial_guess = [1, 3]

            # Solve the system of equations
            solution = fsolve(equations, initial_guess)
            solution
            ```
            ```output
            x = 1 and y = 2
            ```

            Now, solve this question: {question}
            """)

        ## Make a chain of LLM, and the above prompt -->> prompt | llm
        equation_chain=LLMChain(llm=self.model,
                       prompt=prompt_template,
                       verbose=True,
                       return_final_only=False)
        
        equation_tool=Tool(name="Math equation solving expert",
                        func=equation_chain.run,
                        description="Used for answering equations.")
        return equation_tool
