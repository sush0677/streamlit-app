import streamlit as st
import os
from azure.identity import ChainedTokenCredential, ManagedIdentityCredential, AzureCliCredential
from langchain_core.messages import HumanMessage
from langchain_openai import AzureChatOpenAI
from langchain.prompts import (
    ChatPromptTemplate,
    PromptTemplate,
    SystemMessagePromptTemplate,
    AIMessagePromptTemplate,
    HumanMessagePromptTemplate,
)
from langchain.chains import LLMChain, SimpleSequentialChain
from langchain.chains import SequentialChain
from langchain.chains.router.multi_prompt_prompt import MULTI_PROMPT_ROUTER_TEMPLATE
from langchain.chains.router.llm_router import LLMRouterChain,RouterOutputParser
from langchain.chains.router import MultiPromptChain
from langchain.chains import TransformChain
from langchain.prompts import PromptTemplate
from langchain.chains.router.llm_router import LLMRouterChain,RouterOutputParser
from langchain.chains.openai_functions import (
    create_openai_fn_chain,
    create_structured_output_chain,
)
from typing import Optional

# Create an instance of the AzureChatOpenAI model
model = AzureChatOpenAI(
    deployment_name="exq-gpt-35",
    azure_endpoint="https://exquitech-openai-2.openai.azure.com/",
    api_key="4f00a70876a542a18b30f13570248cdb",
    temperature=0,
    openai_api_version="2024-02-15-preview"
)

# Define function to generate a funny company name using LLMChain
def generate_company_name(product):
    human_prompt = HumanMessagePromptTemplate.from_template('Make up a funny company name for a company that makes: {product}')
    chat_prompt_template = ChatPromptTemplate.from_messages([human_prompt])
    chain = LLMChain(llm=model, prompt=chat_prompt_template)
    response = chain.run(product=product)
    return response

# Define function for SimpleSequentialChain: Responds with a random joke
def get_joke():
    template = "Tell me a joke on the {topic}"
    first_prompt = ChatPromptTemplate.from_template(template)
    chain_one = LLMChain(llm=model,prompt=first_prompt)
    template2 = "Make the joke a real one {outline}"
    second_prompt = ChatPromptTemplate.from_template(template2)
    chain_two = LLMChain(llm=model,prompt=second_prompt)
    full_chain = SimpleSequentialChain(chains=[chain_one,chain_two],verbose=True)
    result = full_chain.run('computers')
    return result

# Define function for LLMRouterChain: Responds differently based on input type
def route_response(user_input):
  beginner_template = '''You are a physics teacher who is really
  focused on beginners and explaining complex topics in simple to understand terms. 
  You assume no prior knowledge. Here is the question\n{input}'''

  expert_template = '''You are a world expert physics professor who explains physics topics
  to advanced audience members. You can assume anyone you answer has a 
  PhD level understanding of Physics. Here is the question\n{input}'''


  prompt_infos = [
    {'name':'advanced physics','description': 'Answers advanced physics questions',
     'prompt_template':expert_template},
    {'name':'beginner physics','description': 'Answers basic beginner physics questions',
     'prompt_template':beginner_template},
    
  ]
  destinations = [f"{p['name']}: {p['description']}" for p in prompt_infos]
  destinations_str = "\n".join(destinations)
  destination_chains = {}
  router_template = MULTI_PROMPT_ROUTER_TEMPLATE.format(
    destinations=destinations_str
  )
  router_prompt = PromptTemplate(
    template=router_template,
    input_variables=["input"],
    output_parser=RouterOutputParser(),
  )
  for p_info in prompt_infos:
    name = p_info["name"]
    prompt_template = p_info["prompt_template"]
    prompt = ChatPromptTemplate.from_template(template=prompt_template)
    chain = LLMChain(llm=model, prompt=prompt)
    destination_chains[name] = chain

  default_prompt = ChatPromptTemplate.from_template("{input}")
  default_chain = LLMChain(llm=model,prompt=default_prompt)
  from langchain.chains.router import MultiPromptChain

  router_chain = LLMRouterChain.from_llm(model, router_prompt)

  chain = MultiPromptChain(router_chain=router_chain, 
                         destination_chains=destination_chains, 
                         default_chain=default_chain, verbose=True
                        )

  result = chain.run(input=user_input)
  return result
# Define function for TransformChain: Translates input text to uppercase
def transform_text(input_text):
    return input_text.upper()

# Define function for MathChain: Evaluates mathematical expressions
def evaluate_expression(expression):
    try:
        result = eval(expression)
        return f"The result of {expression} is {result}"
    except:
        return "Invalid expression! Please enter a valid mathematical expression."

# Define main function to render Streamlit app
def main():
    st.title("LangChain Showcase")
    st.sidebar.title("Chains")
    selected_chain = st.sidebar.selectbox("Select a chain", ["LLMChain", "SimpleSequentialChain", "LLMRouterChain", "TransformChain", "MathChain"])

    if selected_chain == "LLMChain":
        st.subheader("LLMChain")
        st.write("Functionality: Generates a funny company name based on a given product.")
        product = st.text_input("Enter a product", "Computers")
        generate_button = st.button("Generate")

        if generate_button:
            company_name = generate_company_name(product)
            st.success(f"Here's a funny company name for a {product} company: {company_name}")

    elif selected_chain == "SimpleSequentialChain":
        st.subheader("SimpleSequentialChain")
        st.write("Functionality: Responds with a random joke.")
        joke_button = st.button("Tell me a joke")

        if joke_button:
            joke = get_joke()
            st.success(joke)

    elif selected_chain == "LLMRouterChain":
        st.subheader("LLMRouterChain")
        st.write("Functionality: Ask me any question related physics ")
        user_input = st.text_input("Enter your input")
        route_button = st.button("Route Response")

        if route_button:
            response = route_response(user_input)
            st.success(response)

    elif selected_chain == "TransformChain":
        st.subheader("TransformChain")
        st.write("Functionality: Translates input text to uppercase.")
        input_text = st.text_input("Enter text")
        transform_button = st.button("Transform")

        if transform_button:
            transformed_text = transform_text(input_text)
            st.success(transformed_text)

    elif selected_chain == "MathChain":
        st.subheader("MathChain")
        st.write("Functionality: Evaluates mathematical expressions.")
        expression = st.text_input("Enter a mathematical expression")
        evaluate_button = st.button("Evaluate")

        if evaluate_button:
            result = evaluate_expression(expression)
            st.success(result)

if __name__ == "__main__":
    main()
