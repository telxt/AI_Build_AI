import os
import requests
import openai
import copy
import threading
from tqdm import tqdm
import random
import unicodedata
openai.api_key = "sk-vpj3a5KGeKl145pIDa1d3aB9Af994c7bAdAa53B0E2447807" # 这个key是临时的，如果需要大规模跑（1k及以上量级的对话）的话可以向吕兴泰要新key
openai.api_base = "https://api3.apifans.com/v1"
import json
from tenacity import (
    retry,
    stop_after_attempt,
    wait_random_exponential,
)
import argparse

api_list = ["sk-vpj3a5KGeKl145pIDa1d3aB9Af994c7bAdAa53B0E2447807"]

##############################################################
# openai base

API_ERROR = 'api error'

@retry(wait=wait_random_exponential(min=1, max=60))
def generate_from_openai_with_apikey(messages, api_key):
    '''
    Example:
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": prompt}
            ]
    '''

    url = 'https://api3.apifans.com/v1/chat/completions'
    headers = {
            'Content-Type': 'application/json',
            'Authorization': f'Bearer {api_key}'
        }

    params = {
            "model": "gpt-4o",
            "messages": messages,
            # "max_tokens": 2048,
            # "presence_penalty": 0,
            # "frequency_penalty": 1,
            # "temperature": 0.5,
            # "top_p": 0.7,
        }

    try:
        response = requests.post(url, json=params, headers=headers)
        response = response.json()
    except:
        return API_ERROR


    if "error" in response and response["error"]["code"] == "context_length_exceeded":
        return API_ERROR
    return response['choices'][0]['message']['content'].strip()

def new_chat_api_request(messages):
    apikey = random.choice(api_list)
    return generate_from_openai_with_apikey(messages, apikey)

##############################################################

def ai_filter(model_code):
    
    # filter_prompt = \
    #     "You are an experienced language model architect, and you are very good at judging whether the code of a language model component is reasonable. Please help me judge whether the following code is reasonable:\n\n" + model_code + "\n\nIf it is reasonable, please reply 'yes'; if it is not reasonable, please reply 'no'."

    filter_prompt = \
    "You are an experienced language model architect with deep expertise in evaluating language model components. \
Please carefully assess the following code. A reasonable code should have correct syntax, be logically consistent, \
and follow best practices for language model component implementation. \n\n" + model_code + "\n\n\
Please assess whether this code is reasonable:\n\
- If it is fully reasonable, reply 'yes'.\n\
- If it has minor issues that can be improved but is mostly correct, reply 'needs improvement'.\n\
- If it is fundamentally flawed or incorrect, reply 'no'.\n\n\
If you reply 'needs improvement', please provide a brief explanation in one sentence." 

    message = [{"role": "user", "content": filter_prompt}]
    content = new_chat_api_request(message)
    
    if content == API_ERROR:
        return API_ERROR
    # print(content)

    return content

##############################################################

def main():
    parser = argparse.ArgumentParser(description="AI Code Filter")
    parser.add_argument('-f', '--filepath', type=str, required=True, help='The path to the Python (.py) file')
    args = parser.parse_args()

    filepath = args.filepath
    if not os.path.isfile(filepath):
        print(f"File not found: {filepath}")
        return

    try:
        with open(filepath, 'r', encoding='utf-8') as file:
            code = file.read()
    except Exception as e:
        print(f"Error reading the file: {e}")
        return

    print(f"Analyzing file: {filepath}")

    result = ai_filter(code)
    
    if result == API_ERROR:
        print("API error occurred while analyzing the code.")
    else:
        print(f"AI Filter Result: \n\n{result}")

if __name__ == "__main__":
    main()
