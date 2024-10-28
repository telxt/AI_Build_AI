from OpenHands.generate_code import run_openhands
import os

def run(prompt):
    os.chdir('OpenHands')
    result = run_openhands(prompt)
    return result

if __name__ == '__main__':
    run('Generate a python file illustrating the architecture of a Transformer named as transformer.py!')