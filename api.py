from flask import Flask, request
import subprocess
import json
import re
import os

app = Flask(__name__)

@app.route('/api', methods=['POST'])
def api():
    print('API called')
    # get the question from the POST request
    question = request.get_json(force=True)
    # get the question from the JSON
    question = question['question']
    # change the directory to the llama directory
    os.chdir('llama.cpp')
    # run the command
    answer = subprocess.check_output(['./main', '--threads', '8', '--n-gpu-layers', '8', '--model', 'llama-2-13b-chat.ggmlv3.q4_0.bin', '--color', '--ctx-size', '2048', '--temp', '0.7', '--repeat_penalty', '1.1', '--n-predict', '-1', '--prompt', '[INST] ' + question + ' [/INST]'], universal_newlines=True)
    # return the answer in json
    answer_cleaned = re.sub(r'\[INST\].*?\[/INST\]\s*', '', answer)
    result = json.dumps({'answer': answer_cleaned})
    return result

if __name__ == '__main__':
    app.run(debug=True, host='localhost', port=8080)
