from flask import Flask, request
import subprocess
import json
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
    answer = subprocess.check_output(['./main', '--threads', '12', '--n-gpu-layers', '8', '--model', 'llama-2-13b-chat.ggmlv3.q4_0.bin', '--color', '--ctx-size', '2048', '--temp', '0.5', '--repeat_penalty', '1.1', '--n-predict', '-1', '--prompt', '[INST] ' + question + ' [/INST]'], universal_newlines=True)
    # return control back to root directory
    os.chdir('../')
    # split the answer by [/INST] tag and get the second part
    answer_cleaned = answer.split('[/INST]')[-1]
    # return the answer in json format
    result = json.dumps({'answer': answer_cleaned})
    return result

if __name__ == '__main__':
    app.run(debug=True, host='localhost', port=8080)
