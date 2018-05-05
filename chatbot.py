# coding:utf-8
from flask import Flask, render_template, request, jsonify
from predict import Predictor
from db import DbClient

p = Predictor()
app = Flask(__name__)
db = DbClient()


@app.route('/')
def hello_world():
    return render_template('index.html')


@app.route('/ask/', methods=['GET', 'POST'])
def get_question():
    try:
        ask = request.args.get('ask', '')
        db.put(ask)
        answer = p.get_answer(ask)
        return jsonify({'answer': answer})
    except Exception as e:
        print(e)
        return jsonify({'answer': '大事不好了，系统故障了，一会再来吧'})


def run():
    app.run(host='0.0.0.0', port=9999, debug=True)


if __name__ == '__main__':
    run()
    db.close()
