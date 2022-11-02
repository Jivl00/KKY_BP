import json
import random
import sys
import logging
logging.basicConfig(level=logging.ERROR)
from speechcloud.quick_answer import google_it
from intent_classifier import intent_classifier_pytorch_minimal as net
# from intent_classifier import intent_classifier_keras_minimal as net


def make_prediction(nn):
    global input_sentence
    model = nn.load_best_model()

    print('Enter sentence: ')
    input_sentence = input()
    return nn.predict_single(input_sentence, model)


def pozdrav(pozdraveni):
    global answers
    return random.choice(answers['POZDRAV'])


def pokyn(input_sentence):
    return "Rozkaz"


def poznej(input_sentence):
    return "Je to tma."


def stop(input_sentence):
    sys.exit(0)


if __name__ == '__main__':
    with open('answers.json', 'r', encoding='utf-8') as f:
        answers = json.load(f)
    input_sentence = ""
    intent_answers = {"POZDRAV": pozdrav, "VYGOOGLI": google_it.search,
                      "POKYN": pokyn, "KALENDÁŘ": google_it.search, "POZNEJ": poznej, "STOP": stop}
    print("Jsem mluvící robot. Mohu vám nějak pomoci?")
    while True:
        y_pred = make_prediction(net)
        print('({})\t{}'.format(y_pred.upper(), intent_answers[y_pred.upper()](input_sentence)
        if intent_answers[y_pred.upper()](input_sentence) else "Nerozumím."))
