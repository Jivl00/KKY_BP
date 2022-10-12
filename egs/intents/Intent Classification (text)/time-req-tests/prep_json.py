import json

times = {"intent_classifier_pytorch": {}, "intent_classifier_keras": {}}
times["intent_classifier_pytorch"]["lib import"] = []
times["intent_classifier_keras"]["lib import"] = []
times["intent_classifier_pytorch"]["model load"] = []
times["intent_classifier_keras"]["model load"] = []
times["intent_classifier_pytorch"]["model inference"] = []
times["intent_classifier_keras"]["model inference"] = []

with open('temp/times_laptop.json', 'w') as f:
    o = json.dumps(times, indent=4)
    f.write(o)