import json

times = {"pytorch_big_time_req": {}, "intent_classifier_keras": {}}
times["pytorch_big_time_req"]["lib import"] = []
times["intent_classifier_keras"]["lib import"] = []
times["pytorch_big_time_req"]["model load"] = []
times["intent_classifier_keras"]["model load"] = []
times["pytorch_big_time_req"]["model inference"] = []
times["intent_classifier_keras"]["model inference"] = []

with open('temp/times_laptop.json', 'w') as f:
    o = json.dumps(times, indent=4)
    f.write(o)