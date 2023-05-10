import time
import json


def load(nn):
    model = nn.load_best_model()
    return model


def inference(nn, model):
    input_sentence = "Ahoj"
    return nn.predict_single(input_sentence, model)


if __name__ == '__main__':
    # load temp/times_laptop.json
    times = json.load(open('temp/times_laptop.json', 'r', encoding='utf-8-sig'))

    t0 = time.time()
    import intent_classifier_pytorch

    t1 = time.time() - t0
    print(f'torch lib import time: {t1}')
    times['intent_classifier_pytorch']['lib import'].append(t1)

    t0 = time.time()
    import intent_classifier_keras

    t1 = time.time() - t0
    print(f'keras lib import time: {t1}')
    times['intent_classifier_keras']['lib import'].append(t1)

    for lib in (intent_classifier_pytorch, intent_classifier_keras):
        t0 = time.time()
        net = intent_classifier_pytorch
        model = load(net)
        t1 = time.time() - t0
        print(f'model load time: {t1}, lib: {lib}')
        times[lib.__name__]['model load'].append(t1)

        t0 = time.time()
        inference(net, model)
        t1 = time.time() - t0
        print(f'model inference time: {t1}, lib: {lib}')
        times[lib.__name__]['model inference'].append(t1)

    with open('temp/times_laptop.json', 'w') as f:
        o = json.dumps(times, indent=4)
        f.write(o)
