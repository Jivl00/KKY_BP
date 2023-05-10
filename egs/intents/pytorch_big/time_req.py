import time
import json


def load(nn):
    model = nn.load_best_model()
    return model


def inference(nn, model):
    input_sentence = "jak řeknu sbohem v čínštině"
    prediction = nn.predict_single(input_sentence, model)
    return prediction


if __name__ == '__main__':
    # load temp/times_laptop.json
    times = json.load(open('temp/times_laptop.json', 'r', encoding='utf-8-sig'))

    import pytorch_big_time_req as intent_classifier_pytorch
    import intent_classifier_keras


    for lib in (intent_classifier_pytorch, intent_classifier_keras):
        t0 = time.time()
        net = lib
        model = load(net)
        t1 = time.time() - t0
        print(f'model load time: {t1}, lib: {lib}')
        times[lib.__name__]['model load'].append(t1)

        t0 = time.time()
        inference(net, model)
        t1 = time.time() - t0
        print(f'model inference time: {t1}, lib: {lib}')
        times[lib.__name__]['model inference'].append(t1)

    # with open('temp/times_laptop.json', 'w') as f:
    #     o = json.dumps(times, indent=4)
    #     f.write(o)
