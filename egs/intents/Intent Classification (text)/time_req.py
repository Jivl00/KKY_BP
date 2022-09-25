import time


def load(nn):
    model = nn.load_best_model()
    return model


def inference(nn, model):
    input_sentence = "Ahoj"
    return nn.predict_single(input_sentence, model)


if __name__ == '__main__':
    start = time.time()
    import intent_classifier_pytorch
    # import intent_classifier_keras

    net = intent_classifier_pytorch
    model = load(net)
    end = time.time()
    time1 = end - start
    start = time.time()
    inference(net, model)
    end = time.time()
    time2 = end - start
    with open('temp/times_laptop_pytorch.txt', 'a') as f:
        f.write('load time: {}  inference time: {}\n'.format(time1, time2))
