import intent_classifier_pytorch
import intent_classifier_keras


net = intent_classifier_pytorch


# net = intent_classifier_keras


def make_prediction(nn):
    model = nn.load_best_model()

    print('Enter sentence: ')
    input_sentence = input()
    return nn.predict_single(input_sentence, model)


if __name__ == '__main__':
    while True:
        y_pred = make_prediction(net)
        print('Predicted class: {}'.format(y_pred.upper()))
