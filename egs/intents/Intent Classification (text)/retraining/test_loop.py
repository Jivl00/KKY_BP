import intent_classifier_pytorch
import intent_classifier_keras

# net = intent_classifier_pytorch


net = intent_classifier_keras


def make_prediction(nn, inputs_and_predictions):
    model = nn.load_best_model()
    prediction = nn.predict_single(inputs_and_predictions["curr"]["input"], model)
    inputs_and_predictions["curr"]["intent"] = prediction
    print('Predicted class: {}'.format(prediction.upper()))
    if prediction == "chyba":
        print("What was the correct intent?")
        correct_intent = input()
        print(inputs_and_predictions)
        nn.retrain(inputs_and_predictions, correct_intent, retrain_all=True, add_intent=False)
    return prediction


if __name__ == '__main__':
    input_output_history = {"prev": {"input": [""], "intent": [""]}, "curr": {"input": [], "intent": []}}
    while True:
        print('Enter sentence: ')
        input_sentence = input()
        input_output_history["prev"]["input"] = input_output_history["curr"]["input"]
        input_output_history["prev"]["intent"] = input_output_history["curr"]["intent"]
        input_output_history["curr"]["input"] = input_sentence
        y_pred = make_prediction(net, input_output_history)

