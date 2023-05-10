import sys
sys.path.append('/home/jivl/catkin_ws/src/beginner_tutorials/scripts')
import logging
logging.basicConfig(level=logging.ERROR)
# from intent_classifier import intent_classifier_pytorch_minimal as net
from intent_classifier import intent_classifier_keras_minimal as net
from speechcloud.quick_answer import google_it
import rospy
from std_msgs.msg import String
import random



def make_prediction(nn):
    global input_sentence
    model = nn.load_best_model()
    return nn.predict_single(input_sentence, model)


def pozdrav(pozdraveni):
    global answers
    return random.choice(answers['POZDRAV'])


def pokyn(input_sentence):
    return "Rozkaz"


def poznej(input_sentence):
    return "Je to tma."


def stop(input_sentence):
    print("Konec")
    sys.exit(0)


def callback(data):
    global input_sentence
    input_sentence = data.data
    print(data.data)
    y_pred = make_prediction(net)
    print('({})\t{}'.format(y_pred.upper(), intent_answers[y_pred.upper()](input_sentence)
    if intent_answers[y_pred.upper()](input_sentence) else "Nerozumím."))

    pub = rospy.Publisher('output', String, queue_size=10)
    pub.publish(intent_answers[y_pred.upper()](input_sentence)
    if intent_answers[y_pred.upper()](input_sentence) else "Nerozumím.")


def listener():

    # In ROS, nodes are uniquely named. If two nodes with the same
    # name are launched, the previous one is kicked off. The
    # anonymous=True flag means that rospy will choose a unique
    # name for our 'listener' node so that multiple listeners can
    # run simultaneously.
    rospy.init_node('listener', anonymous=True)

    rospy.Subscriber('input', String, callback)

    # spin() simply keeps python from exiting until this node is stopped
    rospy.spin()


if __name__ == '__main__':
    answers = {"POZDRAV": ["Ahoj","Dobrý den","Zdravím","Čau"]}
    input_sentence = ""
    intent_answers = {"POZDRAV": pozdrav, "VYGOOGLI": google_it.search,
                      "POKYN": pokyn, "KALENDÁŘ": google_it.search, "POZNEJ": poznej, "STOP": stop}
    listener()
