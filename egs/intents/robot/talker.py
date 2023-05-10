import rospy
from std_msgs.msg import String
import random

def talker():
    pub = rospy.Publisher('input', String, queue_size=10)
    rospy.init_node('talker', anonymous=True)
    # define how many times per second
    # will the data be published
    # let's say 10 times/second or 10Hz
    rate = rospy.Rate(1)
    while not rospy.is_shutdown():
        inputs = ["Ahoj", "Jak se máš?", "Co to je?", "Něco", "Kolik je hodin?", "Kdo je prezidentem USA?", "Jaké je počasí v Plzni?"]
        hello_str = random.choice(inputs)
        rospy.loginfo(hello_str)
        pub.publish(hello_str)
        rate.sleep()

if __name__ == '__main__':
    try:
        talker()
    except rospy.ROSInterruptException:
        pass
