import random

with open('data_robot.txt', 'r', encoding='utf-8') as f:
    data = f.readlines()

# shuffle the data
random.seed(1) #1,3
# 3 add nothing
# 5,6 all classes at once
random.shuffle(data)

with open('data_robot_shuffled.txt', 'w', encoding='utf-8') as f:
    f.writelines(data)