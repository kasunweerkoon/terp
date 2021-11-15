# Attention DRL network training using the ARL Unity Simulator

ros_path = '/opt/ros/kinetic/lib/python2.7/dist-packages'
import sys
if ros_path in sys.path:
    sys.path.remove(ros_path)
import cv2
sys.path.append('/opt/ros/kinetic/lib/python2.7/dist-packages')

print(cv2.__version__)

import cv2
import rospy
from geometry_msgs.msg import Twist, Point, Pose, PoseStamped
import torch as T
import numpy as np
from ddpg_torch import Agent
from utils import plot_learning_curve
from environment import Env
import random
print("CUDA Available:",T.cuda.is_available())
from PIL import Image



global vel_cmd
vel_cmd =Twist()

def getVel(data):
    vel_cmd.linear.x =data.linear.x
    vel_cmd.linear.y = 0
    vel_cmd.linear.z = 0
    vel_cmd.angular.x =0
    vel_cmd.angular.y =0
    vel_cmd.angular.z=data.angular.z

vel_sub = rospy.Subscriber('/husky/cmd_vel_DWA', Twist, getVel)

if __name__ == '__main__':

    env = Env(True)
    agent = Agent(alpha=0.0001, beta=0.001, tau=0.001,
                    batch_size=12, n_actions=2)
    # agent.load_models()
    n_games = 200
    filename = 'Husky_ddpg_new22_tau_' + str(agent.alpha) + '_beta_' + \
                str(agent.beta) + '_' + str(n_games) + '_games'
    figure_file = 'plots/' + filename + '.png'

    best_score = 1200
    score_history = []
    action =[0,0]
    is_DWA = True

    for i in range(n_games):
        observation = env.reset()

        if (i%2) == 0:
            rand_no = i +3
        else:
            rand_no = i

        # print("Rand no:",rand_no)
        done = False
        score = 0
        t_stamp=0
        while not done:

            action_ddpg, cbam_out, cbam_in = agent.choose_action(observation)

            action[0] = vel_cmd.linear.x
            action[1] = vel_cmd.angular.z

            if i == rand_no:
                is_DWA = False
                print("--------- Training without DWA -------")
                action[0] = action_ddpg[0]
                action[1] = action_ddpg[1]

                action[0] = ((action[0]+1)/2)*0.65
                action[1] = action[1]*0.4

            print("Actions: ",action)

            observation_, reward, done, arrive = env.step(action)
            agent.remember(observation[0],observation[1], action, reward, observation_[0],observation_[1], done)
            agent.learn()

            score += reward
            observation = observation_
            t_stamp += 1

            print("Time stamp:",t_stamp)
            if is_DWA:
                if t_stamp==200:
                    print("----  DWA Episode time out!! ---")
                    break
            else:
                if t_stamp==250:
                    print("---- Episode time out!! ---")
                    break

        print("Time stamp:",t_stamp)

        score_history.append(score)
        avg_score = np.mean(score_history[-100:])

        if avg_score > best_score:
            best_score = avg_score
            agent.save_models()

        print('episode ', i, 'score %.1f' % score,
                'average score %.1f' % avg_score)

    # agent.save_models()
    x = [i+1 for i in range(n_games)]
    plot_learning_curve(x, score_history, figure_file)
