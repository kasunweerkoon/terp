# Waypoint planner to test the trained model

import rospy
ros_path = '/opt/ros/kinetic/lib/python2.7/dist-packages'
import sys
if ros_path in sys.path:
    sys.path.remove(ros_path)
import cv2
sys.path.append('/opt/ros/kinetic/lib/python2.7/dist-packages')

print("CV2 Version:",cv2.__version__)

import torch as T
import numpy as np
from ddpg_torch import Agent
from utils import plot_learning_curve
from environment import Env
from PIL import Image
from sklearn.preprocessing import normalize
import matplotlib.pyplot as plt
import time
from geometry_msgs.msg import Twist, Point, Pose, PoseStamped
from skimage.graph import MCP
import math
global vel_cmd
vel_cmd =Twist()
# self.sub_cmd_vel = rospy.Subscriber('/husky/cmd_vel_DWA', Twist, self.getVel)


def getVel(data):
    vel_cmd.linear.x =data.linear.x
    vel_cmd.linear.y = 0
    vel_cmd.linear.z = 0
    vel_cmd.angular.x =0
    vel_cmd.angular.y =0
    vel_cmd.angular.z=data.angular.z
    # print('getting vel outputs')

vel_sub = rospy.Subscriber('/husky/cmd_vel_DWA', Twist, getVel)
pub_goal = rospy.Publisher('/target/position', Twist, queue_size=10)
goal_state_pub = rospy.Publisher('/env_state_pub', Twist, queue_size=10)
goal_topic = Twist()

agent = Agent(alpha=0.0001, beta=0.001, tau=0.01,
                batch_size=8, n_actions=2)

n_games =10
print("CUDA Available:",T.cuda.is_available())
filename = 'Husky_ddpg_test2' + str(agent.alpha) + '_beta_' + \
            str(agent.beta) + '_' + str(n_games) + '_games'
figure_file = 'plots/' + filename + '.png'

env = Env(False)
agent.load_models()
score_history = []

def reset_func():
    vel_cmd2 = Twist()
    vel_cmd2.linear.x = 100
    goal_state_pub.publish(vel_cmd2) #publish a new goal to DWA

    time.sleep(5)

goal_in = [8.5, 0]
without_attention = False

observation = env.reset(goal_in)
score_history = []
t_stamp=0
action =[0,0]
i=0
arrive =False
waypoint_arrived =False
grid_arrived =False
score = 0
Tot_elevation_diff =0
Tot_goal_heading_cost =0
Total_dist_travelled = 0
Tot_pose_cost=0

action_linear = []
action_angular = []

grid_no =1
current_x0,current_y0,current_z0, relative_theta_deg0,cost_pose0 = env.TGC_calculator()
while not arrive:

    action_ddpg, cbam_out, cbam_in = agent.choose_action(observation)

    action[0] = vel_cmd.linear.x
    action[1] = vel_cmd.angular.z
    action_linear.append(action[0])
    action_angular.append(action[1])

    if t_stamp ==0 or grid_arrived:
        time.sleep(0.5)

        grid_waypoint_coords_wrt_odom = env.waypoint_planner(cbam_out,observation[0],without_attention) #need to get goal wrt world

        grid_no_end = np.shape(grid_waypoint_coords_wrt_odom)[0]
        # print("grid length",grid_no_end)
        # print("-------Start navigating to a new waypoint-----")
        grid_no = 1
        # print("grid goal in to r theta cal:",grid_waypoint_coords_wrt_odom[grid_no])
        # grid_waypoint_coords_wrt_odom = env.waypoint_planner(cbam_out,observation[0])
        grid_waypoints = env.angle_dist_calculator(grid_waypoint_coords_wrt_odom[grid_no][0],grid_waypoint_coords_wrt_odom[grid_no][1])
        # print("Grid goal x",grid_waypoint_coords_wrt_odom[grid_no][0])
        # print("Grid goal y",grid_waypoint_coords_wrt_odom[grid_no][1])
        goal_topic.linear.x = grid_waypoints[0] #random.uniform(0, 12)
        goal_topic.linear.y = grid_waypoints[1]
        # print("Grid goal r, theta:",grid_waypoints)
        current_goal = [grid_waypoint_coords_wrt_odom[grid_no][0],grid_waypoint_coords_wrt_odom[grid_no][1]]
        waypoint_goal =  [grid_waypoint_coords_wrt_odom[grid_no_end-1][0],grid_waypoint_coords_wrt_odom[grid_no_end-1][1]]
        print("-------- New local grid goal is assigned -----------------")
        print("New goal: ",(goal_topic.linear.x,goal_topic.linear.y))
        pub_goal.publish(goal_topic)
        # reset_func()

        action[0] = vel_cmd.linear.x
        action[1] = vel_cmd.angular.z
        i +=1
        grid_no +=1


    observation_, reward, done, arrive, waypoint_arrived,grid_arrived,costmap, mask = env.step(action,current_goal,waypoint_goal,grid_arrived,waypoint_arrived)
    current_x,current_y,current_z, relative_theta_deg, cost_pose = env.TGC_calculator()

    elevation_diff = abs(current_z - current_z0)
    distance_diff = math.hypot(current_x - current_x0, current_y - current_y0)
    # print("z , thera diff :",[current_z, relative_theta_deg])
    # print("distance_diff:",distance_diff)


    Tot_elevation_diff += elevation_diff
    Tot_goal_heading_cost += relative_theta_deg
    Total_dist_travelled += distance_diff
    Tot_pose_cost += cost_pose
    score += reward
    observation = observation_
    current_z0 = current_z
    current_x0 = current_x
    current_y0 = current_y
    t_stamp += 1

# score_history.append(score)
# avg_score = np.mean(score_history[-100:])
Tot_TGC = Tot_elevation_diff+ Tot_goal_heading_cost
print("Total distance travelled:", Total_dist_travelled)
print("Tot_goal_heading cost:", Tot_goal_heading_cost)
print("Tot_pose cost:", Tot_pose_cost)
print("Tot_elevation cost:", Tot_elevation_diff)
print("Total TGC cost:", Tot_TGC)
