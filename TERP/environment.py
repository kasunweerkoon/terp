#!/usr/bin/env python3
import os

import subprocess
import time
import rospy
import numpy as np
import math
from math import pi
import random
from scipy.interpolate import griddata
from skimage.graph import MCP
import skimage.transform as st

from skimage import exposure

from geometry_msgs.msg import Twist, Point, Pose, PoseStamped
from sensor_msgs.msg import LaserScan
from nav_msgs.msg import Odometry, OccupancyGrid
from std_msgs.msg import String
from std_srvs.srv import Empty
from robot_localization.srv import SetPose


class Env():
    def __init__(self, is_training):
        rospy.init_node('TERP_environment', anonymous=True)
        self.position = Pose()
        self.goal_position = Pose()
        self.goal_topic = Twist()
        self.vel_cmd = Twist()
        self.vel_cmd.linear.x =0
        self.vel_cmd.angular.z=0
        #
        # self.goal_position.position.x = 20.
        # self.goal_position.position.y = 0.

        self.goal_topic.linear.x = self.goal_position.position.x# this will be r
        self.goal_topic.linear.y = self.goal_position.position.y # this will be theta

        self.odom_topic_name = '/unity_command/ground_truth/husky/odom'
        self.husky_vel_topic_name = '/husky/cmd_vel'
        self.grid_map_topic_name = '/grid_map_visualization/elevation_grid'
        self.scan_topic_name = '/husky/scan'

        self.pub_cmd_vel = rospy.Publisher(self.husky_vel_topic_name, Twist, queue_size=10)

        self.sub_odom = rospy.Subscriber(self.odom_topic_name, Odometry, self.getOdometry)

        self.reset_elevation = rospy.ServiceProxy('/octomap_server/reset', Empty)


        self.past_distance = 0.
        if is_training:
            self.threshold_arrive = 1
        else:
            self.threshold_arrive = 1

    def getGoalDistace(self):
        goal_distance = math.hypot(self.goal_position.position.x - self.position.x, self.goal_position.position.y - self.position.y)
        self.end_goal_wrt_robot_odom = [self.goal_position.position.x,self.goal_position.position.y]
        self.past_distance = goal_distance

        return goal_distance

    def getOdometry(self, odom):
        # print("--------getting odom data-----------")
        self.position = odom.pose.pose.position
        orientation = odom.pose.pose.orientation
        q_x, q_y, q_z, q_w = orientation.x, orientation.y, orientation.z, orientation.w
        roll, pitch, self.yaw_rad = self.euler_from_quaternion(q_x, q_y, q_z, q_w) # roll, pitch , yaw in radians
        yaw = round(math.degrees(self.yaw_rad))
        # yaw = round(math.degrees(math.atan2(2 * (q_x * q_y + q_w * q_z), 1 - 2 * (q_y * q_y + q_z * q_z))))

        if yaw >= 0:
             yaw = yaw
        else:
             yaw = yaw + 360

        rel_dis_x = round(self.goal_position.position.x - self.position.x, 1)
        rel_dis_y = round(self.goal_position.position.y - self.position.y, 1)

        # Calculate the angle between robot and target
        if rel_dis_x > 0 and rel_dis_y > 0:
            theta = math.atan(rel_dis_y / rel_dis_x)
        elif rel_dis_x > 0 and rel_dis_y < 0:
            theta = 2 * math.pi + math.atan(rel_dis_y / rel_dis_x)
        elif rel_dis_x < 0 and rel_dis_y < 0:
            theta = math.pi + math.atan(rel_dis_y / rel_dis_x)
        elif rel_dis_x < 0 and rel_dis_y > 0:
            theta = math.pi + math.atan(rel_dis_y / rel_dis_x)
        elif rel_dis_x == 0 and rel_dis_y > 0:
            theta = 1 / 2 * math.pi
        elif rel_dis_x == 0 and rel_dis_y < 0:
            theta = 3 / 2 * math.pi
        elif rel_dis_y == 0 and rel_dis_x > 0:
            theta = 0
        else:
            theta = math.pi
        rel_theta = round(math.degrees(theta), 2)

        diff_angle = abs(rel_theta - yaw)

        if diff_angle <= 180:
            diff_angle = round(diff_angle, 2) #need ()-) sign ?????
        else:
            diff_angle = round(360 - diff_angle, 2)

        self.rel_theta = rel_theta
        self.pitch = pitch
        self.roll = roll
        self.yaw = yaw
        self.diff_angle = diff_angle
        # print(self.rel_theta)

    def getElevationMap(self, data):
        # print('---- getting elevation data---')
        map_data = np.array(data.data)
        map_resolution = data.info.resolution
        map_width = data.info.width
        map_height = data.info.height
        map2d = np.reshape(map_data, (map_height, map_width))

        if map_width != 40 or map_height != 40:

            map2d = st.resize(map2d, (40, 40))

        map_width = 40
        map_height = 40

        result = np.where(map2d == -1)
        result2 = np.where(map2d != -1)

        values = map2d[result2]
        # print(values)
        points = np.vstack((result2[0], result2[1])).T

        grid_x, grid_y = np.meshgrid(np.arange(0, map_width, 1), np.arange(0, map_height, 1))

        # Nearest neighbour interpolated elevation map
        interpolated_grid = griddata(points, values, (grid_x, grid_y),'nearest')
        self.interpolated_map = interpolated_grid.astype('float64')
        # print(np.size(self.interpolated_map))

        vgrad = np.gradient(self.interpolated_map)
        # Gradient map of the interpolated elevation map
        self.elevation_gradient = np.sqrt(vgrad[0]**2 + vgrad[1]**2)

        # elevation gradient array of robot's heading direction
        heading_coord = np.arange((map_height/2),map_height)

        middle = int(map_height/2)
        endll = (map_height+1)
        # print(middle)
        # print(endll)
        self.heading_gradient = self.elevation_gradient[middle:endll,int((map_width/2)-1)]
        # print(self.heading_gradient)


    def getState(self,scan):

        scan_range = []
        min_range = 0.6
        max_range = 8
        heading_elevation =self.heading_gradient
        yaw = self.yaw
        rel_theta = self.rel_theta
        diff_angle = self.diff_angle
        max_robot_pose_angle = 30
        done = False
        arrive = False
        velocity = self.vel_cmd

        pitch_deg = abs(round(math.degrees(self.pitch)))
        roll_deg = abs(round(math.degrees(self.roll)))

        for i in range(len(scan.ranges)):
            if scan.ranges[i] == float('Inf'):
                scan_range.append(max_range)
            elif np.isnan(scan.ranges[i]):
                scan_range.append(0)
            else:
                scan_range.append(scan.ranges[i])

        if velocity.linear.x < 0 or velocity.linear.x> 1:
            if -0.3 > velocity.angular.z or velocity.angular.z > 0.3 :
                print('----- Velocity limits exceeded !!  -----')
                done =True

        if abs(self.position.z) > 12:
            print('----- Spawn location error !!  -----')
            done =True

        if min_range > min(scan_range) > 0:
            print('------- obstacle around !!---------')
            done = True

        if pitch_deg > max_robot_pose_angle or roll_deg > max_robot_pose_angle:
            print('------- Unsafe orientation !!  --------')
            # print('pitch angle: %f'%pitch_deg)
            # print('roll angle: %f'%roll_deg)
            done = True


        current_distance = math.hypot(self.goal_position.position.x - self.position.x, self.goal_position.position.y - self.position.y)
        if current_distance <= self.threshold_arrive:
            print('------  Goal reached..!!   -----')
            done = True
            arrive = True

        return heading_elevation, current_distance, rel_theta, pitch_deg, roll_deg,diff_angle, done, arrive

    def setReward(self, done, arrive):

        reward_pose = 45*(math.cos(self.roll)**2 + math.cos(self.pitch)**2)

        reward_heading_angle = -abs(self.diff_angle)*0.4


        current_distance = math.hypot(self.goal_position.position.x - self.position.x, self.goal_position.position.y - self.position.y)
        distance_rate = (self.past_distance - current_distance)

        reward_goal_distance = 6000 *distance_rate

        self.past_distance = current_distance
        w_elevation = np.arange(20,0,-1)/20
        reward_elevation_cost = -np.sum(w_elevation*self.heading_gradient)*2.2

        # Overall reward
        reward = (reward_goal_distance + reward_pose + reward_heading_angle + reward_elevation_cost)/5
        # print("Total reward:",reward)

        if done:
            reward = -500


        if arrive:
            reward = 500
            arrive = False

        return reward

    def step(self, action, grid_goal,waypoint_goal,grid_arrived,waypoint_arrived):

        linear_vel = action[0]
        ang_vel = action[1]
        goal_reached_thresh = 0.5

        self.vel_cmd.linear.x = action[0] #(linear_vel+1.5)/2
        self.vel_cmd.angular.z = action[1] #ang_vel*0.25

        self.pub_cmd_vel.publish(self.vel_cmd)

        waypoint_goalx = waypoint_goal[0]
        waypoint_goaly = waypoint_goal[1]

        waypoint_goal_distance = math.hypot(waypoint_goalx - self.position.x, waypoint_goaly - self.position.y)
        # print("Robot's xy:",[self.position.x,self.position.y])
        #print("Current distance to the waypoint:",waypoint_goal_distance)

        if waypoint_goal_distance <= goal_reached_thresh:
            waypoint_arrived = True
            print("------- Waypoint reached -------")
        else:
            waypoint_arrived = False

        grid_goalx = grid_goal[0]
        grid_goaly = grid_goal[1]

        grid_goal_distance = math.hypot(grid_goalx - self.position.x, grid_goaly - self.position.y)
        # print("Current distance to the grid goal:",grid_goal_distance)
        if grid_goal_distance <= goal_reached_thresh:
            grid_arrived = True
            print("---- Local grid goal reached -----")
        else:
            grid_arrived = False


        data = None
        while data is None:
            try:

                # time.sleep(2.5)
                data_elevation = rospy.wait_for_message(self.grid_map_topic_name, OccupancyGrid, timeout=5)
                # print("new ellevation data")
                data = rospy.wait_for_message(self.scan_topic_name, LaserScan, timeout=5)
                data_odom = rospy.wait_for_message(self.odom_topic_name, Odometry, timeout=5)

                self.getOdometry(data_odom)
                self.getElevationMap(data_elevation)
            except:
                pass

        heading_elevation, current_distance, rel_theta, pitch_deg, roll_deg, diff_angle, done, arrive = self.getState(data)
        # print("heading elevation",heading_elevation)
        state_input_1D = [current_distance,rel_theta, pitch_deg, roll_deg,diff_angle]

        for el in heading_elevation:
            state_input_1D.append(el)

        #combine 2D elevation map and 1D state inputs to a single array
        state = [self.interpolated_map,state_input_1D]

        reward = self.setReward(done, arrive)
        # print("Goal distance: ",self.goal_distance)
        return np.asarray(state), reward, done, arrive, waypoint_arrived,grid_arrived,self.costmap_normalized, self.attention_mask


    def reset(self,goal_in):

        self.goal_topic.linear.x = goal_in[0] #random.uniform(0, 12)
        self.goal_topic.linear.y = goal_in[1]#random.uniform(-180, 180)

        radius = self.goal_topic.linear.x # this will be r
        theta = self.goal_topic.linear.y * 0.0174533# this will be theta
        goalX_rob = radius * math.cos(theta)
        goalY_rob = radius * math.sin(theta)

        self.end_goal_wrt_robot_init = [goalX_rob,goalY_rob]

        self.goal_position.position.x =  self.position.x + goalX_rob*math.cos(self.yaw_rad) - goalY_rob*math.sin(self.yaw_rad)
        self.goal_position.position.y = self.position.y + goalX_rob*math.sin(self.yaw_rad) + goalY_rob*math.cos(self.yaw_rad)

        self.end_goal_wrt_robot_odom = [self.goal_position.position.x,self.goal_position.position.y]

        self.goal_distance = self.getGoalDistace()
        print("Goal distance:", self.goal_distance)

        data = None
        while data is None:
            try:
                # time.sleep(0.5)
                # self.reset_elevation()
                # time.sleep(2)
                data_elevation = rospy.wait_for_message(self.grid_map_topic_name, OccupancyGrid, timeout=5)
                # print("new elevation data")
                data = rospy.wait_for_message(self.scan_topic_name, LaserScan, timeout=5)
                data_odom = rospy.wait_for_message(self.odom_topic_name, Odometry, timeout=5)

                self.getOdometry(data_odom)
                self.getElevationMap(data_elevation)

            except:
                pass

        heading_elevation, current_distance, rel_theta, pitch_deg, roll_deg, diff_angle,done, arrive = self.getState(data)
        state_input_1D = [current_distance,rel_theta, pitch_deg, roll_deg,diff_angle]

        for el in heading_elevation:
            state_input_1D.append(el)


        #combine 2D elevation map and 1D state inputs to a single array
        state = [self.interpolated_map,state_input_1D]

        return np.asarray(state) #, end_goal_wrt_robot_odom

    def euler_from_quaternion(self, x, y, z, w):
        """
        Convert a quaternion into euler angles (roll, pitch, yaw)
        roll is rotation around x in radians (counterclockwise)
        pitch is rotation around y in radians  (counterclockwise)
        yaw is rotation around z in radians  (counterclockwise)
        """
        t0 = +2.0 * (w * x + y * z)
        t1 = +1.0 - 2.0 * (x * x + y * y)
        roll_x = math.atan2(t0, t1)

        t2 = +2.0 * (w * y - z * x)
        t2 = +1.0 if t2 > +1.0 else t2
        t2 = -1.0 if t2 < -1.0 else t2
        pitch_y = math.asin(t2)

        t3 = +2.0 * (w * z + x * y)
        t4 = +1.0 - 2.0 * (y * y + z * z)
        yaw_z = math.atan2(t3, t4)

        return roll_x, pitch_y, yaw_z # in radians

    def angle_dist_calculator(self,local_goal_x,local_goal_y):

        data_odom = rospy.wait_for_message(self.odom_topic_name, Odometry, timeout=5)

        self.getOdometry(data_odom)

        rel_dis_x = local_goal_x  - self.position.x
        rel_dis_y = local_goal_y - self.position.y

        x_robot = rel_dis_x * math.cos(self.yaw_rad) + rel_dis_y*math.sin(self.yaw_rad)
        y_robot = -rel_dis_x * math.sin(self.yaw_rad) + rel_dis_y*math.cos(self.yaw_rad)

        if x_robot > 0:
            theta = math.atan(y_robot/x_robot)
        elif x_robot < 0 and y_robot > 0:
            theta = (math.pi - abs(math.atan(y_robot/x_robot)))
        elif x_robot < 0 and y_robot < 0:
            theta = (-math.pi + abs(math.atan(y_robot/x_robot)))

        theta_deg = math.degrees(theta)

        # print("yaw angle:",math.degrees(self.yaw_rad))
        dist = np.sqrt(x_robot**2+y_robot**2)

        return [dist, theta_deg]


    def searchBoundary(self,m,n,r,angle_diff):
    #     # m, n coordinates of the center
    #     # r -  radius to the boundary circle
        print("Angle diff:",angle_diff)
        points_start =[]
        points_end=[]
        for i in range(m-r,m+r+1):
            for j in range(n-r,n+r+1):
                if (i == m-r):
                    # sum += a[i][j]
                    coord = [i,j]
                    points_end.append(coord)
                elif (i == m+r):
                    # sum += a[i][j]
                    coord = [i,j]
                    points_end.append(coord)
                elif (j == n-r):
                    # sum += a[i][j]
                    coord = [i,j]
                    points_end.append(coord)
                elif (j == n+r):
                    # sum += a[i][j]
                    coord = [i,j]
                    points_end.append(coord)

        points_end = np.array(points_end)

        if -45<=angle_diff<=45:
            print("-45<angle_diff<45")
            cols = np.where(points_end[:,0]==m+r)
            points_end_segmented = points_end[cols]
          # print(points_end_segmented)
        elif 45<angle_diff<135:
            print("45<angle_diff<135")
            rows = np.where(points_end[:,1]==n+r)
            # print(points_end[rows])
            points_end_segmented = points_end[rows]
          # print(points_end_segmented)
        elif angle_diff<-135 or angle_diff>135:
            print("angle_diff<-135 or angle_diff>135")
            cols = np.where(points_end[:,0]==m-r)
            points_end_segmented = points_end[cols]
            # print(points_end_segmented)

        elif -135<=angle_diff<=-45:
            rows = np.where(points_end[:,1]==n-r)
            print("-135<angle_diff<-45")
            points_end_segmented = points_end[rows]
            # print(points_end_segmented)


        for k in range(np.shape(points_end_segmented)[0]):
          points_start.append([m,n])

        return points_start, points_end_segmented


    def get_local_waypoints(self,costmap,end_goal_wrt_robot_odom,end_goal_wrt_robot_init):

        waypoint_as_goal_threshold = 10.1
        scaling_factor = 0.5
        current_distance = self.getGoalDistace()

        data_odom = rospy.wait_for_message(self.odom_topic_name, Odometry, timeout=5)
        self.getOdometry(data_odom)

        if current_distance < waypoint_as_goal_threshold:

            print("----- Rerouting to the current waypoint-----------")

            cmap = costmap.astype(dtype=int)
            # print("Cmap:",cmap)
            starts = [[21,19]]

            rel_dis_x = end_goal_wrt_robot_odom[0]  - self.position.x
            rel_dis_y = end_goal_wrt_robot_odom[1] - self.position.y

            x_robot = rel_dis_x * math.cos(self.yaw_rad) + rel_dis_y*math.sin(self.yaw_rad)
            y_robot = -rel_dis_x * math.sin(self.yaw_rad) + rel_dis_y*math.cos(self.yaw_rad)

            end_x = (x_robot / scaling_factor)+19
            end_y = (y_robot / scaling_factor)+19

            # print("end goal xy wrt odom:",end_goal_wrt_robot_odom)
            # print("robot position wrt odom:",[self.position.x,self.position.y])
            # print("robot's yaw:",math.degrees(self.yaw_rad))
            ends = [[round(end_x),round(end_y)]]
            print("End goal in the grid :",ends)
            # Pass full set of start and end points to `MCP.find_costs`

            m = MCP(cmap)
            cost_array, tracebacks_array = m.find_costs(starts, ends)

            # Transpose `ends` so can be used to index in NumPy
            ends_idx = tuple(np.asarray(ends).T.tolist())
            costs = cost_array[ends_idx]

            # Compute exact minimum cost path to each endpoint
            tracebacks = [m.traceback(end) for end in ends]
            # print("Costs: ",costs)

            min_cost_idx = np.where(costs == costs.min())
            num_of_mins = np.size(min_cost_idx[0])
            # print("Num of mins:",num_of_mins)

            waypoint_array_coord= ends
            waypoint_array_paths= tracebacks

            x_center = 19
            y_center = 19

            goal_positionx = end_goal_wrt_robot_odom[0] # wrt robots odom
            goal_positiony = end_goal_wrt_robot_odom[1]

            waypoint_coords =  scaling_factor * (np.array(waypoint_array_coord)- [x_center,y_center])
            # print("Min cost waypoint coordinates wrt robot",waypoint_coords)

            dist_to_goal=[]
            waypoint_coords_wrt_odom =[]
            for i in range(num_of_mins):
                waypoint_coords_x = self.position.x + waypoint_coords[i][0]*math.cos(self.yaw_rad) - waypoint_coords[i][1]*math.sin(self.yaw_rad)
                waypoint_coords_y = self.position.y + waypoint_coords[i][0]*math.sin(self.yaw_rad) + waypoint_coords[i][1]*math.cos(self.yaw_rad)
                waypoint_coords_wrt_odom.append([waypoint_coords_x,waypoint_coords_y])
                dist_val = math.hypot(goal_positionx - waypoint_coords_wrt_odom[i][0], goal_positiony - waypoint_coords_wrt_odom[i][1])
                dist_to_goal.append(dist_val)


            local_gridpoints_to_waypoint = waypoint_array_paths[0]
            # print("Waypoint array path",local_gridpoints_to_waypoint)

            path_length=np.shape(local_gridpoints_to_waypoint)[0]
            # print(path_length)
            grid_waypoint_coords_wrt_odom =[] #pixel by pixel coordinates to the local waypoint
            r_theta_grid_goals =[]
            for i in range(path_length):
                grid_waypoint_coords_x = self.position.x + (local_gridpoints_to_waypoint[i][0]-x_center)*math.cos(self.yaw_rad) - (local_gridpoints_to_waypoint[i][1]-y_center)*math.sin(self.yaw_rad)
                grid_waypoint_coords_y = self.position.y + (local_gridpoints_to_waypoint[i][0]-x_center)*math.sin(self.yaw_rad) + (local_gridpoints_to_waypoint[i][1]-y_center)*math.cos(self.yaw_rad)
                grid_waypoint_coords_wrt_odom.append([grid_waypoint_coords_x,grid_waypoint_coords_y])
                r_theta_grid_goal_val = self.angle_dist_calculator(grid_waypoint_coords_wrt_odom[i][0],grid_waypoint_coords_wrt_odom[i][1])
                r_theta_grid_goals.append(r_theta_grid_goal_val)


        else:
            print("----------- Goal is out of sensing range ----")
            points_start, points_end = self.searchBoundary(19,19,8,self.diff_angle)

            cmap = costmap.astype(dtype=int)
            # print("Cmap:",cmap)
            starts = points_start
            ends = points_end


            m = MCP(cmap)
            cost_array, tracebacks_array = m.find_costs(starts, ends)

            # Transpose `ends` so can be used to index in NumPy
            ends_idx = tuple(np.asarray(ends).T.tolist())
            costs = cost_array[ends_idx]

            # Compute exact minimum cost path to each endpoint
            tracebacks = [m.traceback(end) for end in ends]
            # print("Costs: ",costs)

            min_cost_idx = np.where(costs == costs.min())
            num_of_mins = np.size(min_cost_idx[0])

            waypoint_array_coord=[]
            waypoint_array_paths=[]
            for i in range(num_of_mins):
              waypoint_array_coord.append(points_end[min_cost_idx[0][i]])
              waypoint_array_paths.append(tracebacks[:][:][min_cost_idx[0][i]])
            x_center = 19
            y_center = 19

            goal_positionx = end_goal_wrt_robot_odom[0] # wrt robots odom
            goal_positiony = end_goal_wrt_robot_odom[1]

            waypoint_coords =  scaling_factor * (np.array(waypoint_array_coord)- [x_center,y_center])
            # print("Min cost waypoint coordinates wrt robot",waypoint_coords)

            dist_to_goal=[]
            waypoint_coords_wrt_odom =[]
            for i in range(num_of_mins):
                waypoint_coords_x = self.position.x + waypoint_coords[i][0]*math.cos(self.yaw_rad) - waypoint_coords[i][1]*math.sin(self.yaw_rad)
                waypoint_coords_y = self.position.y + waypoint_coords[i][0]*math.sin(self.yaw_rad) + waypoint_coords[i][1]*math.cos(self.yaw_rad)
                waypoint_coords_wrt_odom.append([waypoint_coords_x,waypoint_coords_y])
                dist_val = math.hypot(goal_positionx - waypoint_coords_wrt_odom[i][0], goal_positiony - waypoint_coords_wrt_odom[i][1])
                dist_to_goal.append(dist_val)

            dist_to_goal = np.array(dist_to_goal)
            # print(dist_to_goal)

            min_dist_idx = np.where(dist_to_goal == dist_to_goal.min())
            num_of_dist_mins = np.size(min_dist_idx[0])

            # print(int(min_dist_idx[0]))
            local_gridpoints_to_waypoint = waypoint_array_paths[int(min_dist_idx[0])]
            print(local_gridpoints_to_waypoint[0])

            path_length=np.shape(local_gridpoints_to_waypoint)[0]

            grid_waypoint_coords_wrt_odom =[] #pixel by pixel coordinates to the local waypoint
            r_theta_grid_goals =[]
            for i in range(path_length):
                grid_waypoint_coords_x = self.position.x + local_gridpoints_to_waypoint[i][0]*math.cos(self.yaw_rad) - local_gridpoints_to_waypoint[i][1]*math.sin(self.yaw_rad)
                grid_waypoint_coords_y = self.position.y + local_gridpoints_to_waypoint[i][0]*math.sin(self.yaw_rad) + local_gridpoints_to_waypoint[i][1]*math.cos(self.yaw_rad)
                grid_waypoint_coords_wrt_odom.append([grid_waypoint_coords_x,grid_waypoint_coords_y])
                # r_theta_grid_goal_val = self.angle_dist_calculator(grid_waypoint_coords_wrt_odom[i][0],grid_waypoint_coords_wrt_odom[i][1])
                # r_theta_grid_goals.append(r_theta_grid_goal_val)

            local_waypoint = waypoint_coords_wrt_odom[int(min_dist_idx[0])]


        return grid_waypoint_coords_wrt_odom

    def arr_normalize(self,x):
        xmax, xmin = x.max(), x.min()
        x = (x - xmin)/(xmax - xmin)
        # arr = arr - arr.mean(axis=0)
        # arr = arr / np.abs(arr).max(axis=0)
        return x

    def neighbors(self,radius, rowNumber, columnNumber):
         return [[a[i][j] if  i >= 0 and i < len(a) and j >= 0 and j < len(a[0]) else 0
                    for j in range(columnNumber-1-radius, columnNumber+radius)]
                        for i in range(rowNumber-1-radius, rowNumber+radius)]

    # def min_cost_check(costmap):
    def threshold_up(self,map,threshold_val,assign_val):
          threshold_idx = np.where(map >= threshold_val)
          map[threshold_idx]= assign_val
          return map

    def waypoint_planner(self,cbam_out, elevation_map,without_attention):
        # print(np.shape(cbam_out))
        # np.save("elevation_map2.npy",elevation_map)
        cbam_out_sum = cbam_out.squeeze(0)[0]
        # # plt.subplot(1,6,1)
        cbam_norm = np.array(self.arr_normalize(cbam_out.squeeze(0)[0]))
        # cbam_in_sum = cbam_in.squeeze(0)[0]
        for i in range(7):
           cbam_norm2 = np.array(self.arr_normalize(cbam_out.squeeze(0)[i+1]))
           cbam_out_sum2 = cbam_out.squeeze(0)[i+1]
           # cbam_in_sum2 = cbam_in.squeeze(0)[i+1]
           cbam_out_sum= cbam_out_sum + cbam_out_sum2
           # cbam_in_sum =cbam_in_sum+cbam_in_sum2
           cbam_norm = cbam_norm + cbam_norm2

        attention_mask = cbam_out_sum
        attention_mask_normalized = self.arr_normalize(attention_mask)

        # print('Max of the Elevation map : ', max(map(max, elevation_map)))
        # print('Min of the Elevation map: ',min(map(min, elevation_map)))

        costmap = np.multiply(elevation_map,attention_mask)
        self.attention_mask= attention_mask
        # costmap = self.arr_normalize(elevation_map + attention_mask)*50
        self.costmap_normalized = self.arr_normalize(costmap)*100    #exposure.equalize_hist(costmap)*100 #
        # print('Max of the Cost map normalized: ', max(map(max, self.costmap_normalized )))
        # print('Min of the Cost map normalized: ',min(map(min, self.costmap_normalized )))

        threshold_val = 80
        assign_val = 1000
        self.costmap_normalized = self.threshold_up(self.costmap_normalized,threshold_val,assign_val)
        # min_cost_check(costmap)

        if without_attention:
            self.costmap_normalized = elevation_map

        waypoints = self.get_local_waypoints(self.costmap_normalized,self.end_goal_wrt_robot_odom,self.end_goal_wrt_robot_init)

        # waypoints = [[5,0],[3,90],[5,-90]]
        return waypoints


    def TGC_calculator(self):
        # Terrain gradient cost
        end_goal_global_xy = self.end_goal_wrt_robot_odom
        [end_goal_dist, relative_theta_deg] = self.angle_dist_calculator(end_goal_global_xy[0],end_goal_global_xy[1])
        current_z = self.position.z
        current_x = self.position.x
        current_y = self.position.x
        cost_pose = 5 *(math.cos(self.roll)**2 + math.cos(self.pitch)**2)

        return current_x,current_y,current_z, math.radians(relative_theta_deg),cost_pose
