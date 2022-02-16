# [TERP: Terrain Elevation-based Robot Path Planning](https://arxiv.org/pdf/2109.05120.pdf)

We present a novel formulation to reliably navigate a ground robot in uneven outdoor environments. Our hybrid architecture combines intermediate output from a DRL network with attention with input elevation data to obtain a cost-map to perform local navigation tasks. We generate locally least-cost waypoints on the obtained cost-map and integrate our approach with an existing DRL method that computes dynamically feasible robot velocities.

A video summary and demonstrations of the system can be found [here](https://youtu.be/Q9yWLKJ1CdU)


https://user-images.githubusercontent.com/72771247/141846766-b9259b59-35bd-4b84-a072-a8895f884794.mp4


# Dependencies

This implementation builds on the Robotic Operating System (ROS-Melodic). 

* [Grid Map](https://github.com/ANYbotics/grid_map)(grid map library for mobile robots)
* [Octomap](http://wiki.ros.org/octomap)
* ARL Unity Simulator              

# Environment

## 1. Create a Conda Environment

```
conda create -n terp python=3.7
conda activate terp
conda install pytorch cudatoolkit -c pytorch
```

## 2. Modify Grid Map launch file

  Goto ```grid_map/grid_map_demos/launch/``` and modify the ```octomap_to_gridmap_demo.launch``` file as follows,

```
  <node pkg="octomap_server" type="octomap_server_node" name="octomap_server">
  <param name="resolution" value="0.5" />

  <param name="frame_id" type="string" value="husky/base" />
  <param name="base_frame_id" type="string" value="husky/base"/>
  <!-- maximum range to integrate (speedup!) -->
  <param name="sensor_model/max_range" value="10" />
  <param name="latch" value="false"/>
  <!-- data source to integrate (PointCloud2) -->
  <remap from="cloud_in" to="/husky/lidar_points" />

  </node>
```

## 3. Installing TERP
To build from source, clone the latest version from this repository into your catkin workspace and compile the package using,

```
cd catkin_ws/src
git clone https://github.com/kasunweerkoon/terp.git
cd ../
catkin_make
```

# Get Started

## 1. Launch ARL Unity Simulator with Husky

```
roslaunch arl_unity_ros_ground simulator_with_husky.launch
```

## 2. Launch Octomap to Grid Elevation Map Converter

```
roslaunch grid_map_demos octomap_to_gridmap_demo.launch
```

## 3. Run the DWA Planner

```
rosrun terp dwa_pozyx_goals.py
```

## 4. For Testing
```
conda activate terp
rosrun terp local_waypoint_planner.py
```

## 5. For Training
```
conda activate terp
rosrun terp main_ddpg.py
```

# TERP Paper
Thank you for citing our [TERP](https://arxiv.org/pdf/2109.05120.pdf) paper if you use any of this code:

```
@misc{weerakoon2021terp,
      title={TERP: Reliable Planning in Uneven Outdoor Environments using Deep Reinforcement Learning}, 
      author={Kasun Weerakoon and Adarsh Jagan Sathyamoorthy and Utsav Patel and Dinesh Manocha},
      year={2021},
      eprint={2109.05120},
      archivePrefix={arXiv},
      primaryClass={cs.RO}
}
```
