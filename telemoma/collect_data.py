import os
import time
import cv2
import copy
import rospy
import numpy as np
import h5py
# from tiago_gym.tiago.tiago_gym import TiagoGym    
from telemoma.robot_interface.franka.franka_gym import FrankaGym
from importlib.machinery import SourceFileLoader
# from tiago_gym.utils.camera_utils import Camera, flip_img, img_processing, depth_processing

from telemoma.human_interface.teleop_policy import TeleopPolicy
# from telemoma.configs.only_keyboard import teleop_config
# from telemoma.configs.only_vr import teleop_config

SINGLE_HAND=True
def collect_trajectory(render=False, teleop_config=None):

    # agentview_left = Camera(img_topic="/agentview_left/color/image_raw",
    #                           depth_topic="/agentview_left/aligned_depth_to_color/image_raw",)
    
    # # flip the camera stream for the right hand side
    # agentview_right = Camera(img_topic="/agentview_right/color/image_raw",
    #                           depth_topic="/agentview_right/aligned_depth_to_color/image_raw",
    #                           img_post_proc_func=lambda x: flip_img(img_processing(x)),
    #                           depth_post_proc_func=lambda x: flip_img(depth_processing(x)))
            
    env = FrankaGym(
            frequency=10,
            head_policy=None,
            base_enabled=False,
            torso_enabled=False,
            arm_enabled=True,
            camera_enabled=True
        )
    
    # rospy.on_shutdown(shutdown_helper)
    obs = env.reset(reset_arms=True)
    input('press enter to start:')

    teleop = TeleopPolicy(teleop_config)
    teleop.start()
    def shutdown_helper():
        teleop.stop()

    trajectory = {'obs':{}, 'actions': [], 'rewards': [], 'dones': []}
    for k in obs.keys():
        trajectory['obs'][k] = []
    
    start_time = time.time()
    while not rospy.is_shutdown():
        action = teleop.get_action(obs)
        buttons = action.extra['buttons'] if 'buttons' in action.extra else {}
        # n_obs, reward, done, trunc, info = env.step(action)
        n_obs, reward, done, info = env.step(action)
        done = buttons.get('A', False)
        
        trajectory['dones'].append(done)

        trajectory['actions'].append(action['right']) 
        
        trajectory['rewards'].append(reward)

        
        for k in obs.keys():  

            trajectory['obs'][k].append(obs[k])

        if done:
            break
        if buttons.get('B', False):
            teleop.stop()
            return None
        
        obs = copy.deepcopy(n_obs)

        if render:
            color_img = obs['image']/255
            cv2.imshow('cam', color_img)
            cv2.waitKey(1)
    teleop.stop()
    print('Total time taken:', time.time()-start_time)

    return trajectory

if __name__=='__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--save_dir", default=None, type=str, help="path to save file")
    parser.add_argument("--render", action="store_true", help="pass flag to render environment while data generation")
    parser.add_argument('--teleop_config', type=str, default ="configs/only_vr.py" ,help='Path to the teleop config to use.')
    args = parser.parse_args()
    teleop_config = SourceFileLoader('conf', args.teleop_config).load_module().teleop_config


    data = collect_trajectory(args.render, teleop_config)
    

    if (args.save_dir is not None) and (data is not None):
        os.makedirs(args.save_dir, exist_ok=True)
        demo_id = len(os.listdir(args.save_dir))
        save_path = os.path.join(args.save_dir, f'demo_{demo_id}.h5')

        f = h5py.File(save_path, 'w')

        for k in data.keys():
            print(f'Saving {k}')
            if k == 'obs':
                obs_grp = f.create_group('obs')

                for obs_k in data['obs'].keys():
                    print(f'Saving {obs_k}')
                    obs_grp.create_dataset(obs_k, data=data['obs'][obs_k])
            else:
                f.create_dataset(k, data=data[k])