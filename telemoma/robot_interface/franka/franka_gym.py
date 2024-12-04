import gym
import time
import sys
# sys.path.append("home/huihanliu/deoxys_control/deoxys")
import numpy as np
from collections import OrderedDict
from deoxys.franka_interface import FrankaInterface
from deoxys.utils.config_utils import get_default_controller_config
from deoxys.utils.input_utils import input2action

from telemoma.utils.camera_utils import RealSenseCamera
from telemoma.utils.general_utils import AttrDict
from telemoma.utils.transformations import rmat_to_quat

class FrankaGym(gym.Env):

    def __init__(self,
                    frequency=10,
                    head_policy=None,
                    base_enabled=True,
                    torso_enabled=False,
                    arm_enabled=True,
                    camera_enabled=False):
        
        super(FrankaGym).__init__()

        self.frequency = frequency
        self.arm_enabled = arm_enabled

        config_path = "/home/huihanliu/deoxys_control/deoxys/config/charmander.yml"
        self.robot_interface = FrankaInterface(config_path, use_visualizer=False)

        self.controller_type = "OSC_POSE"
        # self.controller_type = "OSC_YAW" #"OSC_POSE"
        self.controller_cfg = get_default_controller_config(controller_type=self.controller_type)
        # Cameras are ommited for now
        # self.cameras = OrderedDict()
        # for cam_name in external_cams.keys():
        #     self.cameras[cam_name] = external_cams[cam_name]

        self.steps = 0
        self.gripper_max_width = 0.08
        if(camera_enabled):
            self.camera_enabled = True
            self.camera = RealSenseCamera()
        else:
            self.camera_enabled = False
        # self.robot_interface._state_buffer = [[0, 0, 0, 0, 0, 0, 1]]

    @property
    def observation_space(self):
        # no observation space for testing
        ob_space = OrderedDict()

        ob_space['right'] = gym.spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(7,),
        )

        if self.camera_enabled:
            ob_space['image'] = gym.spaces.Box(
                low=-np.inf,
                high=np.inf,
                shape=self.camera.img_shape,
            ) 

            ob_space['depth'] = gym.spaces.Box(
                low=-np.inf,
                high=np.inf,
                shape=self.camera.depth_shape,
            )

        # for cam in self.cameras.keys():
            
        #     ob_space[f'{cam}_image'] = gym.spaces.Box(
        #         low=-np.inf,
        #         high=np.inf,
        #         shape=self.cameras[cam].img_shape,
        #     ) 

        #     ob_space[f'{cam}_depth'] = gym.spaces.Box(
        #         low=-np.inf,
        #         high=np.inf,
        #         shape=self.cameras[cam].depth_shape,
        #     )

        return gym.spaces.Dict(ob_space)

    @property
    def action_space(self):
        act_space = OrderedDict()
        
        if self.arm_enabled:
            act_space['right'] = gym.spaces.Box(
                low=-1.0,
                high=1.0,
                shape=(7,1),
            )

        return gym.spaces.Dict(act_space)


    def _observation(self):
        while (len(self.robot_interface._state_buffer) == 0) or \
                (len(self.robot_interface._gripper_state_buffer) == 0):
            print('Waiting for robot_interface state buffer...')
            time.sleep(0.01)
        last_state = self.robot_interface._state_buffer[-1]
        last_gripper_state = self.robot_interface._gripper_state_buffer[-1]

        ee_pose = np.array(last_state.O_T_EE).reshape(4, 4)
        pos = ee_pose[-1, :3]
        quat = rmat_to_quat(ee_pose[:3, :3])

        observations = AttrDict({
            'left': np.array([0, 0, 0, 0, 0, 0, 1, 1]),
            'right': np.r_[pos, quat, np.array(last_gripper_state.width)/self.gripper_max_width],
            'base': np.array([0, 0, 0])
        })
        # print(np.array(last_gripper_state.width)/0.08)

        if self.camera_enabled:
            camera_obs = self.camera.get_camera_obs()
            observations['image'] = camera_obs['image']
            observations['depth'] = camera_obs['depth']

        return observations

    def reset(self, *args, **kwargs):
        self.start_time = None
        self.end_time = None
        self.steps = 0
        
        return self._observation()
    
    def step(self, action):
        # print("action incoming")
        # axis seem to be swapped coming out of spacemouse teleop
        # action.right[0] *= -1.0
        # action.right[1] *= -1.0
        # action.right[3] *= -1.0
        # action.right[4] *= -1.0
        
        action = action.right.copy()
        if action is not None:
            action[-1] = 1 if action[-1]==0 else -1
            #temporary insert by Luca
            action[:-1] = action[:-1] * 0.85 # slowing it down to see if the planner is happier
            self.robot_interface.control(
                controller_type=self.controller_type,
                action=action,
                controller_cfg=self.controller_cfg,
            )
        
        self.end_time = time.time()
        if self.start_time is not None:
            # print('Idle time:', 1/self.frequency - (self.end_time-self.start_time))
            time.sleep(max(0., 1/self.frequency - (self.end_time-self.start_time)))
        self.start_time = time.time()

        obs = self._observation()
        rew = 0
        done = False
        info = {}

        self.steps += 1

        return obs, rew, done, info

if __name__=="__main__":
    franka = FrankaGym()
    x = franka.action_space
    breakpoint()


