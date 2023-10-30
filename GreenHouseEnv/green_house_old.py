__credits__ = ["Dmitry CHERNYSHEV"]
import math
from typing import TYPE_CHECKING, List, Optional

import numpy as np

from typing import List

import gym
import os
import sys
import pcse
sys.path.append(os.path.join(
    os.path.abspath(__file__),
    '..',
    'pcse_gym'
    )
                )
from pcse_gym.envs.common_env import PCSEEnv

from pcse.util import WOFOST80SiteDataProvider


from .calcsMethods import *
import cProfile  as cProfile  # Импортируем cProfile


_PATH_TO_FILE = os.path.dirname(os.path.join(
    os.path.abspath(__file__),
    '..',
    'pcse_gym'
    ))
_CONFIG_PATH = os.path.join(
    _PATH_TO_FILE, 
    'configs')

_DEFAULT_AGRO_FILE = 'agromanagement_fertilization.yaml'
_DEFAULT_CROP_FILE = 'lintul3_winterwheat.crop'
_DEFAULT_SITE_FILE = 'lintul3_springwheat.site'
_DEFAULT_SOIL_FILE = 'lintul3_springwheat.soil'

_DEFAULT_AGRO_FILE_PATH = os.path.join(_CONFIG_PATH, 'agro', _DEFAULT_AGRO_FILE)
_DEFAULT_CROP_FILE_PATH = os.path.join(_CONFIG_PATH, 'crop', _DEFAULT_CROP_FILE)
_DEFAULT_SITE_FILE_PATH = os.path.join(_CONFIG_PATH, 'site', _DEFAULT_SITE_FILE)
_DEFAULT_SOIL_FILE_PATH = os.path.join(_CONFIG_PATH, 'soil', _DEFAULT_SOIL_FILE)

_DEFAULT_CONFIG = 'Lintul3.conf'

obs_params_names = ['T_soil','humid_soil','pH','EC','NAVAIL','PAVAIL','KAVAIL','pH_mix','EC_mix','N','P','K' ]
low = np.array([
            10, # T_soil [°C]
            .5, # humid_soil [%]
            3, # pH
            100,# EC [μS/cm]
            43.27, # [kg/ha] 10, # NAVAIL [ppm] 
            8.654, # [kg/ha] 2, # PAVAIL [ppm]
            216.35, # [kg/ha] 50, # KAVAIL [ppm]
            # MIXTURE
            5, # pH_mix
            0,# EC_mix [μS/cm]
            0, # [kg/ha] N [ppm]
            0, # [kg/ha] P [ppm]
            0] # [kg/ha] K [ppm]
    
        ).astype(np.float32)
high = np.array(
    [
        # SOIL
        40, # # T_soil [°C]
        1, # humid_soil [%]
        9, # pH
        3000,# EC [μS/cm]
        21_635, # [kg/ha] 5000, # NAVAIL [ppm] NAVAIL(kg/га) = 5000 * 0.4327 * 10 = 21,635 кг/га
        4_327, # [kg/ha] 1000, # PAVAIL [ppm]
        4_327, # [kg/ha] 1000, # KAVAIL [ppm]
        # MIXTURE
        9, # pH_mix
        3000,# EC_mix [μS/cm]
        216_350 , # [kg/ha] 50_000, # N [ppm]
        108_175 , # [kg/ha] 25_000, # P [ppm]
        216_350 , # [kg/ha] 50_000, # K [ppm]
    ]).astype(np.float32)
low_high_dict = dict(zip(obs_params_names,np.concatenate((low[:,np.newaxis], high[:,np.newaxis]), axis=1)))
#region valves vars
_mixtures_growth = {
    0:{
        'N':20/100,
        'P':10/100,
        'K':20/100,
        'pH':6.0,
        'EC':1.2
    },
    1:{
        'N':12/100,
        'P':8/100,
        'K':10/100,
        'pH':6.5,
        'EC':1.3
    },
    2:{
        'N':0.025/100,
        'P':0.005/100,
        'K':0.015/100,
        'pH':6.2,
        'EC':1.4
    },
}
_mixtures_fruiting = {
    0:{        
        'N':10/100,
        'P':15/100,
        'K':30/100,
        'pH':6.5,
        'EC':1.8
    },
    1:{
        'N':25/100,
        'P':5/100,
        'K':15/100,
        'pH':6.2,
        'EC':1.4
    },
    2:{
        'N':15/100,
        'P':10/100,
        'K':25/100,
        'pH':6.3,
        'EC':1.5
    },
}
water={
    'N':0,
    'P':0,
    'K':0,
    'pH':7,
    'EC':1.8
}
#endregion
#region stages old
# [kg/ha]  = [ppm] * [kg/m^3] / (h)[m]=(0.35)[m]
# _stages = {
#     'seed_sowing':{
#             'T_soil':np.array([20,25]),# T [°C]
#             'humid_soil':np.array([.5,.6]),# phi [%]
#             'pH_soil':np.array([6.0,6.5]),# pH
#             'NAVAIL':np.array([100 , 200]),# N [ppm]
#             'PAVAIL':np.array([50 , 100]),# P [ppm]
#             'KAVAIL':np.array([150 , 300]),# K [ppm]
#             'vel_assim':np.array([0.2,0.5]) # л/час
#        },
#     'seedling':{
#             'T_soil':np.array([18,24]),# T [°C]
#             'humid_soil':np.array([.6,.7]),# phi [%]
#             'pH_soil':np.array([6.0,6.5]),# pH
#             'NAVAIL':np.array([100 , 200]),# N [ppm]
#             'PAVAIL':np.array([50 , 100]),# P [ppm]
#             'KAVAIL':np.array([150 , 300]),# K [ppm]
#             'vel_assim':np.array([0.5,2.0]) # л/час
#         },
#     'flowering_fruiting':{
#             'T_soil':np.array([21,24]),# T [°C]
#             'humid_soil':np.array([.7,.8]),# phi [%]
#             'pH_soil':np.array([6.0,6.5]),# pH
#             'NAVAIL':np.array([100 , 200]),# N [ppm]
#             'PAVAIL':np.array([50 , 100]),# P [ppm]
#             'KAVAIL':np.array([150 , 300]),# K [ppm]
#             'vel_assim':np.array([1,3]) # л/час
#         },
#     'fruit_ripening':{
#             'T_soil':np.array([20,23]),# T [°C]
#             'humid_soil':np.array([.6,.7]),# phi [%]
#             'pH_soil':np.array([6.0,6.5]),# pH
#             'NAVAIL':np.array([100 , 200]),# N [ppm]
#             'PAVAIL':np.array([50 , 100]),# P [ppm]
#             'KAVAIL':np.array([150 , 300]),# K [ppm]
#             'vel_assim':np.array([1.5,4.0]) # л/час
#         }
# }
#endregion
#region stages new
_stages = { 
    'seed_sowing':{ 
            'T_soil':np.array([20,25]),# T [°C] 
            'humid_soil':np.array([.5,.6]),# phi [%] 
            'pH':np.array([6.0,6.5]),# pH 
            'EC':np.array([0.2 , 0.5]),# EC/TDS [mS/cm]
            'NAVAIL':np.array([21635 , 43350]),# N [кг/га] 
            'PAVAIL':np.array([4327 , 8654]),# P [кг/га] 
            'KAVAIL':np.array([12981 , 25962]),# K [кг/га] 
            'vel_assim':np.array([0.2,0.5]) # л/час [4.8,]
        }, 
    'seedling':{ 
            'T_soil':np.array([18,24]),# T [°C] 
            'humid_soil':np.array([.6,.7]),# phi [%] 
            'pH':np.array([6.0,6.5]),# pH 
            'EC':np.array([0.2 , 0.8]),# EC/TDS [ mS/cm]
            'NAVAIL':np.array([21635 , 43350]),# N [кг/га] 
            'PAVAIL':np.array([4327 , 8654]),# P [кг/га] 
            'KAVAIL':np.array([12981 , 25962]),# K [кг/га] 
            'vel_assim':np.array([0.5,2.0]) # л/час 
        }, 
    'flowering_fruiting':{ 
            'T_soil':np.array([21,24]),# T [°C] 
            'humid_soil':np.array([.7,.8]),# phi [%] 
            'pH':np.array([6.0,6.5]),# pH 
            'EC':np.array([1.0 , 2.0]),# EC/TDS [ mS/cm]
            'NAVAIL':np.array([21635 , 43350]),# N [кг/га] 
            'PAVAIL':np.array([4327 , 8654]),# P [кг/га] 
            'KAVAIL':np.array([12981 , 25962]),# K [кг/га] 
            'vel_assim':np.array([1,3]) # л/час 
        }, 
    'fruit_ripening':{ 
            'T_soil':np.array([20,23]),# T [°C] 
            'humid_soil':np.array([.6,.7]),# phi [%] 
            'pH':np.array([6.0,6.5]),# pH 
            'EC':np.array([0.8 , 1.5]),# EC/TDS [ mS/cm]
            'NAVAIL':np.array([21635 , 43350]),# N [кг/га] 
            'PAVAIL':np.array([4327 , 8654]),# P [кг/га] 
            'KAVAIL':np.array([12981 , 25962]),# K [кг/га] 
            'vel_assim':np.array([1.5,4.0]) # л/час 
        } 
}
#endregion
SOIL_DENSITY = np.mean(np.array([1.2,1.5])) # g/cm^3 -> 1000 kg/m^3

class GreenHousePCSEEnv(PCSEEnv):
    name = 'GreenHousePCSE_old'
    def __init__(self,
                 model_config: str = _DEFAULT_CONFIG,
                 agro_config: str = _DEFAULT_AGRO_FILE_PATH,
                 growth_stage:str='seed_sowing',
                 growth_fruiting_vents:list=[0,0],
                 crop_parameters=_DEFAULT_CROP_FILE_PATH,
                 site_parameters=_DEFAULT_SITE_FILE_PATH,
                 soil_parameters=_DEFAULT_SOIL_FILE_PATH,
                 soil_density=SOIL_DENSITY,
                 R_cylinder:float=0.35,
                 flow_in:float=15,# litre/hour
                 flow_out:float=65, # litre/hour
                 h_cylinder:float=0.35,
                 years=None,
                 location=None, 
                 seed: int = None, # seed
                 timestep: int = 1, # day
                 maxsteps:int=72,
                 **kwargs 
        ):
        assert growth_stage in list(_stages.keys())
        assert min(list(_mixtures_growth.keys())) <= growth_fruiting_vents[0] <= max(list(_mixtures_growth.keys()))
        assert min(list(_mixtures_fruiting.keys())) <= growth_fruiting_vents[1] <= max(list(_mixtures_fruiting.keys()))
        super().__init__(model_config,agro_config,crop_parameters,site_parameters,soil_parameters,years,location,seed,timestep,**kwargs)
        

        # Define Gym observation space
        self.observation_space = self._get_observation_space()
        # Define Gym action space
        self.action_space = self._get_action_space()
        # Create a PCSE engine / crop growth model
        self._model = self._init_pcse_model()
        #general
        self._sec_model = {
            'general':{
                'T_soil':self._get_default_space('T_soil',low_high_dict).sample(),
                'pH': self._get_default_space('pH',low_high_dict).sample(),
                'EC': self._get_default_space('EC',low_high_dict).sample(),
                'humid_soil': self._get_default_space('humid_soil',low_high_dict).sample()
                },
            'mixture':{
                'EC' : self._get_default_space('EC_mix',low_high_dict).sample(),
                'pH': self._get_default_space('pH_mix',low_high_dict).sample()
            }
        }
        self.dV = 0
        
        self.optimal_space = _stages[growth_stage]
        
        self.R = R_cylinder
        self.h = h_cylinder
        
        self.V = 8.5 # in barrell
        
        self.soil_density = soil_density * 1e3 # kg/m3
        
        
        # VALVES (actions [0,1])
        self.valve_growth = _mixtures_growth[growth_fruiting_vents[0]]
        self.valve_fruiting = _mixtures_fruiting[growth_fruiting_vents[1]]
        self.valve_water = water
                
        self.flow_in = flow_in
        self.flow_out = flow_out
                        
        self.day_counts = 0
        self.maxsteps=maxsteps
    def step(self, action: np.ndarray):
        self.day_counts += 1
        # Create a dict for storing info
        info = dict()
        
        self._apply_action(action)
        # Run the crop growth model
        self._model.run(days=self._timestep)
        # Get the model output
        output = self._model.get_output()[-self._timestep:]
        info['days'] = [day['day'] for day in output]

        # Construct an observation and reward from the new environment state
        o = self._get_observation(output)
        r = self._get_reward()
        # Check whether the environment has terminated
        done = True \
            if self.day_counts > self.maxsteps or \
                np.abs(r - 100) <= .05  or \
                not(0 < self.V + self.dV <= self.V) \
            else False 
        if done:
            info['output_history'] = self._model.get_output()
            info['summary_output'] = self._model.get_summary_output()
            info['terminal_output'] = self._model.get_terminal_output()
        truncated = False
        terminated = done
        # Return all values
        return o, r, terminated, truncated, info
    def _apply_action(self, action):
        # get valve vars
        valve_growth_coeff,valve_fruiting_coeff,valve_water_coeff,valve_out = action[:4]
        T_air_raw,humid_air_raw = action[4:]
        # Air
        T_air,humid_air = 16 + 14 * T_air_raw, humid_air_raw
        # Calcs irrigation N,P,K
        #region 1. Calcs in volume of water
        N1,N2,N3 = [V_i(dt=self._timestep*24,a_i=a_i,U=self.flow_in) for a_i in [valve_growth_coeff,valve_fruiting_coeff,valve_water_coeff]]
        mixture= {
                "N": 0,
                'P': 0,
                'K': 0,
                'irrigation':0
            }
        #endregion
        #region 2. Calcs N,P,K, irragtion
        N_out = V_i(dt=self._timestep*24,a_i=valve_out,U=self.flow_out + \
            np.random.uniform(self.optimal_space['vel_assim'][0],self.optimal_space['vel_assim'][1]))
        
        for elem in ['N','P','K']:
                mixture[f'{elem}'] = sum(
                    [valve[elem] * N  * 10 / (np.pi * self.R**2) for N, valve in zip([N1, N2, N3], [self.valve_growth, self.valve_fruiting, self.valve_water])])
                # mixture[f'{elem}'] = np.clip(mixture[f'{elem}'], low_high_dict[f'{elem}AVAIL'][0], low_high_dict[f'{elem}AVAIL'][1])
            

        mixture['irrigation'] = sum([N * 10 / (np.pi * self.R**2) for N in [N1, N2, N3]])  - sum([mixture[elem] for elem in ['N','P','K' ]])
        
        self._model._send_signal(signal=pcse.signals.irrigate,
                                 amount=mixture['irrigation'] ,
                                 efficiency=np.random.uniform(0.85,1),
                                 )
        self._model._send_signal(signal=pcse.signals.apply_npk,
                                 N_amount=mixture['N'] ,
                                 P_amount=mixture['P'] ,
                                 K_amount=mixture['K'] ,
                                 N_recovery=np.random.uniform(0.55,0.95),
                                 P_recovery=np.random.uniform(0.55,0.95),
                                 K_recovery=np.random.uniform(0.55,0.95)
                                 )
        #endregion
        # my control
        
        #region 3. Apply EC
        self.dV = N1/24 + N2/24 + N3/24 - N_out/24
        if self.dV < 0 and abs(self.dV) >= self.V: # емкость пустая
            self._sec_model['mixture']['EC'] = 0
            self._sec_model['mixture']['pH'] = 0
        else: # емкость не пустая, то есть прибыток раствора
            self.V = np.clip(self.V + self.dV,0.2,8.5)
            if not(all([N_i==0 for N_i in [N1,N2,N3]])):
                self._sec_model['mixture']['EC'] = ((self.V - self.dV) * self._sec_model['mixture']['EC'] \
                    + N1/sum([N1,N2,N3]) * self.valve_growth['EC'] \
                    + N2//sum([N1,N2,N3]) * self.valve_fruiting['EC'] \
                    + N3//sum([N1,N2,N3]) * self.valve_water['EC'] \
                    - N_out/24 * self._sec_model['mixture']['EC']) / (self.V)
                
                self._sec_model['mixture']['pH'] = ((self.V- self.dV) * self._sec_model['mixture']['pH'] \
                    + N1/sum([N1,N2,N3]) * self.valve_growth['pH'] \
                    + N2/sum([N1,N2,N3]) * self.valve_fruiting['pH'] \
                    + N3/sum([N1,N2,N3]) * self.valve_water['pH'] \
                    - N_out/24 * self._sec_model['mixture']['pH']) / (self.V)   
            else:
                self._sec_model['mixture']['EC'] = ((self.V - self.dV) * self._sec_model['mixture']['EC'] \
                    - N_out/24 * self._sec_model['mixture']['EC']) / (self.V)
                
                self._sec_model['mixture']['pH'] = ((self.V- self.dV) * self._sec_model['mixture']['pH'] \
                    - N_out/24 * self._sec_model['mixture']['pH']) / (self.V)     
                
        self._sec_model['general']['EC'] = EC(
                    ec_tds_0=self._sec_model['general']['EC'],
                    ec_tds_solution=self._sec_model['mixture']['EC'],
                    V=self.V,
                    V_soil=self.R**2*self.h * np.pi,dt=self._timestep) 
        self._sec_model['general']['pH'] = pH(
            ph_0=self._sec_model['general']['pH'],
            pH_solution=self._sec_model['mixture']['pH'],
            V=self.V,
            V_soil=self.R**2*self.h * np.pi,dt=self._timestep) 
        #endregion
        #region 4. Apply Temp and humid
        self._sec_model['general']['humid_soil'] = temp_soil(
            dt=1/24*self._timestep, 
            T_soil_0=self._sec_model['general']['T_soil'], 
            T_air=T_air
            )
        self._sec_model['general']['humid_soil'] += calculate_diff_eq(
            dt=1/24*self._timestep,
            RH=self._sec_model['general']['humid_soil'],
            AH=humid_air, T_air=T_air
        ) 
        #endregion
     #region utils  
    def _get_observation_space(self) -> gym.spaces.Space:
        space = gym.spaces.Dict({
            'crop_model': self._get_observation_space_crop_model(),
            'weather': self._get_observation_space_weather(),
        })
        # Combine the nested dictionaries into a flat dictionary
        flat_space = {}
        for key, value in space.spaces.items():
            if isinstance(value, gym.spaces.Dict):
                for subkey, subvalue in value.spaces.items():
                    flat_space[key + "_" + subkey] = subvalue
            else:
                flat_space[key] = value
        return gym.spaces.Dict(flat_space) 
    def _get_observation_space(self) -> gym.spaces.Space:
        space = gym.spaces.Dict({
            'crop_model': self._get_observation_space_crop_model(),
            'weather': self._get_observation_space_weather()
        })
        # Combine the nested dictionaries into a flat dictionary
        flat_space = {}
        for key, value in space.spaces.items():
            if isinstance(value, gym.spaces.Dict):
                for subkey, subvalue in value.spaces.items():
                    flat_space[key + "_" + subkey] = subvalue
            else:
                flat_space[key] = value
        return gym.spaces.Dict(flat_space) 
    def _get_default_space(self,name,low_high_dict) -> gym.spaces.Space:
        return gym.spaces.Box(low_high_dict[name][0], low_high_dict[name][1], shape=(self._timestep,))
    def _get_action_space(self) -> gym.spaces.Space:
        return gym.spaces.Box(
           np.array([0,0,0,0,0,0]),
           np.array([1,1,1,1,1,1]), 
        )
    def _get_reward(self) -> float:
        output = self._get_observation(self._model.get_output())
        output = output[-1]
        # print(f'output {output}')
        new_output = output.copy()
        check_elems = list(set(self.optimal_space.keys()) & set(output.keys()))
        # check_elems.remove('vel_assim')
        
        new_output['T_soil'] = self._sec_model['general']['T_soil']
        new_output['humid_soil'] = self._sec_model['general']['humid_soil']
        new_output['pH'] = self._sec_model['general']['pH']
        new_output['EC'] = self._sec_model['general']['EC']
        # print(f'new_output.keys() = {new_output.keys()}')
        for key in ['T_soil','humid_soil','pH','EC']:
            check_elems.append(key)
        
        def calc_loc_rew(state_params,param_name,best_r):
          return np.clip(np.exp(-(state_params[param_name] - np.mean(self.optimal_space[param_name]))**2),-100,1)*best_r
        
        reward_list = np.array([calc_loc_rew(self.optimal_space,key,100) for key in check_elems])
        if abs(self.dV ) > 8.5:
            reward_list= np.append(
                reward_list,
                -1000
            )
        else:
            reward_list = np.append(
                reward_list,
                np.clip(np.exp(-(self.dV/8.5)**2), -100, 1) * 100
            )
        return np.mean(reward_list)      
    #endregion
    
    def reset(self,
              *,
              seed: int = None,
              return_info: bool = False,
              options: dict = None):
        self.day_counts = 0
        self._sec_model = {
            'general':{
                'T_soil':self._get_default_space('T_soil',low_high_dict).sample(),
                'pH': self._get_default_space('pH',low_high_dict).sample(),
                'EC': self._get_default_space('EC',low_high_dict).sample(),
                'humid_soil': self._get_default_space('humid_soil',low_high_dict).sample()
                },
            'mixture':{
                'EC' : self._get_default_space('EC_mix',low_high_dict).sample(),
                'pH': self._get_default_space('pH_mix',low_high_dict).sample()
            }
        }
    
        self.dV = 0
        self.V = 8.5
        
        # Optionally set the seed
        super().reset(seed=seed)

        # Create an info dict
        info = dict()

        # Create a PCSE engine / crop growth model
        self._model = self._init_pcse_model()
        output = self._model.get_output()[-self._timestep:]
        o = self._get_observation(output)
        info['date'] = self.date

        return o, info if return_info else o    
    def render(self, mode='human'):
        pass