__credits__ = ["Dmitry CHERNYSHEV"] 
from typing import TYPE_CHECKING, List, Optional

import numpy as np
from datetime import datetime, timedelta
from typing import List

import gym
import os
import sys 
import datetime
sys.path.append(
    os.path.join(
        os.path.abspath(__file__), 
        '..',    
        'pcse_gym'   
        )                
    )
 

from .calcsMethods import *
import cProfile  as cProfile 
 
obs_params_names = [
    'T_soil',
    'humid_soil',
    'pH','EC', 
    'pH_mix','EC_mix', 
    ]
low = np.array([
            15, # T_soil [°C]
            .4, # humid_soil [%]
            3, # pH
            100,# EC [μS/cm] 
            # MIXTURE
            5, # pH_mix
            0,# EC_mix [μS/cm] 
            ]
    
        ).astype(np.float32)
high = np.array(
    [
        # SOIL
        30, # # T_soil [°C]
        1, # humid_soil [%]
        9, # pH
        3000,# EC [μS/cm]
        # MIXTURE
        9, # pH_mix
        3000,# EC_mix [μS/cm]
    ]).astype(np.float32)
low_high_dict = dict(zip(obs_params_names,np.concatenate((low[:,np.newaxis], high[:,np.newaxis]), axis=1)))
 

#region stages new
_stages = {
    'seed_sowing': {
        'T_soil': np.array([20, 25]),  # T [°C]
        'humid_soil': np.array([0.5, 0.6]),  # phi [%]
        'pH': np.array([6.0, 6.5]),  # pH
        'EC': np.array([0.2, 0.5]),  # EC/TDS [mS/cm]
        'vel_assim': np.array([0.2, 0.5])  # л/час [4.8, ]
    },
    'seedling': {
        'T_soil': np.array([18, 24]),  # T [°C]
        'humid_soil': np.array([0.6, 0.7]),  # phi [%]
        'pH': np.array([6.0, 6.5]),  # pH
        'EC': np.array([0.2, 0.8]),  # EC/TDS [mS/cm]
        'vel_assim': np.array([0.5, 2.0])  # л/час
    },
    'flowering_fruiting': {
        'T_soil': np.array([21, 24]),  # T [°C]
        'humid_soil': np.array([0.7, 0.8]),  # phi [%]
        'pH': np.array([6.0, 6.5]),  # pH
        'EC': np.array([1.0, 2.0]),  # EC/TDS [mS/cm]
        'vel_assim': np.array([1, 3])  # л/час
    },
    'fruit_ripening': {
        'T_soil': np.array([20, 23]),  # T [°C]
        'humid_soil': np.array([0.6, 0.7]),  # phi [%]
        'pH': np.array([6.0, 6.5]),  # pH
        'EC': np.array([0.8, 1.5]),  # EC/TDS [mS/cm] 
        'vel_assim': np.array([1.5, 4.0])  # л/час
    }
}
#endregion
SOIL_DENSITY = np.mean(np.array([1.2,1.5])) # g/cm^3 -> 1000 kg/m^3

class GreenHousePCSEEnvVer2(gym.Env):
    name = 'GreenHousePCSEEnvVer2'
    _additional_variables = [
        'T_soil',
        'humid_soil', 
        'pH',
        'EC', 
        ]
    
    def __init__(self, 
                 growth_stage:str='seed_sowing', 
                 soil_density=SOIL_DENSITY,
                 R_cylinder:float=0.35,
                 h_cylinder:float=0.35,
                 V:float=8.5,  
                 seed: int = None, # seed
                 timestep: int = 1, # day
                 flow_in:float=100,# litre/hour
                 flow_out:float=65, # litre/hour
                 maxsteps:int=180,
                 **kwargs
                 ):
        assert growth_stage in list(_stages.keys())
       
        super().reset(seed=seed)
        assert timestep > 0
        
        # If any parameter files are specified as path, convert them to a suitable object for pcse 

        self._timestep = timestep  
        # Define Gym observation space
        self.observation_space = self._get_observation_space()
        # Define Gym action space
        self.action_space = self._get_action_space()
        
        # VALVES (actions [0,1]) 
        self.optimal_space = _stages[growth_stage]
        self.R = R_cylinder*10
        self.h = h_cylinder*10
        self.V = V # in barrell
        self.dV  = 0
        self.V_soil = np.pi * (self.R*10)**2 * (10*self.h) # litres
        
        self.soil_density = soil_density * 1e3 # kg/m3
        # Create  a emperature humid EC and pH model
        self._saved_days = self._init_model()
                        
        self.day_counts = 0
        self.maxsteps = maxsteps
        self.flow_in = flow_in
        self.flow_out = flow_out

    def _init_model(self) -> list:
        init_dict = {
                var:self._get_default_space(var,1).sample() for var in self.__class__._additional_variables
            }
        #
        initial_time = datetime.datetime(2018, 6, 1)
        answer = [{'time':initial_time}]
        for _ in range(self._timestep):
            new_time = answer[-1]['time'] + datetime.timedelta(seconds=3)
            answer.append({'time':new_time})
        
        for i  in  range(len(answer)):
            for key in init_dict.keys():
                answer[i][key] = init_dict.copy()[key][0]
        
        return answer
        
    

    def _get_observation_space(self) -> gym.spaces.Space:
        observation_space = {  }
        # SOIL
        for var in self.__class__._additional_variables:
            observation_space[var] = self._get_default_space(var, self._timestep)
        # MIXTURE
        for var in ['pH_mix','EC_mix']:
            observation_space[var] = self._get_default_space(var, self._timestep)   
        space = gym.spaces.Dict(observation_space) 
        return space

    def _get_action_space(self) -> gym.spaces.Space:
        return gym.spaces.Box(
           np.zeros(6),# irig outlet pH EC T_air humid_air
           np.ones(6), 
        )
    
    def _get_default_space(self,name,numb,low_high_dict=low_high_dict) -> gym.spaces.Space:
        return gym.spaces.Box(low_high_dict[name][0], low_high_dict[name][1], shape=(numb,))
#region properties
    @property
    def output_variables(self) -> list:
        return list(self._output_variables)
    @property
    def summary_variables(self) -> list:
        return list(self._summary_variables)
    @property
    def weather_variables(self):
        return list(self._weather_variables)
    @property
    def _campaigns(self) -> dict:
        return self._agro_management[0]
    @property
    def _first_campaign(self) -> dict:
        return self._campaigns[min(self._campaigns.keys())]
    @property
    def _last_campaign(self) -> dict:
        return self._campaigns[max(self._campaigns.keys())]
    @property
    def start_date(self) -> datetime.date:
        return self._model.agromanager.start_date
    @property
    def end_date(self) -> datetime.date:
        return self._model.agromanager.end_date
    @property
    def date(self) -> datetime.date:
        return self._model.day
#endregion

    def step(self, action: np.ndarray):
        self.day_counts += 1
        # Create a dict for storing info
        info = dict() 
        
        current_args = {
            var:self._get_observation()[var][-1] for var in self.__class__._additional_variables
        } 
        
        current_args = self._apply_action(action,current_args) 
        # Get the model output
        self._update_saved_days(current_args) 
        # Update self._saved_days
        info['times'] = [time['time'] for time in self._saved_days]
        # Construct an observation and reward from the new environment state
        o = self._get_observation( )
        r = self._get_reward()
        # Check whether the environment has terminated
        done = True \
            if self.day_counts > self.maxsteps or \
                np.abs(r - 100) <= .0005  \
            else False  
        truncated = False
        terminated = done
        # Return all values 
        return o, r, terminated, truncated, info
    def _update_saved_days(self,current_args:dict)-> list:
        init_dict = current_args
        new_times = []
        for _ in range(self._timestep):
            new_time = self._saved_days[-1]['time'] + datetime.timedelta(seconds=3)
            new_times.append({'time':new_time}) 
        # print(f'[GreenHousePCSEEnv [l478]] {self._model.get_output()}')
        
        for i  in  range(len(new_times)):
            for key in init_dict.keys():
                new_times[i][key] = init_dict.copy()[key]
        self._saved_days = self._saved_days + new_times 
        self._saved_days = sorted(self._saved_days, key=lambda x: x['time'])     
    def _apply_action(self, action,current_args):
        # get valve vars
        valve_irig,valve_irig_out,valve_pH,valve_EC = action[:4]     
        T_air_raw,humid_air_raw = action[4:]  
        # Air
        T_air,humid_air = 16 + 14 * T_air_raw, humid_air_raw
        _pH = 14*valve_pH
        _EC = 270*valve_EC # сульфат магния
        
        # Calcs irrigation N,P, K 
        N_irig = valve_irig*self.flow_in#V_i(dt=self._timestep*24,a_i=valve_irig,U=self.flow_in) # l
        N_irig_out = valve_irig_out*(self.flow_out + \
            np.random.uniform(self.optimal_space['vel_assim'][0],self.optimal_space['vel_assim'][1]))
        self.dV = N_irig - N_irig_out 
        
        #region 3. Apply EC 
        # EC_final = (EC_soil * V_soil + EC_mix * V_mix) / (V_soil + V_mix)
        current_args['EC'] = np.clip((current_args['EC'] * self.V_soil + _EC * self.V)/( self.V_soil+self.V),0,270)
        # # ΔpH = (pH_mix - pH_soil) × (V_mix / V_soil)
        current_args['pH'] = np.clip(current_args['pH'] + (_pH  - current_args['pH']) * (self.V/self.V_soil),0,14)
        #endregion
        #region 4. Apply Temp and humid
        current_args['T_soil'] = temp_soil(
            dt=1/24*self._timestep, 
            T_soil_0=current_args['T_soil'], 
            T_air=T_air
            )
        def diff_humid_time(theta_t0,t, theta_air, V_soil, N, E):
            '''
            Параметры:
            t - текущее время (в сутках)
            theta_air - влажность воздуха
            V_soil - объем почвы в горшке (в литрах)
            N - объем добавленной жидкости (в литрах)
            E - естественное испарение жидкости из почвы (в литрах в сутки) 
            '''
            theta_initial = theta_t0  #
            k = 0.1  # Примерный коэффициент 
            delta_theta = (theta_air - theta_initial) * (1 - np.exp(-k * t)) - E * t 
            updated_theta_t0 = theta_initial + delta_theta 
            updated_theta_t0 += N / V_soil

            return updated_theta_t0
        current_args['humid_soil'] =  diff_humid_time(
            theta_t0=current_args['humid_soil'],
            t= self._timestep,
            theta_air=humid_air,
            V_soil=1e3*self.h*np.pi*self.R**2,
            N=N_irig,#N/24,
            E =np.random.uniform(0.001,0.01)
            )
        #endregion
        return current_args
    def _get_observation(self) -> dict:
        # Get the datetime objects characterizing the specific days
        additional_vars = {key:[] for key in self._saved_days[0].keys() if key != 'time'}
        for i  in range(len(self._saved_days)-self._timestep,len(self._saved_days)):
            for key in self._saved_days[i].keys():
                if key != 'time':
                    additional_vars[key].append( self._saved_days[i][key])

        o = { 
            **additional_vars
        }

        return o 
    def _get_reward(self) -> dict:
        output = self._get_observation()
        check_params = {
            key:0 for key in self.optimal_space.keys() if key != 'vel' or not(key.endswith('_mix',))
        } #
        for key in output.keys():
            if isinstance(output[key],dict): # crop_model
                # Получите ключи как множества
                keys1 = set(output[key].keys())
                keys2 = set(check_params.keys())

                # Найдите пересечение ключей
                common_keys = keys1 & keys2

                # Преобразуйте множество ключей в список, если это необходимо
                common_keys_list = list(common_keys)
                for _key in common_keys_list:
                    check_params[_key] = output[key][_key][-1]
            else:
                # Получите ключи как множества
                keys1 = set(output.keys())
                keys2 = set(check_params.keys())

                # Найдите пересечение ключей
                common_keys = keys1 & keys2

                # Преобразуйте множество ключей в список, если это необходимо
                common_keys_list = list(common_keys)
                for _key in common_keys_list:
                    check_params[_key] = output[_key][-1]
        # check_params fullfilled now
        
        def calc_loc_rew(state_params,param_name,best_r):
            
            if self.optimal_space[param_name][0] <= state_params[param_name] <= self.optimal_space[param_name][1]:
                return best_r
            else:
                return  np.clip(1-(state_params[param_name] - np.mean(self.optimal_space[param_name]))**2 ,-1,1 )*best_r   
        reward_dict = {}
        for key in check_params.keys():
            if key != 'vel_assim':
                reward_dict[key] = calc_loc_rew(check_params,key,100)
        reward_dict['dV'] = -100 if abs(self.dV ) > 8.5 else np.clip(
            1-(self.dV/8.5)**2,# np.exp(-(self.dV/8.5)**2), 
            -1, 
            1) * 100
        # print(f'Rewards:Mean(r) = {np.mean(np.array(list(reward_dict.values())))}; ({", ".join( [f"{key}: {value}" for key, value in reward_dict.items()])})')
        # # Создайте массив NumPy
        return np.mean(np.array(list(reward_dict.values())))
    def reset(self,
              *,
              seed: int = None,
              return_info: bool = False,
              options: dict = None
              ): 
        #region new
        super().reset(seed=seed)
        info = dict()
        # Create a PCSE engine / crop growth model
        self._saved_days = self._init_model()
        self.dV=0 
        # Создайте словарь, аналогичный выводу env._get_observation_space().sample()
        observation_space = self._get_observation()
        return observation_space, info if return_info else observation_space
        #endregion
    def render(self, mode="human"):
        pass  # Nothing to see here
