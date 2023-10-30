__credits__ = ["Dmitry CHERNYSHEV"]
import math
from typing import TYPE_CHECKING, List, Optional

import numpy as np
import yaml

from typing import List

import gym
import os
import sys
import pcse
import datetime
sys.path.append(
    os.path.join(
        os.path.abspath(__file__), 
        '..',    
        'pcse_gym'   
        )                
    )
from pcse_gym.envs.common_env import PCSEEnv,replace_years,get_weather_data_provider,Engine

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

# Однако, в общих чертах, максимальные концентрации для азота, фосфора и калия в питательных растворах для гидропоники и подкормок растений могут составлять:

# Максимальная концентрация азота (N) в растворе: до 1500 г/л (1.5 кг/л).
# Максимальная концентрация фосфора (P) в растворе: до 200 г/л (0.2 кг/л).
# Максимальная концентрация калия (K) в растворе: до 2000 г/л (2 кг/л).

obs_params_names = [
    'NAVAIL','PAVAIL','KAVAIL',
    ]
low = np.array([
            10,#43.27, # [kg/ha] 10, # NAVAIL [ppm] 
            10,#8.654, # [kg/ha] 2, # PAVAIL [ppm]
            10,#216.35, # [kg/ha] 50, # KAVAIL [ppm]
            ] 
    
        ).astype(np.float32)
high = np.array(
    [
        400,#21_635, # [kg/ha] 5000, # NAVAIL [ppm] NAVAIL(kg/га) = 5000 * 0.4327 * 10 = 21,635 кг/га
        400,#4_327, # [kg/ha] 1000, # PAVAIL [ppm]
        400,#4_327, # [kg/ha] 1000, # KAVAIL [ppm]
    ]).astype(np.float32)
low_high_dict = dict(zip(obs_params_names,np.concatenate((low[:,np.newaxis], high[:,np.newaxis]), axis=1)))
#region valves vars
_mixtures_growth = {
    0:{
        'N':20/100,
        'P':10/100,
        'K':20/100,
    },
    1:{
        'N':12/100,
        'P':8/100,
        'K':10/100,
    },
    2:{
        'N':0.025/100,
        'P':0.005/100,
        'K':0.015/100,
    },
}
_mixtures_fruiting = {
    0:{        
        'N':10/100,
        'P':15/100,
        'K':30/100,
    },
    1:{
        'N':25/100,
        'P':5/100,
        'K':15/100,
    },
    2:{
        'N':15/100,
        'P':10/100,
        'K':25/100,
    },
}
water={
    'N':0,
    'P':0,
    'K':0,
}
#endregion

#region stages new
_stages = {
    'seed_sowing': {
        'NAVAIL': np.array([40, 120]),  # N [кг/га]
        'PAVAIL': np.array([10, 40]),  # P [кг/га]
        'KAVAIL': np.array([60, 200]),  # K [кг/га]
    },
    'seedling': {
        'NAVAIL': np.array([60, 150]),  # N [кг/га]
        'PAVAIL': np.array([15, 45]),  # P [кг/га]
        'KAVAIL': np.array([70, 220]),  # K [кг/га]
    },
    'flowering_fruiting': {
        'NAVAIL': np.array([70, 180]),  # N [кг/га]
        'PAVAIL': np.array([20, 60]),  # P [кг/га]
        'KAVAIL': np.array([80, 240]),  # K [кг/га]
    },
    'fruit_ripening': {
        'NAVAIL': np.array([50, 130]),  # N [кг/га]
        'PAVAIL': np.array([12, 42]),  # P [кг/га]
        'KAVAIL': np.array([65, 210]),  # K [кг/га]
    }
}
#endregion
SOIL_DENSITY = np.mean(np.array([1.2,1.5])) # g/cm^3 -> 1000 kg/m^3

"""
    Gymnasium Environment built around the PCSE library for crop simulation
    Gym:  https://github.com/Farama-Foundation/Gymnasium
    PCSE: https://github.com/ajwdewit/pcse
    
    Based on the PCSE-Gym environment built by Hiske Overweg (https://github.com/BigDataWUR/crop-gym)
    
"""


def replace_years(agro_management, years):
    if not isinstance(years, list):
        years = [years]

    updated_agro_management = [{k.replace(year=year): v for k, v in agro.items()} for agro, year in
                               zip(agro_management, years)]

    def replace_year_value(d, year):
        for k, v in d.items():
            if isinstance(v, dict):
                replace_year_value(v, year)
            else:
                if isinstance(v, datetime.date):
                    up_dict = {k: v.replace(year=year)}
                    d.update(up_dict)

    for agro, year in zip(updated_agro_management, years):
        replace_year_value(agro, year)
    return updated_agro_management


def get_weather_data_provider(location) -> pcse.db.NASAPowerWeatherDataProvider:
    wdp = pcse.db.NASAPowerWeatherDataProvider(*location)
    return wdp


class Engine(pcse.engine.Engine):
    """
    Wraps around the PCSE engine/crop model to set a flag when the simulation has terminated
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._flag_terminated = False

    @property
    def terminated(self):
        return self._flag_terminated

    def _terminate_simulation(self, day):
        super()._terminate_simulation(day)
        self._flag_terminated = True


class NPKPCSEEnv(gym.Env):
    
    name = 'NPKPCSEEnv'
    _PATH_TO_FILE = os.path.dirname(os.path.realpath(__file__))
    _CONFIG_PATH = os.path.join(_PATH_TO_FILE, 'configs')

    _DEFAULT_AGRO_FILE = 'agromanagement_fertilization.yaml'
    _DEFAULT_CROP_FILE = 'lintul3_winterwheat.crop'
    _DEFAULT_SITE_FILE = 'lintul3_springwheat.site'
    _DEFAULT_SOIL_FILE = 'lintul3_springwheat.soil'

    _DEFAULT_AGRO_FILE_PATH = os.path.join(_CONFIG_PATH, 'agro', _DEFAULT_AGRO_FILE)
    _DEFAULT_CROP_FILE_PATH = os.path.join(_CONFIG_PATH, 'crop', _DEFAULT_CROP_FILE)
    _DEFAULT_SITE_FILE_PATH = os.path.join(_CONFIG_PATH, 'site', _DEFAULT_SITE_FILE)
    _DEFAULT_SOIL_FILE_PATH = os.path.join(_CONFIG_PATH, 'soil', _DEFAULT_SOIL_FILE)

    _DEFAULT_CONFIG = 'Lintul3.conf'

    def __init__(self,
                 model_config: str = _DEFAULT_CONFIG,
                 agro_config: str = _DEFAULT_AGRO_FILE_PATH,
                 crop_parameters=_DEFAULT_CROP_FILE_PATH,
                 site_parameters=_DEFAULT_SITE_FILE_PATH,
                 soil_parameters=_DEFAULT_SOIL_FILE_PATH,
                 years=None,
                 location=None,
                 seed: int = None,
                 timestep: int = 1,
                 growth_stage:str='seed_sowing',
                 _max_steps:int=250,
                 **kwargs
                 ):

        assert timestep > 0

        # Optionally set the seed
        super().reset(seed=seed)

        # If any parameter files are specified as path, convert them to a suitable object for pcse
        if isinstance(crop_parameters, str):
            crop_parameters = pcse.fileinput.PCSEFileReader(crop_parameters)
        if isinstance(site_parameters, str):
            site_parameters = pcse.fileinput.PCSEFileReader(site_parameters)
        if isinstance(soil_parameters, str):
            soil_parameters = pcse.fileinput.PCSEFileReader(soil_parameters)

        # Set location
        if location is None:
            location = (52, 5.5)
        self._location = location
        self._timestep = timestep
        self._max_steps = _max_steps


        # Store the crop/soil/site parameters
        self._crop_params = crop_parameters
        self._site_params = site_parameters
        self._soil_params = soil_parameters

        # Store the agro-management config
        with open(agro_config, 'r') as f:
            self._agro_management = yaml.load(f, Loader=yaml.SafeLoader)

        if years is not None:
            self._agro_management = replace_years(self._agro_management, years)

        # Store the PCSE Engine config
        self._model_config = model_config

        # Get the weather data source
        self._weather_data_provider = get_weather_data_provider(self._location)

        # Create a PCSE engine / crop growth model
        self._model = self._init_pcse_model()

        # Use the config files to extract relevant settings
        model_config = pcse.util.ConfigurationLoader(model_config)
        self._output_variables = model_config.OUTPUT_VARS  # variables given by the PCSE model output
        self._summary_variables = model_config.SUMMARY_OUTPUT_VARS  # Summary variables are given at the end of a run
        self._weather_variables = list(pcse.base.weather.WeatherDataContainer.required)

        # Define Gym observation space
        self.observation_space = self._get_observation_space()
        # Define Gym action space
        self.action_space = self._get_action_space()
        self.optimal_space = _stages[growth_stage]
        
    def _init_pcse_model(self, *args, **kwargs) -> Engine:

        # Combine the config files in a single PCSE ParameterProvider object
        self._parameter_provider = pcse.base.ParameterProvider(cropdata=self._crop_params,
                                                               sitedata=self._site_params,
                                                               soildata=self._soil_params,
                                                               )
        # Create a PCSE engine / crop growth model
        model = Engine(self._parameter_provider,
                       self._weather_data_provider,
                       self._agro_management,
                       config=self._model_config,
                       )
        # The model starts with output values for the initial date
        # The initial observation should contain output values for an entire timestep
        # If the timestep > 1, generate the remaining outputs by running the model
        if self._timestep > 1:
            model.run(days=self._timestep - 1)
        return model

    def _get_observation_space(self) -> gym.spaces.Space:
        space = gym.spaces.Dict({
            'crop_model': self._get_observation_space_crop_model(),
            'weather': self._get_observation_space_weather(),
        })
        flat_space = {}
        for key, value in space.spaces.items():
            if isinstance(value, gym.spaces.Dict):
                for subkey, subvalue in value.spaces.items():
                    flat_space[f"{key}_{subkey}"] = subvalue
            else:
                flat_space[key] = value
        return gym.spaces.Dict(flat_space)


    def _get_observation_space_weather(self) -> gym.spaces.Space:
        return gym.spaces.Dict(
            {
                'IRRAD': gym.spaces.Box(0, np.inf, (self._timestep,)),
                'TMIN': gym.spaces.Box(-np.inf, np.inf, (self._timestep,)),
                'TMAX': gym.spaces.Box(-np.inf, np.inf, (self._timestep,)),
                'VAP': gym.spaces.Box(0, np.inf, (self._timestep,)),
                'RAIN': gym.spaces.Box(0, np.inf, (self._timestep,)),
                'E0': gym.spaces.Box(0, np.inf, (self._timestep,)),
                'ES0': gym.spaces.Box(0, np.inf, (self._timestep,)),
                'ET0': gym.spaces.Box(0, np.inf, (self._timestep,)),
                'WIND': gym.spaces.Box(0, np.inf, (self._timestep,)),
            }
        )

    def _get_observation_space_crop_model(self) -> gym.spaces.Space:
        return gym.spaces.Dict(
            {var: gym.spaces.Box(0, np.inf, shape=(self._timestep,)) for var in self._output_variables}
        )

    def _get_action_space(self) -> gym.spaces.Space:
        return gym.spaces.Box(
                low=np.array([0, 0, 0, 0]),
                high=np.array([1e6, 1e6, 1e6, 1e6]),
                dtype=np.float32
            )
        # return gym.spaces.Dict(
        #     {
        #         'irrigation': gym.spaces.Box(0, np.inf, shape=()),
        #         'N': gym.spaces.Box(0, np.inf, shape=()),
        #         'P': gym.spaces.Box(0, np.inf, shape=()),
        #         'K': gym.spaces.Box(0, np.inf, shape=()),
        #     }
        # )
        # return gym.spaces.Box(low=np.array([0, 0, 0, 0]), high=np.array([250, 5e4, 5e4, 5e4]), shape=(4,))

    """
    Properties of the crop model config file
    """
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

    """
    Properties derived from the agro management config:
    """

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

    """
    Other properties
    """

    @property
    def date(self) -> datetime.date:
        return self._model.day
#endregion
    """
    Gym functions
    """

    def step(self, action) -> tuple:
        """
        Perform a single step in the Gym environment. The provided action is performed and the environment transitions
        from state s_t to s_t+1. Based on s_t+1 an observation and reward are generated.

        :param action: an action that respects the action space definition as described by `self._get_action_space()`
        :return: a 4-tuple containing
            - an observation that respects the observation space definition as described by `self._get_observation_space()`
            - a scalar reward
            - a boolean flag indicating whether the environment/simulation has ended
            - a dict containing extra info about the environment and state transition
        """
        # Create a dict for storing info
        info = dict() 

        # Apply action
        # if isinstance(action, np.ndarray):
        #     action = action[0]
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
        done = True if r-100<=5e-3 or len(output) >= self._max_steps else False#self._model.terminated
        if done:
            info['output_history'] = self._model.get_output()
            info['summary_output'] = self._model.get_summary_output()
            info['terminal_output'] = self._model.get_terminal_output()
        truncated = False
        terminated = done
        # Return all values
        return o, r, terminated, truncated, info

    def _apply_action(self, action):
        irrigation, N, P, K = action

        self._model._send_signal(signal=pcse.signals.irrigate,
                                 amount=irrigation,
                                 efficiency=0.8,
                                 )

        self._model._send_signal(signal=pcse.signals.apply_npk,
                                 N_amount=N,
                                 P_amount=P,
                                 K_amount=K,
                                 N_recovery=0.7,
                                 P_recovery=0.7,
                                 K_recovery=0.7,
                                 )

    def _get_observation(self, output) -> dict:
        # Get the datetime objects characterizing the specific days
        days = [day['day'] for day in output]
        # Get the output variables for each of the days
        crop_model_observation = {v: [day[v] for day in output] for v in self._output_variables}
        # Get the weather data of the passed days
        weather_data = [self._weather_data_provider(day) for day in days]
        # Cast the weather data into a dict
        weather_observation = {var: [getattr(weather_data[d], var) for d in range(len(days))] for var in
                               self._weather_variables}
        # Объедините значения в один словарь
        observation = {
            'crop_model': crop_model_observation,
            'weather': weather_observation,
        }
        # Преобразуйте словарь в одномерный массив (numpy.ndarray)
        flat_observation = {}
        for key, value in observation.items():
            if isinstance(value, dict):
                for subkey, subvalue in value.items():
                    flat_observation[f"{key}_{subkey}"] = np.array(subvalue)
            else:
                flat_observation[key] = np.array(value)
        return flat_observation
    def _get_reward(self) -> float:
        """
        Generate a reward based on the current environment state

        :param var: the variable extracted from the model output
        :return: a scalar reward. The default implementation gives the increase in yield during the last state transition
                 if the environment is in its initial state, the initial yield is returned
        """

        output = self._model.get_output()
        check_params = {
                key:0 for key in self.optimal_space.keys() 
            } #
        # print(output[-1].keys())
        for key in output[-1].keys():
            if isinstance(output[-1][key],dict): # crop_model
                # Получите ключи как множества
                keys1 = set(output[-1][key].keys())
                keys2 = set(check_params.keys())

                # Найдите пересечение ключей
                common_keys = keys1 & keys2

                # Преобразуйте множество ключей в список, если это необходимо
                common_keys_list = list(common_keys)
                for _key in common_keys_list:
                    check_params[_key] = output[-1][key][_key]#[-1] 
            else:
                # Получите ключи как множества
                keys1 = set(output[-1].keys())
                keys2 = set(check_params.keys())

                # Найдите пересечение ключей
                common_keys = keys1 & keys2

                # Преобразуйте множество ключей в список, если это необходимо
                common_keys_list = list(common_keys)
                for _key in common_keys_list:
                    # print(f'output[-1][{_key}] {output[-1][_key]}')
                    check_params[_key] = output[-1][_key]#[-1]
        # check_params fullfilled now
        
        def calc_loc_rew(state_params,param_name,best_r):
            return best_r  if self.optimal_space[param_name][0] <= state_params[param_name] <= self.optimal_space[param_name][1] \
                    else np.clip(1-(state_params[param_name] - np.mean(self.optimal_space[param_name]))**2 ,-1,1 )*best_r
            
        
        reward_dict = {}
        for key in check_params.keys():
            reward_dict[key] = calc_loc_rew(check_params,key,100)
        # print(f'Rewards:Mean(r) = {np.mean(np.array(list(reward_dict.values())))}; ({", ".join( [f"{key}: {value}" for key, value in reward_dict.items()])})')

        return np.mean(np.array(list(reward_dict.values())))
    
    def reset(self,
              *,
              seed: int = None,
              return_info: bool = False,
              options: dict = None
              ):
        """
        Reset the PCSE-Gym environment to its initial state

        :param seed:
        :param return_info: flag indicating whether an info dict should be returned
        :param options: optional dict containing options for reinitialization
        :return: depending on the `return_info` flag, an initial observation is returned or a two-tuple of the initial
                 observation and the info dict
        """

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

    def render(self, mode="human"):
        pass  # Nothing to see here
