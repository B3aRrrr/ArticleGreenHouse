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
    'T_soil',
    'humid_soil',
    'pH','EC',
    'NAVAIL','PAVAIL','KAVAIL',
    'pH_mix','EC_mix',
    'N','P','K'
    ]
low = np.array([
            15, # T_soil [°C]
            .4, # humid_soil [%]
            # 3, # pH
            # 100,# EC [μS/cm]
            10,#43.27, # [kg/ha] 10, # NAVAIL [ppm] 
            10,#8.654, # [kg/ha] 2, # PAVAIL [ppm]
            10,#216.35, # [kg/ha] 50, # KAVAIL [ppm]
            # MIXTURE
            # 5, # pH_mix
            # 0,# EC_mix [μS/cm]
            10, # [kg/ha] N [ppm]
            10, # [kg/ha] P [ppm]
            10] # [kg/ha] K [ppm]
    
        ).astype(np.float32)
high = np.array(
    [
        # SOIL
        30, # # T_soil [°C]
        1, # humid_soil [%]
        # 9, # pH
        # 3000,# EC [μS/cm]
        400,#21_635, # [kg/ha] 5000, # NAVAIL [ppm] NAVAIL(kg/га) = 5000 * 0.4327 * 10 = 21,635 кг/га
        400,#4_327, # [kg/ha] 1000, # PAVAIL [ppm]
        400,#4_327, # [kg/ha] 1000, # KAVAIL [ppm]
        # MIXTURE
        # 9, # pH_mix
        # 3000,# EC_mix [μS/cm]
        400,#216_350 , # [kg/ha] 50_000, # N [ppm]
        400,#108_175 , # [kg/ha] 25_000, # P [ppm]
        400,#216_350 , # [kg/ha] 50_000, # K [ppm]
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
    'seed_sowing': {
        'T_soil': np.array([20, 25]),  # T [°C]
        'humid_soil': np.array([0.5, 0.6]),  # phi [%]
        # 'pH': np.array([6.0, 6.5]),  # pH
        # 'EC': np.array([0.2, 0.5]),  # EC/TDS [mS/cm]
        # 'NAVAIL': np.array([40, 120]),  # N [кг/га]
        # 'PAVAIL': np.array([10, 40]),  # P [кг/га]
        # 'KAVAIL': np.array([60, 200]),  # K [кг/га]
        # 'vel_assim': np.array([0.2, 0.5])  # л/час [4.8, ]
    },
    'seedling': {
        'T_soil': np.array([18, 24]),  # T [°C]
        'humid_soil': np.array([0.6, 0.7]),  # phi [%]
        # 'pH': np.array([6.0, 6.5]),  # pH
        # 'EC': np.array([0.2, 0.8]),  # EC/TDS [mS/cm]
        # 'NAVAIL': np.array([60, 150]),  # N [кг/га]
        # 'PAVAIL': np.array([15, 45]),  # P [кг/га]
        # 'KAVAIL': np.array([70, 220]),  # K [кг/га]
        # 'vel_assim': np.array([0.5, 2.0])  # л/час
    },
    'flowering_fruiting': {
        'T_soil': np.array([21, 24]),  # T [°C]
        'humid_soil': np.array([0.7, 0.8]),  # phi [%]
        # 'pH': np.array([6.0, 6.5]),  # pH
        # 'NAVAIL': np.array([70, 180]),  # N [кг/га]
        # 'PAVAIL': np.array([20, 60]),  # P [кг/га]
        # 'KAVAIL': np.array([80, 240]),  # K [кг/га]
        # 'EC': np.array([1.0, 2.0]),  # EC/TDS [mS/cm]
        # 'vel_assim': np.array([1, 3])  # л/час
    },
    'fruit_ripening': {
        'T_soil': np.array([20, 23]),  # T [°C]
        'humid_soil': np.array([0.6, 0.7]),  # phi [%]
        # 'pH': np.array([6.0, 6.5]),  # pH
        # 'EC': np.array([0.8, 1.5]),  # EC/TDS [mS/cm]
        # 'NAVAIL': np.array([50, 130]),  # N [кг/га]
        # 'PAVAIL': np.array([12, 42]),  # P [кг/га]
        # 'KAVAIL': np.array([65, 210]),  # K [кг/га]
        # 'vel_assim': np.array([1.5, 4.0])  # л/час
    }
}
#endregion
SOIL_DENSITY = np.mean(np.array([1.2,1.5])) # g/cm^3 -> 1000 kg/m^3

class GreenHousePCSEEnv(gym.Env):
    name = 'GreenHousePCSE'
    _additional_variables = [
        'T_soil',
        'humid_soil',
        # 'NAVAIL','PAVAIL','KAVAIL',
        # 'pH',
        # 'EC',
        # 'pH_mix',
        # 'EC_mix'
        ]
    
    def __init__(self,
                 model_config: str = _DEFAULT_CONFIG,
                 agro_config: str = _DEFAULT_AGRO_FILE_PATH,
                 growth_stage:str='seed_sowing',
                #  growth_fruiting_vents:list=[2,2],
                 crop_parameters=_DEFAULT_CROP_FILE_PATH,
                 site_parameters=_DEFAULT_SITE_FILE_PATH,
                 soil_parameters=_DEFAULT_SOIL_FILE_PATH,
                 soil_density=SOIL_DENSITY,
                 R_cylinder:float=0.35,
                 h_cylinder:float=0.35,
                 V:float=8.5,
                 years=None,
                 location=None, 
                 seed: int = None, # seed
                 timestep: int = 1, # day
                 maxsteps:int=72,
                 **kwargs
                 ):
        assert growth_stage in list(_stages.keys())
       
        super().reset(seed=seed)
        assert timestep > 0
        
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
        
        # VALVES (actions [0,1]) 
        self.optimal_space = _stages[growth_stage]
        self.R = R_cylinder*10
        self.h = h_cylinder*10
        self.V = V # in barrell
        self.V_soil = np.pi * (self.R*10)**2 * (10*self.h) # litres
        
        self.soil_density = soil_density * 1e3 # kg/m3
        # Create  a emperature humid EC and pH model
        self._saved_days = self._init_sec_model()
                        
        self.day_counts = 0
        self.maxsteps=maxsteps

    def _init_sec_model(self) -> list:
        init_dict = {
                var:self._get_default_space(var,1).sample() for var in self.__class__._additional_variables
            }
        answer = [{'day':day['day']} for day in self._model.get_output()[-self._timestep:]]
        
        for i  in  range(len(answer)):
            for key in init_dict.keys():
                answer[i][key] = init_dict.copy()[key][0]
        
        return answer
        
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
        observation_space = {
                **self._get_observation_space_crop_model(),
                **self._get_observation_space_weather(),
        }
        
        for var in self.__class__._additional_variables:
            observation_space [var] = self._get_default_space(var, self._timestep)
        space = gym.spaces.Dict(observation_space) 
        return space

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
           np.zeros(2),# n p k irig T_air humid_air
           np.ones(2), 
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
            var:self._get_observation(self._model.get_output())[var][-1] for var in self.__class__._additional_variables
        } 
        
        current_args = self._apply_action(action,current_args)
        # Run the crop growth model
        self._model.run(days=self._timestep)
        # Get the model output
        self._update_saved_days(current_args)
        output = self._model.get_output()[-self._timestep:]
        # Update self._saved_days
        info['days'] = [day['day'] for day in output]
        # Construct an observation and reward from the new environment state
        o = self._get_observation(output)
        r = self._get_reward()
        # Check whether the environment has terminated
        done = True \
            if self.day_counts > self.maxsteps or \
                np.abs(r - 100) <= .0005  \
            else False 
        if done:
            info['output_history'] = self._model.get_output()
            info['summary_output'] = self._model.get_summary_output()
            info['terminal_output'] = self._model.get_terminal_output()
        truncated = False
        terminated = done
        # Return all values
        # print(f'[GreenHousePCSEEnv [l_474]] info = {info.keys()}')
        return o, r, terminated, truncated, info
    def _update_saved_days(self,current_args:dict)-> list:
        init_dict = current_args
        answer = [{'day':day['day']} for day in self._model.get_output()[-self._timestep:]]
        # print(f'[GreenHousePCSEEnv [l478]] {self._model.get_output()}')
        
        for i  in  range(len(answer)):
            for key in init_dict.keys():
                answer[i][key] = init_dict.copy()[key]
        self._saved_days = self._saved_days + answer
        # print(f'[GreenHousePCSEEnv [l484]] {self._saved_days}')
        # Отсортируйте объединенный список по значению "day" в каждом словаре
        self._saved_days = sorted(self._saved_days, key=lambda x: x['day'])     
    def _apply_action(self, action,current_args):
        # get valve vars
        # valve_N,valve_P,valve_K,valve_pH,valve_EC = action[:5]     
        # valve_N,valve_P,valve_K,valve_irig = action[:4]    
        T_air_raw,humid_air_raw = action[:] 
        # T_air_raw,humid_air_raw = action[5:]
        # Air
        T_air,humid_air = 16 + 14 * T_air_raw, humid_air_raw
        # _pH = 14*valve_pH
        # _EC = 270*valve_EC # сульфат магния
        
        # Calcs irrigation N,P, K
        #region 1. Calcs in volume of water
        # N = valve_irig*self.V*24#V_i(dt=self._timestep*24,a_i=valve_irig,U=self.flow_in) # l
        # mixture= {
        #         "N": N*valve_N * 1.5/(0.0001*(np.pi*self.R**2)), # kg/ha
        #         'P':N*valve_P * 0.2/(0.0001*(np.pi*self.R**2)), # kg/ha
        #         'K': N*valve_K * 0.2/(0.0001*(np.pi*self.R**2)), # kg/ha
        #         'irrigation':N*1000/(1e4*np.pi*self.R**2) # cm
        #     }
        # #endregion
        # #region 2. Calcs N,P,K, irragtion
        # # N_out =  V_i(dt=self._timestep*24,a_i=valve_out,U=self.flow_out +  np.random.uniform(self.optimal_space['vel_assim'][0],self.optimal_space['vel_assim'][1]))
       
        # self._model._send_signal(signal=pcse.signals.irrigate,
        #                          amount=mixture['irrigation'] ,
        #                          efficiency=np.random.uniform(0.68,.72),
        #                          )
        # self._model._send_signal(signal=pcse.signals.apply_npk,
        #                          N_amount=mixture['N'] ,
        #                          P_amount=mixture['P'] ,
        #                          K_amount=mixture['K'] ,
        #                          N_recovery=np.random.uniform(0.68,.72),
        #                          P_recovery=np.random.uniform(0.68,.72),
        #                          K_recovery=np.random.uniform(0.68,.72),
        #                          )
        #endregion
        #region 3. Apply EC 
        # EC_final = (EC_soil * V_soil + EC_mix * V_mix) / (V_soil + V_mix)
        # current_args['EC'] = np.clip((current_args['EC'] * self.V_soil + _EC * self.V)/( self.V_soil+self.V),0,270)
        # # ΔpH = (pH_mix - pH_soil) × (V_mix / V_soil)
        # current_args['pH'] = np.clip(current_args['pH'] + (_pH  - current_args['pH']) * (self.V/self.V_soil),0,14)
        #endregion
        #region 4. Apply Temp and humid
        current_args['T_soil'] = temp_soil(
            dt=1/24*self._timestep, 
            T_soil_0=current_args['T_soil'], 
            T_air=T_air
            )
        def diff_humid_time(theta_t0,t, theta_air, V_soil, N, E):
            # Параметры:
            # t - текущее время (в сутках)
            # theta_air - влажность воздуха
            # V_soil - объем почвы в горшке (в литрах)
            # N - объем добавленной жидкости (в литрах)
            # E - естественное испарение жидкости из почвы (в литрах в сутки)

            # Влажность почвы в начальный момент времени (может быть равной начальной влажности)
            theta_initial = theta_t0  # или устанавливаете начальное значение почвы

            # Коэффициент убывания влажности (в данном случае, можно настроить для желаемой скорости изменения)
            k = 0.1  # Примерный коэффициент

            # Вычисляем изменение влажности с течением времени (экспоненциальное убывание) с учетом E
            delta_theta = (theta_air - theta_initial) * (1 - np.exp(-k * t)) - E * t

            # Обновляем влажность почвы
            updated_theta_t0 = theta_initial + delta_theta

            # Учтем также добавленную жидкость N
            updated_theta_t0 += N / V_soil

            return updated_theta_t0
        current_args['humid_soil'] =  diff_humid_time(
            theta_t0=current_args['humid_soil'],
            t=1/24*self._timestep,
            theta_air=humid_air,
            V_soil=1e3*self.h*np.pi*self.R**2,
            N=8.5,#N/24,
            E =np.random.uniform(0.001,0.01)
            )
        #endregion
        return current_args
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
        
        # additional vars
        additional_vars = {key:[] for key in self._saved_days[0].keys() if key != 'day'}
        for i  in range(len(self._saved_days)-self._timestep,len(self._saved_days)):
            for key in self._saved_days[i].keys():
                if key != 'day':
                    additional_vars[key].append( self._saved_days[i][key])

        o = {
            **crop_model_observation,
            **weather_observation,
            **additional_vars
        }

        return o 
    def _get_reward(self) -> dict:
        output = self._get_observation(self._model.get_output())
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
        # reward_dict['dV'] = -100 if abs(self.dV ) > 8.5 else np.clip(
        #     1-(self.dV/8.5)**2,# np.exp(-(self.dV/8.5)**2), 
        #     -1, 
        #     1) * 100
        # print(f'Rewards:Mean(r) = {np.mean(np.array(list(reward_dict.values())))}; ({", ".join( [f"{key}: {value}" for key, value in reward_dict.items()])})')
        # # Создайте массив NumPy
        return np.mean(np.array(list(reward_dict.values())))
    def reset(self,
              *,
              seed: int = None,
              return_info: bool = False,
              options: dict = None
              ):
    #region old
        # Optionally set the seed
        # super().reset(seed=seed)
        # # Create an info dict
        # info = dict()
        # # Create a PCSE engine / crop growth model
        # self._model = self._init_pcse_model()
        # # Create  T humid EC  
        # self._saved_days = self._init_sec_model() # [{day,T,humid,EC,pH}]
        # output = self._model.get_output()[-self._timestep:]
        
        # o = self._get_observation(output)
        # info['date'] = self.date

        # return o, info if return_info else o
        #endregion
        #region new
        super().reset(seed=seed)
        info = dict()
        # Create a PCSE engine / crop growth model
        self._model = self._init_pcse_model()
        self._saved_days = self._init_sec_model() 
        # Создайте словарь, аналогичный выводу env._get_observation_space().sample()
        observation_space = self._get_observation_space().sample()
        return observation_space, info if return_info else observation_space
        #endregion
    def render(self, mode="human"):
        pass  # Nothing to see here
