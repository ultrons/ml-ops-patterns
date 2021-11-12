import os
import json

import torch
from ts.torch_handler.base_handler import BaseHandler

import copy
from pathlib import Path
import warnings

import numpy as np
import pandas as pd
import pytorch_lightning as pl
import torch

from pytorch_forecasting import TimeSeriesDataSet

import logging
logger = logging.getLogger(__name__)

class ForcastHandler(BaseHandler):
    def __init__(self):
        super(ForcastHandler, self).__init__()
        self.initialized = False
        
    def initialize(self, ctx):
        """ Loads the model.pt file and initialized the model object.
        Instantiates Tokenizer for preprocessor to use
        Loads labels to name mapping file for post-processing inference response
        """
        self.manifest = ctx.manifest

        properties = ctx.system_properties
        model_dir = properties.get("model_dir")
        self.device = torch.device("cuda:" + str(properties.get("gpu_id")) if torch.cuda.is_available() else "cpu")

        # Read model serialize/pt file
        serialized_file = self.manifest["model"]["serializedFile"]
        model_pt_path = os.path.join(model_dir, serialized_file)
        if not os.path.isfile(model_pt_path):
            raise RuntimeError("Missing the model.pt or pytorch_model.bin file")
        
        # Load model
        self.model = torch.load(model_pt_path)
        self.model.to(self.device)
        self.model.eval()
        logger.debug('Forecasting model from path {0} loaded successfully'.format(model_dir))
        

        self.initialized = True
        
    def preprocess(self, inputs):
        results = []
        for example in inputs:
            data = example['data']
            
            # Creating forecasting model expects either a dataloader or a timeseriese dataset
            # or a pandas dataframe
            # Eventual input to the model is dictionary of features
            max_prediction_length = 24
            max_encoder_length = 120
            data = pd.DataFrame.from_dict(data)

            # Adding time_idx, time_idx is NOT a feature for the model
            # And is only used in the dataset creation 
            #data["time_idx"] =  data["Date_Time"].dt.year*365*24 + data["Date_Time"].dt.dayofyear * 24 + data["Date_Time"].dt.hour
            #data["time_idx"] -= data["time_idx"].min()
            
            # Adding prediction length entries to all dataset creation
            # It requires at least encoder length + prediction length inputs
            data = pd.concat([data, data.tail(max_prediction_length)], ignore_index=True)
            data['time_idx'] = np.arange(data.shape[0])

            time_varying_known_reals = [
                'p__mbar',
                'Tpot__K',
                'Tdew__degC',
                'rh__percent',
                'VPmax__mbar',
                'VPact__mbar',
                'VPdef__mbar',
                'sh__g_per_kg',
                'H2OC__mmol_per_mol',
                'rho__gm_per_cubic_m',
                'wv__m_per_s',
                'max_w__vm_per_s',
                'wd__deg',
                'time_idx'
            ]
            inference_set = TimeSeriesDataSet(
                data,
                time_idx="time_idx",
                target="T__degC",
                group_ids=["series"],
                time_varying_unknown_reals=["T__degC"],
                time_varying_known_reals=time_varying_known_reals,
                max_encoder_length=max_encoder_length,
                max_prediction_length=max_prediction_length,
                min_encoder_length=max_encoder_length,
                min_prediction_length=max_prediction_length,
                add_relative_time_idx=True,
                add_target_scales=True,
                add_encoder_length=True,
                allow_missing_timesteps=True,
                randomize_length=False,
            )
            #inference_set = TimeSeriesDataSet.from_dataset(inference_set, data, predict=False, stop_randomization=True)
            results.append(inference_set)
        logger.debug('Done creating the inference set(s).')
        
        return results      

    def inference(self, inputs):
        results = []
        for inf_set in inputs:
            results.append(self.model.predict(inf_set))
        return torch.stack(results, dim=0)
    
    def postprocess(self, inputs):
        return inputs.tolist()
        #return inputs
