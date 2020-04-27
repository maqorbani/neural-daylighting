#
#
#  Copyright (c) 2020.  Yue Liu
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#      https://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.
#
#  If you find this code useful please cite:
#  Predicting Annual Equirectangular Panoramic Luminance Maps Using Deep Neural Networks,
#  Yue Liu, Alex Colburn and and Mehlika Inanici.  16th IBPSA International Conference and Exhibition, Building Simulation 2019.
#
#
#

import os

os.environ["CUDA_VISIBLE_DEVICES"] = str(0)

import deep_light
from deep_light.predictLumMaps import train_model, analyze_model, generate_train_val_data, relative_err
from deep_light.random_selection_threeInputs import select_test_samples
from deep_light.genData import get_data_path, set_data_path


def main():

    data_root='./ALL_DATA_FP16'
    kwargs = {'data_root': data_root,
              'RETRAIN': False,
              'LOSS_FUNCTION_TYPE': relative_err,
              'LOG_BOOL': True,
              'SKYMAP_BOOL': False,
              'MODEL_TYPE': "Dense",
              'GAMMA_VALUE': 1.5,
              'LOG_BASE_VALUE': 10,
              'NUM_CLUSTERS': 250,
              'LATITUDE': 47,
              'LONGITUDE': 122,
              'SM': 120}

    
    select_test_samples(data_root)
    generate_train_val_data(**kwargs)
    train_model(**kwargs)
    analyze_model(**kwargs)


if __name__ == "__main__":
    main()