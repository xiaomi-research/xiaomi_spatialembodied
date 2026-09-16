# UPDATE: Replace placeholder paths with your actual paths.
import pickle
import os

# data_dict_pkl = 'data/OmniDrive/data_nusc/data_dict_sample.pkl'

# data_dict_infos = pickle.load(open(data_dict_pkl, 'rb'))
# info_pkl = 'data/OmniDrive/omni_pkls/nuscenes_infos_val.pkl'
# agentdriver_info_pkl = "data/DriveLMM-o1-main/data/tool_results/val/000681a060c04755a1537cf83b53ba57.pkl"
# info_pkl = agentdriver_info_pkl

info_pkl = "data/OmniDrive/omni_pkls/nuscenes_ego_infos_val.pkl"
# eval_pkl = "data/OmniDrive/data_nusc/eval_cf/0a0d6b8c2e884134a3b48df43d54c36a.pkl"
infos = pickle.load(open(info_pkl, 'rb'))
if isinstance(infos, dict):
    print(infos.keys())
    print(infos['data_list'][0:2])
else:
    print(infos[0:2])
