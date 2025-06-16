from flask import Flask, request, jsonify
import torch
import threading
import logging
import numpy as np
import quaternion
import json
from matplotlib import pyplot as plt
from matplotlib.animation import FuncAnimation
import csv
import pandas as pd
from server.utils.data_util import *
from server.cache.datacache import *

logging.basicConfig(filename='server/log_txt/run_log.txt', level=logging.DEBUG)


app = Flask(__name__)
        
dir = {}
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
network = get_model('output/checkpoints/checkpoint_latest.pt')

with open('preds_all.csv', 'w',newline='') as f:
    writer = csv.writer(f)
    writer.writerow(['x', 'y'])
    
def com_data(data):
    id      = data['id']
    acce    = np.array(data['acce'])
    gyro    = np.array(data['gyro'])
    game_rv = np.array(data['game_rv'])
    uwb_flag = False
    if data['uwb'] is not None:
        uwb = np.array(data['uwb'])
    
    acce[:,0]    = acce[:,0]/1e09
    gyro[:,0]    = gyro[:,0]/1e09
    game_rv[:,0] = game_rv[:,0]/1e09
        
    
    # 处理对齐数据
    all_sources = {'acce':acce,'gyro':gyro,'game_rv':game_rv}
    
    output_time = compute_output_time(all_sources)
    
    
    # print(f'output_time:{len(output_time)}')
    
    source_vector = {'gyro', 'acce'}
    source_quaternion = {'game_rv'}
    source_all = source_vector.union(source_quaternion)
    
    processed_sources = {}
    
    for source in all_sources.keys():
        if source in source_vector:
            processed_sources[source] = process_data_source(all_sources[source], output_time, 'vector')
        else:
            processed_sources[source] = process_data_source(
                all_sources[source][:, [0, 4, 1, 2, 3]], output_time, 'quaternion')
            
            
    # print(f'processed_sources["acce"] {len(processed_sources["acce"])}')
    # print(f'processed_sources["gyro"] {len(processed_sources["gyro"])}')
    # print(f'processed_sources["game_rv"] {len(processed_sources["game_rv"])}')
    # 添加进cache
    cache = dir.get(id,DataCache(id))
    dir[data['id']] = cache
    
    
    with cache.lock:
        for row in processed_sources['acce']:
            cache.acce.append(row)
        
        for row in processed_sources['gyro']:
            cache.gyro.append(row)
            
        for row in processed_sources['game_rv']:
            cache.game_rv.append(row)
        
        # print(f'cache.acce {len(cache.acce)}')
        # print(f'cache.gyro {len(cache.gyro)}')
        # print(f'cache.game_rv {len(cache.game_rv)}')
        count = 0
        while len(cache.acce) >=200:
            
            temp_acce = np.array(cache.acce[0: 200])
            temp_gyro = np.array(cache.gyro[0: 200])
            temp_game_rv = np.array(cache.game_rv[0: 200])
            
            ori_q     = quaternion.from_float_array(temp_game_rv)
            gyro_q    = quaternion.from_float_array(np.concatenate([np.zeros([temp_gyro.shape[0], 1]), temp_gyro], axis=1))
            acce_q    = quaternion.from_float_array(np.concatenate([np.zeros([temp_acce.shape[0], 1]), temp_acce], axis=1))
            glob_gyro = quaternion.as_float_array(ori_q * gyro_q * ori_q.conj())[:, 1:]
            glob_acce = quaternion.as_float_array(ori_q * acce_q * ori_q.conj())[:, 1:]
            
            features = np.concatenate([glob_gyro, glob_acce], axis=1,dtype=np.float32).T
            
            features = torch.from_numpy(features).unsqueeze(0) 


            with torch.no_grad(): 
                # print(features.shape)
                pred = network(features.to(device)).cpu().detach().numpy()
                
            if count == 130:
                uwb_pos = [0,-2]
                cache.recon_traj_with_preds(uwb_pos)
            
            if count == 200:
                uwb_pos = [4,-2]
                cache.recon_traj_with_preds(uwb_pos)
            if count == 300:
                uwb_pos = [4,3]
                cache.recon_traj_with_preds(uwb_pos)
            
            count+=1

            with open('preds_all.csv', 'a',newline='') as f:
                writer = csv.writer(f)
                writer.writerow([pred[0][0],pred[0][1]])

            cache.preds_all.append(pred)

            cache.acce = cache.acce[10:]
            cache.gyro = cache.gyro[10:]
            cache.game_rv = cache.game_rv[10:]
            dir[data['id']] = cache
        # cache.recon_traj_with_preds(uwb_pos)
        cache.recon_traj_with_preds(uwb_pos)
        cache.save_csv()
        
        dir[data['id']] = cache

def test():
    with open('20241125014657R_WiFi_SfM (1)/acce.txt', 'r', encoding='utf-8') as file:
        acce_content = file.read()
    with open('20241125014657R_WiFi_SfM (1)/gyro.txt', 'r', encoding='utf-8') as file:
        gyro_content = file.read()
    with open('20241125014657R_WiFi_SfM (1)/game_rv.txt', 'r', encoding='utf-8') as file:
        game_rv_content = file.read()
    acce_list = []
    gyro_list = []
    game_rv_list = []
    uwb_list = None
    for line in acce_content.splitlines():
        acce_list.append(list(map(float,line.split(' ')[:4])))
    for line in gyro_content.splitlines():
        gyro_list.append(list(map(float,line.split(' ')[:4])))
    for line in game_rv_content.splitlines():
        game_rv_list.append(list(map(float,line.split(' ')[:5])))
    

    
    com_data({'id':123,'acce': acce_list, 'gyro': gyro_list, 'game_rv': game_rv_list,'uwb': uwb_list})

@app.route('/prediction', methods=['POST'])
def echo():
    
    data = request.get_json()
    
    if data is None:
        return jsonify({'error': 'No data received'}), 400
    
    com_data(data)
            
    return jsonify({'success': 'The data has been accepted'}), 200

if __name__ == '__main__':
    # app.run(host='0.0.0.0',debug=True,threaded=True,port=45535)
    test()
