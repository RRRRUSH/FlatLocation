# -*- coding: GBK -*-
import os
import tkinter as tk
from tkinter import filedialog
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import math
import re
# ����������֧������
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False


def get_colors(num_colors=36):
    """
    ����һ�����Բ�ͬ����ɫ��
    
    ����:
        num_colors (int): ��Ҫ���ɵ���ɫ������
        
    ����:
        list: ����RGB��ʽ��ɫֵ���б�
    """
    np.random.seed(0) # �̶���������Ա����ֽ��
    colors = []
    for i in range(num_colors):
        hue = (360.0 / num_colors) * i # ���ȷֲ���ɫ����
        saturation = np.random.uniform(0.4, 1.0) # �����ƫ��߱��Ͷ�
        lightness = np.random.uniform(0.4, 0.8) # �����ƫ���е�����
        
        # ��HSLת��ΪRGB
        temp1 = lightness * (1 + saturation) if lightness < 0.5 else (lightness + saturation) - (lightness * saturation)
        temp2 = 2 * lightness - temp1
        def hsl_to_rgb(v):
            if v < 0: v += 1
            if v > 1: v -= 1
            if v * 6 < 1: return temp2 + (temp1 - temp2) * 6 * v
            if v * 2 < 1: return temp1
            if v * 3 < 2: return temp2 + (temp1 - temp2) * ((2/3) - v) * 6
            return temp2
            
        rgb = [hsl_to_rgb(hue / 360.0 + x) for x in (-1/3, 1/3, 0)]
        colors.append(tuple(rgb))
    np.random.shuffle(colors) # ����˳�������Ӷ�����
    return colors

def remove_duplicates(s):
    return ''.join(sorted(set(s), key=s.index))
def convert_to_system_coordinates(antenna_number, antenna_angle):
    # ����ÿ������0����ϵͳ����ϵ�е���ʼ�Ƕ�
    offsets = {
        0: 0,
        1: 0,
        2: 0,
        3: 0
    }
    
    # ��ȡ��Ӧ���ߵ�ƫ����
    offset = offsets.get(antenna_number, None)
    
    if offset is None:
        raise ValueError("Invalid antenna number. Must be 1, 2, 3, or 4.")
    
    # ����ϵͳ����ϵ�еĽǶȣ�ע�ⷽ��ת��
    system_angle = (offset - antenna_angle) % 360
    
    return system_angle

def calculate_antenna_coordinate(system_coordinate, antenna_index):
    """
    ����ϵͳ��������������ꡣ
    
    :param system_coordinate: ϵͳ���꣨0��360�ȣ�
    :param antenna_index: ���߱�ţ���1��ʼ��
    :param total_antennas: ����������
    :return: ��������
    """
    base_angle = 360 / 4
    antenna_coordinate = ( -system_coordinate + (antenna_index ) * base_angle) % 360
    return antenna_coordinate

def merge_txt_files(folder_path, folder_name):
    file_names = sorted(
        [f for f in os.listdir(folder_path) if f.endswith('.txt')],
        key=lambda x: int(x.split('.')[0])
    )
    print(file_names)
    output_folder = os.path.join(folder_path, '�ϲ����')
    os.makedirs(output_folder, exist_ok=True)

    output_file = os.path.join(output_folder, f'{folder_name}�����߽Ƕ����ݣ�����������180��.txt')

    with open(output_file, 'w', encoding='utf-8') as outfile:
        for file_name in file_names:
            file_path = os.path.join(folder_path, file_name)
            with open(file_path, 'r', encoding='latin-1') as infile:
                outfile.write(infile.read())
                outfile.write('\n')

    return output_folder, output_file

def process_merged_file(output_folder, file_path, folder_name):
    base_name = f"{folder_name}������"
    output_file_in = os.path.join(output_folder, f'{base_name}�Ƕȣ���180�ڣ�.txt')
    output_file_outer = os.path.join(output_folder, f'{base_name}�Ƕȣ���180�⣩.txt')

    in_data = []
    outer_data = []

    with open(file_path, 'r', encoding='utf-8') as infile:
        for line in infile:
            values = line.strip().split(',')
            if len(values) < 3:
                continue
            try:
                pdoa = float(values[2])
                if abs(pdoa) < 180:
                    pdoa_rad = pdoa * math.pi / 180.0
                    angle = round(math.degrees(math.asin(pdoa_rad / math.pi)), 2)
                    values.append(f"{angle:.2f}")
                    in_data.append(','.join(values) + '\n')
                else:
                    outer_data.append(','.join(values) + '\n')
            except ValueError:
                continue

    with open(output_file_in, 'w', encoding='utf-8') as outfile_in:
        outfile_in.writelines(in_data)

    with open(output_file_outer, 'w', encoding='utf-8') as outfile_outer:
        outfile_outer.writelines(outer_data)

    return output_file_in, output_file_outer

def plot_angle(file_path, folder_name):
    data = pd.read_csv(file_path, header=None)
    excel_file = file_path.replace('.txt', '.xlsx')
    data.to_excel(excel_file, index=False, header=False)

    plt.figure(figsize=(30, 18))
    plt.scatter(data.index, data[3])
    plt.ylim(-90, 90)
    plt.yticks(range(-90, 95, 5))
    plt.title(f'�Ƕ�ɢ��ͼ\n{folder_name}������')
    plt.xlabel('����')
    plt.ylabel('�Ƕ� (��)')
    plt.grid(True)

    plot_file = file_path.replace('.txt', '�Ƕ�ɢ��ͼ.png')
    plt.savefig(plot_file)
    plt.close()

def plot_pdoa(file_path, folder_name):
    data = pd.read_csv(file_path, header=None)
    excel_file = file_path.replace('.txt', '.xlsx')
    data.to_excel(excel_file, index=False, header=False)

    plt.figure(figsize=(30, 18))
    plt.scatter(data.index, data[2])
    plt.ylim(-360, 360)
    plt.yticks(range(-360, 370, 10))
    plt.title(f'PDOAɢ��ͼ\n{folder_name}������')
    plt.xlabel('����')
    plt.ylabel('PDOA')
    plt.grid(True)

    plot_file = file_path.replace('.txt', 'PDOAɢ��ͼ.png')
    
    plt.savefig(plot_file)
    plt.close()

def extract_number_from_filename(filename):
    match = re.search(r'-?\d+', filename)
    return int(match.group()) if match else None

def calculate_system_coordinates(file_names, folder_index):
    system_coordinates = {}
    for file_name in enumerate(file_names):
        angle = int(extract_number_from_filename(file_name[1]))
        adjusted_angle = convert_to_system_coordinates(folder_index,angle)
        system_coordinates[file_name[1]] = adjusted_angle
    return system_coordinates

def extract_folder_index(folder_name):
    match = re.search(r'\d+', folder_name)
    return int(match.group()) 

def plot_angle_data(folder_path, folder_name):
    folder_index = extract_folder_index(folder_name)
    ant_id = [0,1,2,3]
    pos_neg_ant_id = ant_id[folder_index]
    print(">>>>>",pos_neg_ant_id)
    file_names = sorted(
        [f for f in os.listdir(folder_path) if f.endswith('.txt')],
        key=lambda x: int(os.path.splitext(x)[0])
    )
    print(file_names)

    #system_coordinates = calculate_system_coordinates(file_names, folder_index)

    #print(system_coordinates)

    x_ticks = []
    x_labels = []

    x_position = 0
    colors = get_colors(len(file_names))

    plt.figure(figsize=(30, 18))
    pos_angle_y = [[]]
    neg_angle_y = [[]]
    x_all_labels = [[]]

    for color, file_name in zip(colors, file_names):
        file_path = os.path.join(folder_path, file_name)
        
        try: 
            data = pd.read_csv(file_path, header=None, on_bad_lines='skip')
        except pd.errors.ParserError as e:
            print(f"Error reading {file_name}: {e}")
            continue

        data[2] = data[2].fillna(500.0)

        angle_x = []
        angle_y = []
        for index, row in data.iterrows():
            try:
                pdoa = float(row[2])
                angle = float(row[3])
                if abs(pdoa) < 180:
                    angle_x.append(x_position + index)
                    angle_y.append(angle)
            except ValueError:
                print(f"Skipping invalid data in file {file_name} at row {index}")
        if angle_x and angle_y:         
            l_pos_angle_y = []
            l_neg_angle_y = []
            for i in range(len(angle_y)):
                l_pos_angle_y.append( (angle_y[i] + pos_neg_ant_id * (360 / 4) ) % 360)
                l_neg_angle_y.append( (angle_y[i] + 180 + pos_neg_ant_id * (360 / 4)) % 360)
            pos_angle_y.append(l_pos_angle_y)
            neg_angle_y.append(l_neg_angle_y)
        if angle_x and angle_y:
            x_ant_labels = int(calculate_antenna_coordinate(int(os.path.splitext(file_name)[0]),pos_neg_ant_id))
            if(x_ant_labels > 90 and x_ant_labels < 270):
                x_ant_labels = '����' + str(180 - x_ant_labels) + '��'
            elif x_ant_labels < 360 and x_ant_labels >= 270:
                x_ant_labels = x_ant_labels - 360
            #plt.scatter(angle_x, angle_y, color=color)
            #x_ticks.append(x_position + len(data) / 2)
            #x_labels.append(f"{x_ant_labels}\n({system_coordinates[file_name]:.0f}��)")
            x_all_labels.append(f"{x_ant_labels}\n({int(os.path.splitext(file_name)[0])}��)")
            print(f"�������꣺{x_ant_labels},�豸���꣺{int(os.path.splitext(file_name)[0])},���ļ���{file_name}")
        else:
            print(f"No valid data in file {file_name}")

        x_position += len(data) + 150

    pos_angle_y.remove([])
    neg_angle_y.remove([])
    x_all_labels.remove([])

    print(len(pos_angle_y))
    print(len(neg_angle_y))

    sorted_indices = sorted(range(len(x_all_labels)), key=lambda i: int(x_all_labels[i].split('(')[1].replace('��)', '')))
    x_all_labels_sorted = [x_all_labels[i] for i in sorted_indices]

    # Reorder plot_all_y according to sorted indices
    pos_angle_y_sorted = [pos_angle_y[i] for i in sorted_indices]
    pos_angle_y = pos_angle_y_sorted
    
    neg_angle_y_sorted = [neg_angle_y[i] for i in sorted_indices]
    neg_angle_y = neg_angle_y_sorted

    x_all_labels = x_all_labels_sorted

    x_position = 0
    pos_angle_max_len = len(pos_angle_y[0])
    for index in range(len(pos_angle_y)):
        if(len(pos_angle_y[index]) > pos_angle_max_len):
            pos_angle_max_len = len(pos_angle_y[index])
    for color,pangle_y,nangle_y,x_labels1 in zip(colors, pos_angle_y,neg_angle_y,x_all_labels):
        angle_x = []
        #print(pangle_y)
        for index in range(len(pangle_y)):
            angle_x.append(x_position + index)
        #print(angle_x)
        #print(len(pangle_y))
        plt.scatter(angle_x, pangle_y, color=color)
        plt.scatter(angle_x, nangle_y, color=color)
        x_ticks.append(x_position + len(pangle_y) / 2)
        x_labels.append(f"{x_labels1}")
        x_position += pos_angle_max_len + 200
        
    plt.ylim(0, 360)
    plt.yticks(range(0, 365, 5))
    plt.xticks(x_ticks, x_labels, rotation=0, multialignment='center')
    plt.title(f'�Ƕ�ɢ��ͼ\n{folder_name}������')
    plt.xlabel('��������ϵ����90�㣩\n�豸����ϵ��360�㣩\n��ʵ�Ƕ�')
    plt.ylabel('�������Ƕ� (��)')
    plt.grid(True)

    output_folder = os.path.join(folder_path, '�ϲ����')
    os.makedirs(output_folder, exist_ok=True)
    plt.savefig(os.path.join(output_folder, f'{folder_name}������_�Ƕ�ɢ��ͼ(PDOA��180��).png'))
    plt.close()

    print("�Ƕ�ͼ�ѱ������ļ���")

def plot_pdoa_data(folder_path, folder_name):
    folder_index = extract_folder_index(folder_name)
    
    file_names = sorted(
        [f for f in os.listdir(folder_path) if f.endswith('.txt')],
        key=lambda x: int(os.path.splitext(x)[0])
    )
    print(file_names)

    #system_coordinates = calculate_system_coordinates(file_names, folder_index)
    

    x_ticks = []
    x_labels = []

    x_position = 0
    colors = get_colors(len(file_names))

    #print(colors)
    plt.figure(figsize=(50, 30))

    plot_all_y = [[]]
    x_all_labels = [[]]

    for color, file_name in zip(colors, file_names):
        file_path = os.path.join(folder_path, file_name)

        try:
            data = pd.read_csv(file_path, header=None, on_bad_lines='skip')
        except pd.errors.ParserError as e:
            print(f"Error reading {file_name}: {e}")
            continue

        pdoa_x = []
        pdoa_y = []
        
        for index, row in data.iterrows():
            pdoa_x.append(x_position + index)
            pdoa_y.append(row[2])

        #plot_all_x.append(pdoa_x)
        plot_all_y.append(pdoa_y)



        if pdoa_x and pdoa_y:
            x_ant_labels = int(calculate_antenna_coordinate(int(os.path.splitext(file_name)[0]),folder_index))
            if(x_ant_labels > 90 and x_ant_labels < 270):
                x_ant_labels = '����' + str(180 - x_ant_labels) + '��'
            elif x_ant_labels < 360 and x_ant_labels >= 270:
                x_ant_labels = x_ant_labels - 360
            #print(pdoa_x)
            #print(pdoa_y)
            #plt.scatter(pdoa_x, pdoa_y, color=color)
            #x_ticks.append(x_position + len(data) / 2)
            #x_labels.append(f"{x_ant_labels}\n({system_coordinates[file_name]:.0f}��)")
            x_all_labels.append(f"{x_ant_labels}\n({int(os.path.splitext(file_name)[0])}��)")
            print(f"�������꣺{x_ant_labels},�豸���꣺{int(os.path.splitext(file_name)[0])},���ļ���{file_name}")
        else:
            print(f"No valid data in file {file_name}")

        x_position += len(data) + 150
    #plot_all_x.remove([])
    plot_all_y.remove([])
    x_all_labels.remove([])

    sorted_indices = sorted(range(len(x_all_labels)), key=lambda i: int(x_all_labels[i].split('(')[1].replace('��)', '')))
    x_all_labels_sorted = [x_all_labels[i] for i in sorted_indices]

    # Reorder plot_all_y according to sorted indices
    plot_all_y_sorted = [plot_all_y[i] for i in sorted_indices]
    plot_all_y = plot_all_y_sorted
    x_all_labels = x_all_labels_sorted

    x_position = 0
    pos_angle_max_len = len(plot_all_y[0])
    for index in range(len(plot_all_y)):
        if(len(plot_all_y[index]) > pos_angle_max_len):
            pos_angle_max_len = len(plot_all_y[index])
    for color, pdoa_y,x_labels1 in zip(colors, plot_all_y,x_all_labels):
        pdoa_x = []
        for index in range(len(pdoa_y)):
            pdoa_x.append(x_position + index)
        #print(pdoa_x)
        #print(pdoa_y)
        #print(len(pdoa_x))
        #print(len(pdoa_y))
        plt.scatter(pdoa_x, pdoa_y, color=color)
        x_ticks.append(x_position + len(pdoa_y) / 2)
        x_labels.append(f"{x_labels1}")
        x_position += pos_angle_max_len + 200

    plt.ylim(-360, 360)
    plt.yticks(range(-360, 370, 10))
    plt.xticks(x_ticks, x_labels, rotation=0, multialignment='center')
    plt.title(f'PDOAɢ��ͼ\n{folder_name}������')
    plt.xlabel('��������ϵ����90�㣩\nϵͳ����ϵ��360�㣩\n��ʵ�Ƕ�')
    plt.ylabel('�������Ƕ�PDOA')
    plt.grid(True)

    output_folder = os.path.join(folder_path, '�ϲ����')
    os.makedirs(output_folder, exist_ok=True)
    plt.savefig(os.path.join(output_folder, f'{folder_name}������_PDOAɢ��ͼ(����������180).png'))
    plt.close()

    print("PDOAͼ�ѱ������ļ���")


def main():
    root = tk.Tk()
    root.withdraw()
    folder_path = filedialog.askdirectory(title="��ѡ��Ҫ������ļ���")
    if folder_path:
        folder_name = os.path.basename(folder_path)
        output_folder, merged_file = merge_txt_files(folder_path, folder_name)
        file_in, file_outer = process_merged_file(output_folder, merged_file, folder_name)
        plot_angle(file_in, folder_name)
        plot_pdoa(merged_file, folder_name)
        plot_pdoa_data(folder_path,folder_name)
        plot_angle_data(folder_path, folder_name)

    else:
        print("δѡ���κ��ļ���")

if __name__ == '__main__':
    main()
