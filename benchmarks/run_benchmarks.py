import sys
import os
sys.path.insert(1, 'image_classification/alexnet/')
sys.path.insert(1, 'image_classification/mobilenetsv2/')
sys.path.insert(1, 'image_classification/resnet/')
sys.path.insert(1, 'image_classification/lenet/')
sys.path.insert(1, 'image_classification/vgg/')
sys.path.insert(1, 'image_classification/effnet/')
sys.path.insert(1, 'image_classification/shufflenet/')
sys.path.insert(1, 'image_classification/yolo4-tiny/')


import run_alexnet
import run_mobilenet
import run_resnet
import run_lenet
import run_vgg
import run_effnet
import run_shufflenet
import run_yolo

import os

# main
def run_models():
    if(len(sys.argv) != 3):
        print('Error to parse. [name_model] arguments were exptected.')
        print('Usage: python '+sys.argv[0]+' [name_model]')
        exit(1)
    name_model = sys.argv[1].replace('"', '')  # 移除引号
    simulation_file = sys.argv[2].strip('"')  # 移除引号

    # "alexnet" "../dramsim3/configs/cache_mee_128_8.ini-alexnet-../simulation_files/cfg/tpu_16_16_pes.cfg"

    # split_result = simulation_file.split('-')
    # dram_name = split_result[0].split('/')[-1].split('.')[0]    #ini
    # arch_name = split_result[2].split('/')[-1].split('_')[0]    #cfg
    #
    # folder_path = "../dramsim3/output1/" +  arch_name + "_" + name_model  + "_" + dram_name
    # os.makedirs(folder_path, exist_ok=True)

    if (name_model != ''):
    	os.environ['NETNAME'] = name_model

    if(name_model == 'alexnet'):
        run_alexnet.run_model(simulation_file)

    elif (name_model == 'mobilenets'):
        run_mobilenet.run_model(simulation_file)

    elif (name_model == 'resnet'):
        run_resnet.run_model(simulation_file)

    elif (name_model == 'lenet'):
        run_lenet.run_model(simulation_file)

    elif (name_model == 'vgg'):
        run_vgg.run_model(simulation_file)

    elif (name_model == 'effnet'):
        run_effnet.run_model(simulation_file)

    elif (name_model == 'shufflenet'):
        run_shufflenet.run_model(simulation_file)

    elif (name_model == 'yolo'):
        run_yolo.run_model(simulation_file)

run_models()
