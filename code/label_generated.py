
import torch
import numpy as np
import pickle
import os
import cv2
from sklearn.preprocessing import MinMaxScaler

def generated_train_batch_label(cls, csvpath):
    if cls == "Panicle":
        pipe_path = r'../data/Panicle/train/labels.pkl'
    if cls == "Rice":
        pipe_path = r'../data/Rice/train/labels.pkl'
    if cls == "Maize":
        pipe_path = r'../data/Maize/train/labels.pkl'
    if cls == "Cotton":
        pipe_path = r'../data/Cotton/train/labels.pkl'

    # 1 filenames.txt generate: run ../data/<cls>/images/.bat, replace to ../data/Panicle/train/filenames.txt
    # 2 label.pkl
    f1 = open(csvpath, "r")
    case_train = np.loadtxt(f1, delimiter=',', skiprows=0)
    case_train1 = np.array(case_train)
    # print(case_train1)
    scaler = MinMaxScaler()
    scaler.fit(case_train1)
    c = scaler.transform(case_train1)
    c = torch.from_numpy(c).float()
    print(c.shape)
    with open(pipe_path, 'wb') as fw:
        pickle.dump(c, fw, protocol=2)

def generated_test_batch_label(cls, csvpath):
    if cls == "Panicle":
        path = r'../data/Panicle/test/filenames.txt'
        pipe_path = r'../data/Panicle/test/labels.pkl'
        imgpath = r'../data/Panicle/images'
    if cls == "Rice":
        path = r'../data/Rice/test/filenames.txt'
        pipe_path = r'../data/Rice/test/labels.pkl'
        imgpath = r'../data/Rice/images'
    if cls == "Maize":
        path = r'../data/Maize/test/filenames.txt'
        pipe_path = r'../data/Maize/test/labels.pkl'
        imgpath = r'../data/Maize/images'
    if cls == "Cotton":
        path = r'../data/Cotton/test/filenames.txt'
        pipe_path = r'../data/Cotton/test/labels.pkl'
        imgpath = r'../data/Cotton/images'

    # 1 filenames.txt generate
    file = open(path, 'w')
    total = len(open(csvpath).readlines())
    for r in range(0, total):
        file.write('ept_' + str(r) + '.png' + '\n')

    # 2 label.pkl
    f1 = open(csvpath, "r")
    case_train = np.loadtxt(f1, delimiter=',', skiprows=0)
    case_train1 = np.array(case_train)
    # print(case_train1)
    scaler = MinMaxScaler()
    scaler.fit(case_train1)
    c = scaler.transform(case_train1)
    c = torch.from_numpy(c).float()
    print(c.shape)
    with open(pipe_path, 'wb') as fw:
        pickle.dump(c, fw, protocol=2)

    # 3 ept images
    img = np.zeros((256, 256, 3), np.uint8)
    img.fill(200)  # 浅灰色背景
    for r in range(0, total):
        name = 'ept_' + str(r) + '.png'
        full = os.path.join(imgpath, name)
        # 保存空图
        cv2.imwrite(full, img)

def generated_test_sigle_label(cls, single_traits, single_image_name):
    img = np.zeros((256, 256, 3), np.uint8)
    img.fill(200)  # 浅灰色背景
    if cls == "Panicle":
        origin_t_path = r'../data/Panicle/test/single_default/traits.csv'
        path = r'../data/Panicle/test/filenames.txt'
        pipe_path = r'../data/Panicle/test/labels.pkl'
        imgpath = r'../data/Panicle/images'
    if cls == "Rice":
        origin_t_path = r'../data/Rice/test/single_default/traits.csv'
        path = r'../data/Rice/test/filenames.txt'
        pipe_path = r'../data/Rice/test/labels.pkl'
        imgpath = r'../data/Rice/images'
    if cls == "Maize":
        origin_t_path = r'../data/Maize/test/single_default/traits.csv'
        path = r'../data/Maize/test/filenames.txt'
        pipe_path = r'../data/Maize/test/labels.pkl'
        imgpath = r'../data/Maize/images'
    if cls == "Cotton":
        origin_t_path = r'../data/Cotton/test/single_default/traits.csv'
        path = r'../data/Cotton/test/filenames.txt'
        pipe_path = r'../data/Cotton/test/labels.pkl'
        imgpath = r'../data/Cotton/images'
    listx = single_traits
    list_new = list(map(str, listx))

    # single txt
    file = open(path, 'w')
    if (cls == "Panicle") | (cls == "Maize") | (cls == "Cotton"):
        for i in range(0, 1):
            file.write(single_image_name + '\n')
    if cls == "Rice":
        for i in range(0, 2):
            file.write(single_image_name+'\n')

    # single ept image
    full = os.path.join(imgpath, single_image_name)
    cv2.imwrite(full, img)

    # single pkl
    fp = open(origin_t_path, 'a+', encoding='utf-8', newline='')
    #fp.write('\n' + ','.join(list_new))
    fp.write(','.join(list_new)+'\n')
    fp.close()
    f1 = open(origin_t_path, "r")
    case_train = np.loadtxt(f1, delimiter=',', skiprows=0)
    case_train1 = np.array(case_train)
    # print(case_train1)
    scaler = MinMaxScaler()
    scaler.fit(case_train1)
    c = scaler.transform(case_train1)
    c = torch.from_numpy(c).float()
    # print(c)
    if (cls == "Panicle") | (cls == "Maize") | (cls == "Cotton"):
        d = c[-1:, :]
    if cls == "Rice":
        d = c[-2:, :]
    # print(d)
    # print(d.shape)
    with open(pipe_path, 'wb') as fw:
        pickle.dump(d, fw, protocol=2)



