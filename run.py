#!/usr/bin/python3

import os
import csv
import time
import keras
import numpy as np
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense, Dropout, GRU
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.utils import to_categorical, plot_model
from sklearn.preprocessing import Normalizer
import matplotlib.pyplot as plt

#  ----------------------------
# | Execution time measurement |
#  ----------------------------
start = time.perf_counter()

#  --------------------
# | Data preprocessing |
#  --------------------

# Attack type classification
print("[+] NSL-KDD data classification to 4 attack data types")
print("--------------------------------------------------------------------------------")
dos_type = ['back','land','neptune','pod','smurf','teardrop','processtable','udpstorm','mailbomb','apache2']
probing_type = ['ipsweep','mscan','nmap','portsweep','saint','satan']
r2l_type = ['ftp_write','guess_passwd','imap','multihop','phf','warezmaster','warezclient','spy','sendmail','xlock','snmpguess','named','xsnoop','snmpgetattack','worm']
u2r_type = ['buffer_overflow','loadmodule','perl','rootkit','xterm','ps','httptunnel','sqlattack']
type2id = {'normal':0}

for i in dos_type:
    type2id[i] = 1
for i in r2l_type:
    type2id[i] = 2
for i in u2r_type:
    type2id[i] = 3
for i in probing_type:
    type2id[i] = 4
print(f"Classified attack data types: {type2id}\n")

# Every protocol type within NSL-KDD gets it's id
print("[+] NSL-KDD dataset protocol types")
print("------------------------------------------------------------")
all_protocol = ['tcp', 'udp', 'icmp']
protocol_dict = {}
for id,name in enumerate(all_protocol):
    protocol_dict[name] = id
print(f"Protocol types: {protocol_dict}\n")

# Every service type within NSL-KDD gets it's id
print("[+] NSL-KDD dataset service types")
print("-----------------------------------------------------------")
all_service = ['aol', 'auth', 'bgp', 'courier', 'csnet_ns', 'ctf', 'daytime', 'discard', 'domain', 'domain_u', 'echo', 'eco_i', 'ecr_i', 'efs', 'exec', 'finger', 'ftp', 'ftp_data', 'gopher', 'harvest', 'hostnames', 'http', 'http_2784', 'http_443', 'http_8001', 'imap4', 'IRC', 'iso_tsap', 'klogin', 'kshell', 'ldap', 'link', 'login', 'mtp', 'name', 'netbios_dgm', 'netbios_ns', 'netbios_ssn', 'netstat', 'nnsp', 'nntp', 'ntp_u', 'other', 'pm_dump', 'pop_2', 'pop_3', 'printer', 'private', 'red_i', 'remote_job', 'rje', 'shell', 'smtp', 'sql_net', 'ssh', 'sunrpc', 'supdup', 'systat', 'telnet', 'tftp_u', 'tim_i', 'time', 'urh_i', 'urp_i', 'uucp', 'uucp_path', 'vmnet', 'whois', 'X11', 'Z39_50']
service_dict = {}
for id,name in enumerate(all_service):
    service_dict[name] = id
print(f"Service types: {service_dict}\n")

# Every flag within NSL-KDD gets it's id
print("[+] NSL-KDD dataset flag types")
print("--------------------------------------------------------------")
all_flag = ['OTH', 'REJ', 'RSTO', 'RSTOS0', 'RSTR', 'S0', 'S1', 'S2', 'S3', 'SF', 'SH']
flag_dict = {}
for id,name in enumerate(all_flag):
    flag_dict[name] = id
print(f"Flags: {flag_dict}\n")

# Read training data
all_train_data = []
trainX = []
trainY = []
with open(os.getcwd()+'/NSL-KDD/KDDTrain+.txt', newline='') as trainingData:
    trainingData = csv.reader(trainingData, delimiter=',')
    for row in trainingData:
        all_train_data.append(row) # Duomenys nuskaitomi eilutė po eilutės
print("[+] Training data from NSL-KDD")
print("-------------------------------------------------")
print(all_train_data[0],"\n")

# Encoding training data
for i in all_train_data:
    i[1] = protocol_dict[i[1]]
    i[2] = service_dict[i[2]]
    i[3] = flag_dict[i[3]]
    i[-2] = type2id[i[-2]]
    trainX.append(i[:41])
    trainY.append(i[-2])
print("[+] Encoded training data")
print("-------------------------------")
print(f"{trainX[0]}\n")
print("[+] Encoded cyber attack type (0-5)")
print("----------------------------------------------")
print(trainY[0],"\n")

# Reading testing data
all_test_data = []
testX = []
testY = []
with open(os.getcwd()+'/NSL-KDD/KDDTest+.txt', newline='') as testData:
    testData = csv.reader(testData, delimiter=',')
    for row in testData:
        all_test_data.append(row) # Read data row by row
print("[+] Testing data from NSL-KDD")
print("--------------------------------------------------")
print(all_test_data[0],"\n")

# Encoding testing data
for i in all_test_data:
    i[1] = protocol_dict[i[1]]
    i[2] = service_dict[i[2]]
    i[3] = flag_dict[i[3]]
    i[-2] = type2id[i[-2]]
    testX.append(i[:41])
    testY.append(i[-2])

print("[+] Encoded testing data")
print("--------------------------------")
print(f"{testX[0]}\n")
print("[+] Encoded cyber attack type (0-5)")
print("----------------------------------------------")
print(f"{testY[0]}\n")

# Data preprocessing
trainX = Normalizer().fit_transform(trainX)
print("\n[+] Transformed training data")
print("-----------------------------------")
print(f"{trainX}\n")
testX = Normalizer().fit_transform(testX)
print("[+] Transformed testing data")
print("------------------------------------")
print(f"{testX}\n")

# Data transformation to 3D array
trainX = np.reshape(trainX, (trainX.shape[0], 1, trainX.shape[1]))
print("[+] Training data transformed to 3D array")
print("-----------------------------------------------")
print(trainX,"\n")
testX = np.reshape(testX, (testX.shape[0], 1, testX.shape[1]))
print("[+] Testing data transformed to 3D array")
print("------------------------------------------------")
print(testX,"\n")

# One-hot encoding of classified data
trainY = tf.keras.utils.to_categorical(trainY, num_classes=5)
print("[+] Encoding NSL-KDD training data of attack classes by one-hot method")
print("----------------------------------------------------------------------------")
print(trainY,"\n")
testY = tf.keras.utils.to_categorical(testY, num_classes=5)
print("[+] Encoding NSL-KDD testing data of attack classes by one-hot method")
print("-----------------------------------------------------------------------------")
print(testY,"\n")

print("[+] Data is prepared\n")

#  ------------------
# | Data is prepared |
#  ------------------

#  -------
# | Model |
#  -------

# Creating model
model = Sequential()
model.add(GRU(256, input_shape=(1,41)))     # Gated Recurrent Unit
model.add(Dropout(0.2))
model.add(Dense(64, activation='relu'))     # Hidden layer #1
model.add(Dropout(0.2))
model.add(Dense(64, activation='relu'))     # Hidden layer #2
model.add(Dropout(0.2))
model.add(Dense(5, activation='softmax'))   # Output layer

sgd = SGD(learning_rate=0.01, decay=1e-6, momentum=0.9, nesterov=True)
model.compile(loss='categorical_crossentropy',
              optimizer=sgd,
              metrics=['accuracy'])

print("\n[+] Model training and validation")
print("---------------------------------------")
history = model.fit(trainX, trainY,
          epochs=20,
          batch_size=16, validation_split=0.1, verbose=1)

print("\n[+] Model testing")
print("------------------------")
_, score = model.evaluate(testX, testY, batch_size=16)
print("\n[+] Modelio accuracy")
print("-----------------------")
print('Accuracy: %.2f' % (score*100)+"%")

#  --------
# | Graphs |
#  --------

print("\n[+] Graphs of a created neural network")
print("------------------------------------------------")
print("[+] Neural network block scheme")
plot_model(model, to_file='Models/model.png')
print("[+] Graph saved at: "+os.getcwd()+"/Models/model.png")

# Graph of training and validation accuracy values
print("[+] Neural network accuracy graph")
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('Model accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper left')
plt.savefig(os.getcwd()+'/Models/acc.png')
print("[+] Graph saved at: "+os.getcwd()+"/Models/acc.png")

# Graph of training and validation loss values
print("[+] Neural network loss value graph")
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper left')
plt.savefig(os.getcwd()+'/Models/loss.png')
print("[+] Graph saved at: "+os.getcwd()+"/Models/loss.png")

#  ----------------------------
# | Execution time measurement |
#  ----------------------------
print("\n[+] Execution time")
print("------------------------------")
end = time.perf_counter()
time = end-start
if (time < 60):
    print("Execution time: %.2f" % time+" s")
if (time > 60):
    minutes = time / 60
    seconds = time - int(minutes) * 60
    print("Execution time: %.0f" % minutes+" min. ir %.2f" % seconds+" s")
