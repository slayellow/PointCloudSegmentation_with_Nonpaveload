import os
import time
import numpy as np
import struct
import UtilityManagement.config as cf


log_path = cf.paths["log_path"]

# Log File Open
# Struct Size(xyz) : 1843873
# Before Sensor Struct : 673
file = open(log_path, "rb")
count = 1;
while True:
    end = time.time()
    # SyncTime Read
    byte = file.read(8)

    if not byte:
        break
    SyncTime = struct.unpack('d', byte)
    # Header Read
    byte = file.read(665)

    # Lidar Data Read
    byte = file.read(1843200)

    data_list = []
    for idx in range(230400):
        x, y, z, layer, intensity = struct.unpack('hhhBB', byte[idx*8:idx*8+8])
        if x ==0 and y ==0 and z == 0:
            continue
        data_list.append([x * 0.01 ,y * 0.01 ,z * 0.01, intensity / 255])
    data_list = np.array(data_list, dtype=np.float32)

    path = os.path.join(cf.paths["log_save_path"], str(count).zfill(6) + ".bin")
    data_list.tofile(path)
    print("Index : " + str(count) + ", Data Load --> SyncTime : " + str(SyncTime) + " Loading Time : " + str(time.time() - end) + ", Path : " + path)
    count = count + 1


