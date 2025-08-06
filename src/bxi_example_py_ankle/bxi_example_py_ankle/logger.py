# Copyright (c) 2025 Xuxin @ 747302550. 保留所有权利. 未经许可，禁止复制、修改或分发
import datetime
import csv
import os
import os.path as osp

class FileLogger:
    def __init__(self, root_dir, dt, variable_name="observation"):
        self.dt=dt
        current_time = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        os.makedirs(root_dir,exist_ok=True)
        csv_name = f"{variable_name}@{current_time}.csv"
        csv_path = os.path.join(root_dir,csv_name)
        self.file=open(csv_path, 'w', newline='')
        self.csv_writer = csv.writer(self.file)
        self.count = 0

    def data_log(self,observation_to_log):
        log_time= self.count * self.dt
        data1=list(observation_to_log)
        data1.insert(0,log_time)
        self.csv_writer.writerow(data1)
        self.count+=1