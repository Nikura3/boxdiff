import pathlib
import time
import datetime

from config import RunConfig

import torch

class Logger:
    def __init__(self, path: pathlib.Path):
        self._logger = open(path / "log.txt", "w")
        self._runs = []
    def log_gpu_memory_instance(self,config: RunConfig):
        name = torch.cuda.get_device_name(torch.cuda.current_device())
        total_memory = torch.cuda.get_device_properties(torch.cuda.current_device()).total_memory
        reserved_memory = torch.cuda.memory_reserved(torch.cuda.current_device())

        # print("Assigned GPU:", name)
        # print(f"Total memory: {total_memory / pow(2, 30):.2f} GB")
        # print(f"Reserved memory: {reserved_memory / pow(2, 30):.2f} GB")
        # print(f"Free memory: {free_memory / pow(2, 30):.2f} GB")

        lines=[datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S-")+config.prompt+"\n",
               "Assigned GPU:"+str(name)+"\n",
               f"Total memory: {total_memory / pow(2, 30):.2f} GB\n",
               f"Reserved memory: {torch.cuda.max_memory_reserved(torch.cuda.current_device()) / pow(2, 30):.2f} GB\n"
               ]
        self._logger.writelines(lines)

    def log_time_run(self,start,end):
        self._runs.append((start,end))

    def log_execution_time(self):
        total_elapsed = 0

        for run in enumerate(self._runs):
            elapsed = run[1][1]-run[1][0]
            self._logger.writelines("Run " + str(run[0]) + ": {:.2f}s\n".format(elapsed))
            total_elapsed = total_elapsed + elapsed

        self._logger.writelines("Average time over " + str(len(self._runs)) + " runs is: {:.2f}s\n".format(total_elapsed / len(self._runs)))
        self._logger.close()



