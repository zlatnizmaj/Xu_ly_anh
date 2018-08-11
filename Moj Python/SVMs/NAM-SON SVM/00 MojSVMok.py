import os
import glob

path_files_CSV = "./dataset"
# files = os.listdir(path_files_CSV)
# files = glob.glob(path_files_CSV+'/*.csv')
# print(files)

directory = os.path.join(path_files_CSV, "path")
for root, dirs, files in os.walk(directory):
    for file in files:
        if file.endswith(".csv"):
            f = open(file, 'r')
            #  perform calculation
            f.close()
