import os
import glob
import csv

directory_path = os.path.dirname(os.getcwd()+'\param_est_heeds_Study_1\POST_0\Design1')
output_path = os.path.dirname(os.getcwd()+'\..\results')
plot_files = glob.glob(directory_path + "/**/Design Table.plot", recursive = True)
txt_files = os.path.join(directory_path, '/**/Design Table.plot')

# print(plot_files)
for txt_file in glob.glob(txt_files):
    with open(txt_file, "rb") as input_file:
        in_txt = csv.reader(input_file, delimiter=',')
        filename = os.path.splitext(os.path.basename(txt_file))[0] + '.csv'

        with open(os.path.join(output_path, filename), 'wb') as output_file:
            out_csv = csv.writer(output_file)
            out_csv.writerows(in_txt)