import os
import shutil
import csv


def get_all_files_in_directory(directory):
    return [os.path.join(directory, f) for f in os.listdir(directory) if os.path.isfile(os.path.join(directory, f))]


def create_yolo_label_csv(list_of_yolo_label_directories):
    with open("yolo_labels.csv", 'w', newline='') as csvfile:
        csv_writer = csv.writer(csvfile, delimiter=',')
        csv_writer.writerow(list_of_yolo_label_directories)
        super_detected_list = []
        for directory in list_of_yolo_label_directories:
            all_files = get_all_files_in_directory(directory)
            detected_list = []
            for file in all_files:
                people_detected = 0
                with open(file, 'r') as f:
                    lines = f.readlines()
                    for line in lines:
                        if line[0] == '0':
                            people_detected += 1
                detected_list.append(people_detected)
            super_detected_list.append(detected_list)

        print(len(super_detected_list))
        print(len(super_detected_list[0]))
        print(len(super_detected_list[1]))
        print(len(super_detected_list[2]))


        max_len = 0
        for i in super_detected_list:
            if len(i) > max_len:
                max_len = len(i)

        for i in range(max_len):
            this_row = []
            for j in range(len(super_detected_list)):
                if i < len(super_detected_list[j]):
                    this_row.append(str(super_detected_list[j][i]))
                else:
                    this_row.append("")
            csv_writer.writerow(this_row)


yolo_label_directories = [
    r'C:\Users\Regicide\Desktop\Honors Project\Data\Lidar\src\yolo_labels\exp5\labels',
    r'C:\Users\Regicide\Desktop\Honors Project\Data\Lidar\src\yolo_labels\exp6\labels',
    r'C:\Users\Regicide\Desktop\Honors Project\Data\Lidar\src\yolo_labels\exp7\labels'
]
create_yolo_label_csv(yolo_label_directories)