import os
import shutil

def get_all_files_in_directory(directory):
    return [os.path.join(directory, f) for f in os.listdir(directory) if os.path.isfile(os.path.join(directory, f))]

def get_all_files_in_directory_list(directory_list):
    all_files = []
    for directory in directory_list:
        all_files.extend(get_all_files_in_directory(directory))
    return all_files

def remove_unwanted_files(file_list):
    first_file_transitions = [54, 77, 83, 93, 101, 112, 120, 131, 140, 155, 175, 189, 199, 216, 222, 240]
    # second and third file have no transitions
    fourth_file_transitions = [7, 22, 31, 37, 48, 51, 54, 64, 71, 82, 138, 184, 195, 204, 213, 233, 243, 271, 278, 282,
                               358, 371, 387, 413, 448, 450, 458, 478, 491, 492, 495, 496]

    remove_list = []
    for transition in first_file_transitions:
        for file in file_list:
            # remove the moment of the transition
            if ("first_" + str(transition)) in file:
                if file not in remove_list:
                    remove_list.append(file)
            # remove the two previous moments
            elif ("first_" + str(transition - 1)) in file:
                if file not in remove_list:
                    remove_list.append(file)
            elif ("first_" + str(transition - 2)) in file:
                if file not in remove_list:
                    remove_list.append(file)

    for file in remove_list:
        file_list.remove(file)
    remove_list = []

    for transition in fourth_file_transitions:
        for file in file_list:
            # remove the moment of the transition
            if ("fourth_" + str(transition)) in file:
                if file not in remove_list:
                    remove_list.append(file)
            # remove the two previous moments
            elif ("fourth_" + str(transition - 1)) in file:
                if file not in remove_list:
                    remove_list.append(file)
            elif ("fourth_" + str(transition - 2)) in file:
                if file not in remove_list:
                    remove_list.append(file)

    for file in remove_list:
        file_list.remove(file)

    return file_list


def copy_files_to_new_directory(file_list, new_directory):
    for file in file_list:
        copyfile(file, new_directory + file)


def create_trainable_set():
    directory_list = [r'C:\Users\Regicide\Desktop\Honors Project\Data\Lidar\frames\2023-03-23 17-05-33',
                      r'C:\Users\Regicide\Desktop\Honors Project\Data\Lidar\frames\2023-03-23 17-09-56',
                      r'C:\Users\Regicide\Desktop\Honors Project\Data\Lidar\frames\2023-03-23 17-10-09',
                      r'C:\Users\Regicide\Desktop\Honors Project\Data\Lidar\frames\2023-03-23 17-35-46']
    all_files = get_all_files_in_directory_list(directory_list)
    all_files = remove_unwanted_files(all_files)
    if not os.path.exists("trainable_set/"):
        os.mkdir("trainable_set/")
    for file in all_files:
        shutil.copy(file, "trainable_set/" + os.path.basename(file))


# create_trainable_set()