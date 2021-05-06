from os import listdir
from os.path import isfile, join
import numpy as np
import json


class DataLoader(object):
    def __init__(self, input_timestamp_length, output_timestamp_length, path_json_folder):
        self.input_timestamp_length = input_timestamp_length
        self.output_timestamp_length = output_timestamp_length
        self.path_json_folder = path_json_folder
        self.X = []
        self.Y = []
        print("Input timestamp length: ", input_timestamp_length)
        print("Output timestamp length: ", output_timestamp_length)

    def SlidingWindow(self, list_dx, list_dy, list_vx,
                      list_vy, length_input, length_output):

        # Single X - > Y  : Input shape (1, input_length, 4(dx, dy, vx, vy)) | Output shape (1, 4*lenght_output(dx, dy, vx, vy))
        input_samples = []
        output_samples = []

        if len(list_dx) >= length_input + length_output:
            for i in range(len(list_dx) - length_output - length_input + 1):
                # Inputs:
                input_new_dx = list_dx[i:length_input + i]
                input_new_dy = list_dy[i:length_input + i]
                input_new_vx = list_vx[i:length_input + i]
                input_new_vy = list_vy[i:length_input + i]
                # Outputs:
                output_dx = list_dx[length_input + i:length_input + i + length_output]
                output_dy = list_dy[length_input + i:length_input + i + length_output]
                output_vx = list_vx[length_input + i:length_input + i + length_output]
                output_vy = list_vy[length_input + i:length_input + i + length_output]

                sample_input_all_timestamps = []
                # Iterate over each timestamp:
                for j in range(length_input):
                    single_time_stamp_data = [input_new_dx[j], input_new_dy[j], input_new_vx[j], input_new_vy[j]]
                    sample_input_all_timestamps.append(single_time_stamp_data)
                sample_input_all_timestamps = np.array(sample_input_all_timestamps).reshape((1, length_input, 4))
                input_samples.append(sample_input_all_timestamps)

                output = []
                for j in range(length_output):
                    output_single_timestamp = [output_dx[j], output_dy[j], output_vx[j], output_vy[j]]
                    output.append(output_single_timestamp)
                output = np.array(output).reshape((1, 4 * length_output))
                output_samples.append(output)

                '''print("Input shape: ", sample_input_all_timestamps.shape)
                print(sample_input_all_timestamps)
                print("Output shape: ", output.shape)
                print(output)
                print("***************************************************************************************")'''

            return (input_samples, output_samples)

        else:
            #print("Timestamp length is not enough.")
            return None

    def Normalize(self, all_samples, min, range):
        all_samples_normalized = []
        for id, single_sample in enumerate(all_samples):
            dx = np.array(single_sample['dx'])
            dy = np.array(single_sample['dy'])
            vx = np.array(single_sample['vx'])
            vy = np.array(single_sample['vy'])

            dx_normalized = list((dx - min) / range)
            dy_normalized = list((dy - min) / range)
            vx_normalized = list((vx - min) / range)
            vy_normalized = list((vy - min) / range)

            single_sample_normalized = {'dx': dx_normalized, 'dy': dy_normalized, 'vx': vx_normalized,
                                        'vy': vy_normalized}
            all_samples_normalized.append(single_sample_normalized)

        return all_samples_normalized

    def LoadDateset(self):
        # Opening JSON file
        list_json_names = [f for f in listdir(self.path_json_folder) if isfile(join(self.path_json_folder, f))]

        # Iterate over json files, each json file represents motion of the different Waymo object:
        all_numbers = []
        all_samples = []
        for json_name in list_json_names:
            json_path = self.path_json_folder + '/' + json_name
            with open(json_path) as data_file:
                data = json.load(data_file)
                for item in data['dx']:
                    all_numbers.append(item)
                for item in data['dy']:
                    all_numbers.append(item)
                for item in data['vx']:
                    all_numbers.append(item)
                for item in data['vy']:
                    all_numbers.append(item)
                single_sample = {'dx': data['dx'], 'dy': data['dy'], 'vx': data['vx'], 'vy': data['vy']}
                all_samples.append(single_sample)

        # Normalizing coeff:
        max_val = max(all_numbers)
        min_val = min(all_numbers)
        range = max_val - min_val

        print("Data read is done.")
        print("Total object count: ", len(all_samples))

        all_samples_normalized = self.Normalize(all_samples, min_val, range)

        # Iterate over the whole normalized data:
        for single_sample_normalized in all_samples_normalized:
            data = self.SlidingWindow(single_sample_normalized['dx'],
                                      single_sample_normalized['dy'],
                                      single_sample_normalized['vx'],
                                      single_sample_normalized['vy'],
                                      int(self.input_timestamp_length),
                                      int(self.output_timestamp_length))
            if data is not None:
                X = data[0]
                Y = data[1]
                for x in X:
                    self.X.append(x)
                for y in Y:
                    self.Y.append(y)

        # Make Numpy array:
        self.X = np.array(self.X)
        self.X = np.squeeze(self.X, axis=(1))
        self.Y = np.array(self.Y)
        self.Y = np.squeeze(self.Y, axis=(1))
        print("X shape: ", self.X.shape)
        print("Y shape: ", self.Y.shape)
        return self.X, self.Y

        '''# Test:
        dx = ["x1", "x2", "x3", "x4", "x5", "x6", "x7", "x8", "x9", "x10"]
        dy = ["y1", "y2", "y3", "y4", "y5", "y6", "y7", "y8", "y9", "y10"]
        vx = ["vx1", "vx2", "vx3", "vx4", "vx5", "vx6", "vx7", "vx8", "vx9", "vx10"]
        vy = ["vy1", "vy2", "vy3", "vy4", "vy5", "vy6", "vy7", "vy8", "vy9", "vy10"]
        self.SlidingWindow(dx, dy, vx, vy, 5, 3)'''





