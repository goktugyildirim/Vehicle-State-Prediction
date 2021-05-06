from os import listdir
from os.path import isfile, join
import numpy as np
from waymo_open_dataset.utils import range_image_utils
from waymo_open_dataset.utils import transform_utils
from waymo_open_dataset.utils import frame_utils
from waymo_open_dataset import dataset_pb2 as open_dataset
import tensorflow.compat.v1 as tf
import math
import json
import io
try:
    to_unicode = unicode
except NameError:
    to_unicode = str

class WaymoDataGenerator():
    def __init__(self):
        dataset_path = '/home/goktug/Desktop/Waymo_Valid_Dataset/valid0'
        list_name_tfrecords = [f for f in listdir(dataset_path) if isfile(join(dataset_path, f))]
        self.list_path_tfrecords = [dataset_path + '/' + name for name in list_name_tfrecords]
        self.object_pose_vel_all = []
        self.DataGenerator()

    def isRotationMatrix(self, R) :
        Rt = np.transpose(R)
        shouldBeIdentity = np.dot(Rt, R)
        I = np.identity(3, dtype = R.dtype)
        n = np.linalg.norm(I - shouldBeIdentity)
        return n < 1e-6

    def rotationMatrixToEulerAngles(self, R) :
        assert(self.isRotationMatrix(R))
        sy = math.sqrt(R[0,0] * R[0,0] +  R[1,0] * R[1,0])
        singular = sy < 1e-6
        if  not singular :
            x = math.atan2(R[2,1] , R[2,2])
            y = math.atan2(-R[2,0], sy)
            z = math.atan2(R[1,0], R[0,0])
        else :
            x = math.atan2(-R[1,2], R[1,1])
            y = math.atan2(-R[2,0], sy)
            z = 0
        return np.array([x, y, z])

    def euler_to_quaternion(self, roll, pitch, yaw):
        qx = np.sin(roll/2) * np.cos(pitch/2) * np.cos(yaw/2) - np.cos(roll/2) * np.sin(pitch/2) * np.sin(yaw/2)
        qy = np.cos(roll/2) * np.sin(pitch/2) * np.cos(yaw/2) + np.sin(roll/2) * np.cos(pitch/2) * np.sin(yaw/2)
        qz = np.cos(roll/2) * np.cos(pitch/2) * np.sin(yaw/2) - np.sin(roll/2) * np.sin(pitch/2) * np.cos(yaw/2)
        qw = np.cos(roll/2) * np.cos(pitch/2) * np.cos(yaw/2) + np.sin(roll/2) * np.sin(pitch/2) * np.sin(yaw/2)
        return [qx, qy, qz, qw]

    def DataGenerator(self):
        # Iterate over TFRecords:
        for path_tfrecord in self.list_path_tfrecords:
            dataset = tf.data.TFRecordDataset(path_tfrecord, compression_type='')
            set_object_ids = []
            context_name = ''

            for frame_id, data in enumerate(dataset):
                frame = open_dataset.Frame()
                frame.ParseFromString(bytearray(data.numpy()))
                context_name = frame.context.name
                objects_labels = frame.laser_labels

                for object_label in objects_labels:
                    if object_label.type == 1:
                        set_object_ids.append(object_label.id)
                set_object_ids = list(set(set_object_ids))

                '''# Vehicle pose:
                rotation_matrix = np.array([(frame.pose.transform[0], frame.pose.transform[1], frame.pose.transform[2]),
                                            (frame.pose.transform[4], frame.pose.transform[5], frame.pose.transform[6]),
                                            (frame.pose.transform[8], frame.pose.transform[9], frame.pose.transform[10])])
                r, p, y = self.rotationMatrixToEulerAngles(rotation_matrix)
                quaternion = self.euler_to_quaternion(r, p, y)
                
                qx = quaternion[0]
                qy = quaternion[1]
                qz = quaternion[2]
                qw = quaternion[3]
                dx = frame.pose.transform[3]
                dy = frame.pose.transform[7]
                dz = frame.pose.transform[11]'''

            #print("Set of objects ids in a single tfrecord:")
            print("Context name:", context_name)
            print("Vehicle object count:", len(set_object_ids))

            # Iterate over the single tfrecord to collect object pose and velocities:
            for single_object_id in set_object_ids:
                single_object_px = []
                single_object_py = []
                single_object_vx = []
                single_object_vy = []
                print("Object id:", single_object_id)
                for frame_id, data in enumerate(dataset):
                    frame = open_dataset.Frame()
                    frame.ParseFromString(bytearray(data.numpy()))

                    for object_label in frame.laser_labels:
                        if object_label.id == single_object_id:
                            dx = object_label.box.center_x
                            dy = object_label.box.center_y
                            speed_x = object_label.metadata.speed_x
                            speed_y = object_label.metadata.speed_y
                            single_object_px.append(dx)
                            single_object_py.append(dy)
                            single_object_vx.append(speed_x)
                            single_object_vy.append(speed_y)

                single_object_data =\
                {
                    "context_name": context_name,
                    "object_id": single_object_id,
                    "count_timestamp": len(single_object_px),
                    "dx": single_object_px,
                    "dy": single_object_py,
                    "vx": single_object_vx,
                    "vy": single_object_vy,
                }

                json_name = '/home/goktug/projects/tracking_predictor/waymo_valid_0/' + str(single_object_id) + '.json'
                with io.open(json_name, 'w', encoding='utf8') as outfile:
                    str_ = json.dumps(single_object_data,
                          indent=7, sort_keys=False,
                          separators=(',', ': '), ensure_ascii=False)
                    outfile.write(to_unicode(str_))
















if __name__ == '__main__':
    generator = WaymoDataGenerator()
