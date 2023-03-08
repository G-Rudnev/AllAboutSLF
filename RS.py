import pyrealsense2 as rs
import numpy as np
from numpy.linalg import norm
import cv2
import open3d
import time
import matplotlib.pyplot as plt
import seaborn as sns

import csv

from math import sqrt, inf, asin, degrees, cos, sin, pi


view_matrix = np.asarray([[1, 0, 0], [0, -1, 0], [0, 0, -1]], dtype=float)

# поворот облака для вертикального отображения при наклонах камеры
vertical_alignment_matrix = np.zeros([3, 3], dtype=float)

normal_verts = []

n_points = 0

class RealSense:
    def __init__(self):
        # Create a pipeline
        self.pipeline = rs.pipeline()
        self.orientation = np.zeros([3], dtype=float)
        self.gyro_bias = np.zeros([3], dtype=float)
        self.rot_matrix = np.zeros([3, 3], dtype=float)
        # Create a config and configure the pipeline to stream
        #  different resolutions of color and depth streams
        config = rs.config()
        colorizer = rs.colorizer()
        config.enable_stream(rs.stream.depth, 1280, 720, rs.format.z16, 30)
        config.enable_stream(rs.stream.color, 1280, 800, rs.format.rgb8, 30)
        # Accelerometer available FPS: {63, 250}Hz
        config.enable_stream(rs.stream.accel, rs.format.motion_xyz32f, 250)  # acceleration
        # Gyroscope available FPS: {200,400}Hz
        config.enable_stream(rs.stream.gyro, rs.format.motion_xyz32f, 200)  # gyroscope


        # # Get device product line for setting a supporting resolution
        # pipeline_wrapper = rs.pipeline_wrapper(self.pipeline)
        # pipeline_profile = config.resolve(pipeline_wrapper)
        # device = pipeline_profile.get_device()
        # device_product_line = str(device.get_info(rs.camera_info.product_line))

        # Start streaming
        profile = self.pipeline.start(config)
        # Getting the depth sensor's depth scale (see rs-align example for explanation)
        depth_sensor = profile.get_device().first_depth_sensor()
        depth_scale = depth_sensor.get_depth_scale()
        print("Depth Scale is: ", depth_scale)
        # Create an align object
        # rs.align allows us to perform alignment of depth frames to others frames
        # The "align_to" is the stream type to which we plan to align depth frames.
        align_to = rs.stream.color
        align = rs.align(align_to)

    def getRGBImage(self):  # TODO: НА ВЫХОДЕ КАКОЙ-ТО НЕ СОВСЕМ RGB ДА ЕЩЁ И ТЁМНЫЙ
        frame = self.pipeline.wait_for_frames()
        color_frame = frame.get_color_frame()
        color_image = np.asanyarray(color_frame.get_data())
        return color_image

    def getDepthImage(self):
        frame = self.pipeline.wait_for_frames()
        depth_frame = frame.get_depth_frame()
        depth_image = np.asanyarray(depth_frame.get_data())
        return depth_image

    def getPointCloud(self):
        global n_points

        pc = rs.pointcloud()
        decimate = rs.decimation_filter()
        frame = self.pipeline.wait_for_frames()
        depth_frame = decimate.process(frame.get_depth_frame())
        points = pc.calculate(depth_frame)

        v = points.get_vertices()
        verts = np.asanyarray(v).view(np.float32).reshape(-1, 3)  # xyz

        # зададим ROI для облака точек, чтобы не тянуть лишние шумы и блики
        xmin, xmax, ymin, ymax, zmin, zmax = -inf, inf, -inf, inf, 0.5, 5
        ind1 = np.all(verts > np.array([xmin, ymin, zmin]), axis=1)
        ind2 = np.all(verts < np.array([xmax, ymax, zmax]), axis=1)

        return verts[np.logical_and(ind1, ind2)]

    # достаём данные IMU
    def get_imu_data(self):  # достаём данные IMU
        t = time.time()
        frame = self.pipeline.wait_for_frames()
        lin_a_data = frame[2].as_motion_frame().get_motion_data()  # линейные ускорения
        ang_w_data = frame[3].as_motion_frame().get_motion_data()  # угловые скорости
        return lin_a_data, ang_w_data, time.time() - t

# color_img = cam.getRGBImage()
# cv2.imshow('Color', cv2.cvtColor(color_img, cv2.COLOR_RGB2BGR))

# cutted_verts = [[0, 0, 0]]
# # print(verts)
# for point in normal_verts:
#     # неэффективный алгоритм в лоб. Поскольку на картинке положение тележки строго горизонтально полу,
#     # и тележка стоит перед наклонной плоскостью "ровно" и сама плоскость имеет равномерный наклон (в одной плоскости)
#     # Переберем все облако и выкинем ненжные точки сравнивая угол между началом СК и точкой с заданным углом.
#     # z вперед х вправо у вниз
#
#     h = 0.15  # высота камеры
#     alpha = 45  # заданный угол
#
#     p = np.array(point[1], point[2])
#     a = np.array([0 - h, 0])
#     b = np.array([0 - h + np.sin(np.deg2rad(alpha)), 1])
#
#     ab = b - a
#     ap = p - a
#     angle = np.arccos(np.dot(ab, ap) / (np.linalg.norm(ab) * np.linalg.norm(ap)))
#     angle = np.rad2deg(angle)
#
#     #НЕУДАЧНОЕ МЕСТО, APPEND НЕ ДОПУСТИМ
#     if angle - alpha > 0:
#         cutted_verts.append([point[0], point[1], point[2]])
#     else:
#         1 + 1
#
# cutted_pcd = open3d.geometry.PointCloud()
# cutted_pcd.points = open3d.utility.Vector3dVector(np.array(cutted_verts))
#
# open3d.visualization.draw_geometries([cutted_pcd], window_name='cutted')

    def pcd_rotate(self, pcd):
        trans_pcd = np.asarray(pcd.points).T
        print(trans_pcd.shape)
        pitch = self.orientation[0]
        roll = self.orientation[2]
        self.rot_matrix[:, :] = \
            np.asarray([[1, 0, 0], [0, cos(pitch), -sin(pitch)], [0, sin(pitch), cos(pitch)]], dtype=float) @ \
            np.asarray([[cos(roll), -sin(roll), 0], [sin(roll), cos(roll), 0], [0, 0, 1]], dtype=float)

        return np.transpose(self.rot_matrix @ trans_pcd)


    # показать 1 облако точек глубины
    def visualize_single_pcd(self):
        global normal_verts

        normal_verts = self.getPointCloud()

        # with open('some.csv', 'w', newline='') as f:
        #     writer = csv.writer(f)
        #     writer.writerows(normal_verts)

        for_view_verts = normal_verts @ view_matrix
        for_view_pcd = open3d.geometry.PointCloud()

        for_view_pcd.points = open3d.utility.Vector3dVector(for_view_verts)
        downpcd = for_view_pcd.voxel_down_sample(voxel_size=0.01)
        # print(np.asarray(for_view_pcd.points).shape[0], np.asarray(downpcd.points).shape[0])

        downpcd.points = open3d.utility.Vector3dVector(self.pcd_rotate(downpcd))

        open3d.visualization.draw_geometries([downpcd])
        # open3d.visualization.draw_geometries([for_view_pcd], window_name='normal')

    # Показать последовательность облаков точек. Input: время работы в секундах, "частота обновления"
    def visualize_pcd_sequence(self, t: float, fps: float):
        global normal_verts

        # Create an o3d.visualizer class
        vis = open3d.visualization.Visualizer()
        vis.create_window()
        for_view_pcd = open3d.geometry.PointCloud()
        downpcd = open3d.geometry.PointCloud()
        ttt = []
        t0 = time.time()
        while time.time() - t0 < t:
            tfd = time.time()
            normal_verts = self.getPointCloud()
            for_view_verts = normal_verts @ view_matrix
            for_view_pcd.points = open3d.utility.Vector3dVector(for_view_verts)
            downpcd.clear()  # очистка облака от предыдущих вокселей
            downpcd = for_view_pcd.voxel_down_sample(voxel_size=0.01)
            print(np.asarray(for_view_pcd.points).shape[0], np.asarray(downpcd.points).shape[0])
            # Put point cloud into visualizer
            vis.add_geometry(downpcd)
            # Let visualizer render the point cloud
            vis.update_geometry(downpcd)
            vis.update_renderer()
            vis.poll_events()

            ttt.append(time.time() - tfd)

            # time.sleep(1/fps)
        return ttt

    def visualize_depth_image_sequence(self):
        try:
            while True:
                depth_image = self.getDepthImage()
                depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(depth_image, alpha=0.075), cv2.COLORMAP_JET)
                cv2.imshow('Depth image', depth_colormap)
                cv2.waitKey(1)

        except KeyboardInterrupt:
            pass

        finally:
            self.pipeline.stop()
            cv2.destroyAllWindows()

    def est_initial_pitch_roll(self):  # калибровка по времени занимает минуту, так сказать, на потупить перед дорогой
        print('Initial pitch and roll estimation. Waiting for IMU to stabilize')
        stabilization_meas_num = 1500  # число измерений для стабилизации
        calibration_meas_num = 300  # число используемых в оценке измерений

        for i in range(stabilization_meas_num):  # разгон IMU
            self.get_imu_data()

        print('Initial pitch and roll estimation. Maintaining estimation')
        a_x_sum = 0.0
        a_y_sum = 0.0
        a_z_sum = 0.0
        abs_sum = 0.0

        w_x_sum = 0.0
        w_y_sum = 0.0
        w_z_sum = 0.0
        x, y, z = [], [], []

        for i in range(calibration_meas_num):  # непосредственно, вычисление - среднее арифметическое
            imu_lin_a, imu_ang_w, _ = self.get_imu_data()

            a_x_sum += imu_lin_a.x
            a_y_sum += imu_lin_a.y
            a_z_sum += imu_lin_a.z

            w_x_sum += imu_ang_w.x
            x.append(imu_ang_w.x)
            w_y_sum += imu_ang_w.y
            y.append(imu_ang_w.y)
            w_z_sum += imu_ang_w.z
            z.append(imu_ang_w.z)

            abs_a = sqrt(imu_lin_a.x ** 2 + imu_lin_a.y ** 2 + imu_lin_a.z ** 2)
            abs_sum += abs_a

        # lin_a_std = np.std(a, axis=1)
        # ang_v_std = np.std(v, axis=1)

        self.orientation[0] = asin(a_z_sum / abs_sum)  # тангаж
        self.orientation[2] = asin(a_x_sum / sqrt(a_x_sum ** 2 + a_y_sum ** 2))  # крен
        self.gyro_bias[:] = np.asarray([w_x_sum / calibration_meas_num, w_y_sum / calibration_meas_num, w_z_sum / calibration_meas_num])

        fig, ax = plt.subplots(3, 1)
        ax[0].plot(x)
        ax[1].plot(y)
        ax[2].plot(z)
        plt.show()

    def estimate_current_orientation(self):
        """ использование измерений IMU для определения текущей ориентации (тангаж, крен) камеры
        как относительно инициализированной ориентации - по гироскопам,
        так и относительно вектора гравитации - акселерометры TODO(целесообразно ли в движении????) """
        # Ну и что тут? Калмана применять?
        # 10 измерений занимают 330 мс. Если доставать одно значение, TODO: то точно придётся Калмана прикрутить - как раз из калибровки можно достать отклонения
        # 1 измерение - 33 мс, соответственно

        t = time.time()
        imu_lin_a, imu_ang_w, dt = self.get_imu_data()
        self.orientation[0] += (imu_ang_w.x - self.gyro_bias[0]) * dt
        self.orientation[2] += (imu_ang_w.z - self.gyro_bias[2]) * dt

        # abs_a = sqrt(imu_lin_a.x ** 2 + imu_lin_a.y ** 2 + imu_lin_a.z ** 2)
        #
        # theta_g = degrees(asin(imu_lin_a.z / abs_a))  # тангаж
        # phi_g = degrees(asin(imu_lin_a.x / sqrt(imu_lin_a.x ** 2 + imu_lin_a.y ** 2)))  # крен
        # print("Measurement time: ", time.time() - t)

        print(np.degrees(self.orientation), 'UNITS: degrees')


cam = RealSense()

t0 = time.time()
cam.est_initial_pitch_roll()
print("Calibration time: ", time.time() - t0)
print(np.degrees(cam.orientation), 'UNITS: degrees')

# while True:
#     cam.estimate_current_orientation()
#     # time.sleep(0.1)


# img = cam.getRGBImage()
# d_img = cam.getDepthImage()
# cv2.imshow('color', cv2.cvtColor(img, cv2.COLOR_RGB2GRAY))
# cv2.imshow('depth', d_img)


# ttt = cam.visualize_pcd_sequence(30, 1)
# fig, ax = plt.subplots()
# ax.plot(ttt)
# plt.show()
sd = time.time()
cam.visualize_single_pcd()
print("Alignment time:", time.time() - sd)
# cam.visualize_depth_image_sequence()

'''
normal_verts = np.ndarray([1, 3], dtype=float)

with open('some.csv', newline='') as f:
    reader = csv.reader(f)
    i = 0
    for row in reader:
        if not i:
            normal_verts[0, :] = np.asarray([float(row[0]), float(row[1]), float(row[2])]).reshape([1, 3])
            i += 1
        else:
            normal_verts = np.append(normal_verts, np.asarray([float(row[0]), float(row[1]), float(row[2])]).reshape([1, 3]), axis=0)

for_view_verts = normal_verts @ view_matrix
for_view_pcd = open3d.geometry.PointCloud()
for_view_pcd.points = open3d.utility.Vector3dVector(for_view_verts)
open3d.visualization.draw_geometries([for_view_pcd], window_name='recovered normal')
'''

