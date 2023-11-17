import pyrealsense2 as rs
import numpy as np
from numpy.linalg import norm
import time
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
import threading
import sys
from RoboMath import *   #always needed
from Config import *     #always needed

from PointCloudLidarification import *
import lidarVector

__all__ = ['RealSense']

class RealSense:

    _rnd_key = 9846    #it is just a random number
    _N_of_RS = 0

    @staticmethod
    def __internal_init(cam):
        ''' Something like init and reinit method for Realsense configuration.
        Used in __init__() and Stop() before Start function. Calling this method is URGENT'''
        # Cam declaration rules for sensor streams and frames
        if cam.__sensors_mode == 1: # accel (250 Hz)
            # Accelerometer available FPS: {63, 250}Hz
            cam.__config.enable_stream(rs.stream.accel, rs.format.motion_xyz32f, 250)  # accelerometer
            cam.__accel_frame_id = 0

        elif cam.__sensors_mode == 2: # IMU: accel + gyro (200 Hz)
            # Accelerometer available FPS: {63, 250}Hz
            cam.__config.enable_stream(rs.stream.accel, rs.format.motion_xyz32f, 250)  # accelerometer
            cam.__accel_frame_id = 0
            # Gyroscope available FPS: {200,400}Hz
            cam.__config.enable_stream(rs.stream.gyro, rs.format.motion_xyz32f, 200)  # gyroscope
            cam.__gyro_frame_id = 1

        elif cam.__sensors_mode == 3: # full set: depth + accel + gyro (30 Hz)
            cam.__config.enable_stream(rs.stream.depth, 1280, 720, rs.format.z16, 30) # depth
            cam.__depth_frame_id = 0
            # Accelerometer available FPS: {63, 250}Hz
            cam.__config.enable_stream(rs.stream.accel, rs.format.motion_xyz32f, 250)  # accelerometer
            cam.__accel_frame_id = 1
            # Gyroscope available FPS: {200,400}Hz
            cam.__config.enable_stream(rs.stream.gyro, rs.format.motion_xyz32f, 200)  # gyroscope
            cam.__gyro_frame_id = 2
        # No need for frame indexification
        elif cam.__sensors_mode == 4: # depth (30 Hz)
            cam.__config.enable_stream(rs.stream.depth, 1280, 720, rs.format.z16, 30) # depth
            cam.__config.enable_stream(rs.stream.color, 1280, 720, rs.format.rgb8, 30) # rgb
            cam.__N_cloud_pts = 230400 # according to RealSense
        else:
            print("Check mode")
            cam.__sensors_mode = 4

    def __init__(self, rnd_key, RS_ID, vis=False, do_plot=False, sensors_mode=4):
        "Use create function only!!!"

        if (rnd_key != RealSense._rnd_key):
            print("Use RealSense.Create function only!!!", exc = True) # Exception about exc parameter

        # REALSENSE CAMERA INITIALIZATION SPECIFIC PART

        self.__sensors_mode = sensors_mode

        # Create a pipeline
        self.__pipeline = rs.pipeline()
        self.__config = rs.config()
        self.__profile = None

        RealSense.__internal_init(self)

        # INTERNAL PURPOSE FIELDS
        self._is_visual = vis
        self._do_plot = do_plot
        if self._do_plot:
            if self._do_plot:
                self.fig = plt.figure()
                if self._is_visual:
                    ax1 = self.fig.add_subplot(221)
                    ax2 = self.fig.add_subplot(222)
                    ax3 = self.fig.add_subplot(223)
                    ax3.set_aspect('equal')
                    ax4 = self.fig.add_subplot(224)
                    ax4.set_aspect('equal')
                    self.ax = [[ax1, ax2], [ax3, ax4]]

                else:
                    self.ax = self.fig.add_subplot(111)
        self._point_cloud = rs.pointcloud()
        self._filter = rs.decimation_filter()

        # VISUAL DATA
        if self._is_visual:
            self._color_image = np.zeros([720, 1280, 3], dtype=np.uint8)# color image (resolution considered shape)
            self._depth_image = np.zeros([720, 1280], dtype=np.uint16)# depth image (internally fixed? shape)
            # self._images = np.asanyarray([self._color_image, self._depth_image])

        #IDs
        self.RS_ID = RS_ID
        RealSense._N_of_RS += 1
        self.timestamp = 0   # last captured public timestamp
        self.composite_frame = [] # last public composite frame
        self._composite_frame = [] # last got composite frame

        #LOCKS
        self._thread = []
        self._mutex = threading.RLock()

        #EVENTS AND HANDLES
        self.ready = threading.Event()
        self.ready.clear()
        self._isFree = threading.Event()
        self._isFree.set()

        # PRIMARY DATA (POINT CLOUD PROCESSING - CONTOURING)
        self._primary_points = np.zeros([3, self.__N_cloud_pts], dtype=np.float64) # raw Cartesian coordinates
        self._are_points_in_roi = np.zeros([self.__N_cloud_pts], dtype=bool) # flags wether each point is in region of interest (trimming parameters)
        self._N_roi_pts = np.zeros([1], dtype=np.int64)
        self._roi_polar_coords = np.zeros([2, self.__N_cloud_pts], dtype=np.float64)
        self.min_dist = float(mainDevicesPars[f"rs_min_dist_ID{self.RS_ID}"]) # RealSense cannot sense depth closer than 0.5 m
        self.max_dist = float(mainDevicesPars[f"rs_max_dist_ID{self.RS_ID}"]) # used for left, right, front trimming
        self.min_y = float(mainDevicesPars[f"rs_min_y_ID{self.RS_ID}"]) # used for up trimming (y-axis looks down)
        self.max_y = float(mainDevicesPars[f"rs_max_y_ID{self.RS_ID}"]) # used for doun trimming (y-axis looks down)
        self._trimming_pars = (self.min_dist, self.max_dist, self.min_y, self.max_y) # unchangable        

        # PREVECTORIZATION DATA
        self.N = 200
        self._xy = np.zeros([3, self.N], dtype = np.float64)
        self._xy[2, :] = 1.0
        self._phi = np.zeros([self.N], dtype = np.float64)
        self._Npnts = np.array([self.N], dtype = np.int64)

        # VECTORIZATION PARAMETERS
        self.half_width = float(globalPars["half_width"])
        self.half_length = float(globalPars["half_length"])
        self.safetyBox = float(globalPars["safetyBox"])
        self.dist_range = float(mainDevicesPars[f"rs_range_ID{self.RS_ID}"])
        self.half_dphi = float(mainDevicesPars[f"rs_half_dPhi_ID{self.RS_ID}"])
        self.regression_tolerance = float(mainDevicesPars[f"rs_regression_tolerance_ID{self.RS_ID}"])
        self.mount = np.zeros([2], dtype=np.float64)
        for i, el in enumerate(mainDevicesPars[f"rs_mount_ID{self.RS_ID}"].split()):
            if i > 1:
                break
            self.mount[i] = float(el)

        self._cppPars = (self.half_width, self.half_length, self.safetyBox / 100.0, self.dist_range, self.half_dphi, self.regression_tolerance, self.mount)

        # OUTPUT DATA
        self._linesXY = np.zeros([3, self.N], dtype=np.float64)
        self._linesXY[2, :] = 1.0
        self._Nlines = np.zeros([1], dtype=np.int64)
        self._gapsIdxs = np.zeros([self.N], dtype=np.int64)
        self._Ngaps = np.zeros([1], dtype=np.int64)

        # Extension module initialization
        self.cppID = lidarVector.init(self._xy, self._phi, self._Npnts, self._linesXY, self._Nlines, self._gapsIdxs, self._Ngaps, self._cppPars)

        

    # SENSOR DATA COLLECTION
    def _extract_sensors_data(self, first=False):
        ''' Extract data from each sensor separately
        pyrealsense has time delay for nearly 15 seconds when RS is disconnected before it raises an Esception
        during this time RS can be reconnected and start streaming without pain'''

        try:
            if self.__sensors_mode == 4:
                if first:
                    # self._composite_frame = self.__pipeline.wait_for_frames(timeout_ms=1000)
                    t0 = time.time()
                    while True:
                        self._composite_frame = self.__pipeline.poll_for_frames()
                        if not self._composite_frame:
                            if time.time() - t0 > 2.0:
                                raise Exception('Initial frame delivery timeout. Check RealSense connection.')
                            time.sleep(0.001)
                        else:
                            break
                else:
                    t0 = time.time()
                    while True:
                        self._composite_frame = self.__pipeline.poll_for_frames()
                        if not self._composite_frame:
                            if time.time() - t0 > 0.5:
                                raise Exception('Frame delivery timeout. Check RealSense connection.')
                            time.sleep(0.001)
                        else:
                            break

                if self._composite_frame.get_depth_frame().height == 720 and self._composite_frame.get_depth_frame().width == 1280:
                    self._primary_points[:, :] = \
                        np.asarray(self._point_cloud.calculate(self._filter.process(self._composite_frame.get_depth_frame())).get_vertices(dims=2), dtype=np.float64).T[:, :] # To NumPy array xyz coords (3x230400)

                if self._is_visual:
                    self._color_image[:,:,:] = np.asanyarray(self._composite_frame.get_color_frame().get_data())[:,:,:]
                    self._depth_image[:,:] = np.asanyarray(self._composite_frame.get_depth_frame().get_data())[:,:]

                self.composite_frame = self._composite_frame

        # TODO: make an exception handler
        except Exception as e:
            print('Extraction error:', e)
            if self.ready.is_set():
                self.Stop(False) # means man_stop = False, means it's not a manually caused stop (= RS has problems)


    @classmethod
    def Create(cls, RS_ID, vis=False, do_plot=False):
        try:
            cam = RealSense(cls._rnd_key, RS_ID, vis=vis, do_plot=do_plot)
            cam.Start = cam._Start_wrp(cam.Start)
            return cam
        except:
            if (sys.exc_info()[0] is not RoboException):
                print(f"RealSense {cam.RS_ID} error! {sys.exc_info()[1]}; line: {sys.exc_info()[2].tb_lineno}")
            return None
    
    def _Start_wrp(self, f):
        def func(*args, **kwargs):
            if (self._isFree.wait(timeout = 0.5)):  #for the case of delayed stopping
                self._thread = threading.Thread(target=f, args=args, kwargs=kwargs)
                self._thread.start()
                while self._thread.is_alive() and not self.ready.is_set():
                    time.sleep(0.01)
            return self.ready.is_set()
        return func

    
    def Start(self):
        print('RealSense', self.RS_ID, 'thread started')
        print('Waiting for RealSense to start')
        self._isFree.clear()
        '''Search for camera plugged in is provided as internal functionality in pyrealsense (Windows at least)
        pyrealsense has time delay for nearly 15 seconds when RS is not connected before it raises an Esception
        during this time RS can be connected and start streaming without pain'''
        try:
            self.__profile = self.__pipeline.start(self.__config)
        except Exception as e:
            self.ready.clear()
            self._isFree.set()
            print('RealSense', self.RS_ID, 'thread finished')
            return
        else:
            print("RealSense detected")

        self.ready.set()
        self._extract_sensors_data(first=True) # required for the first after RS launch frame extraction 
        while (self.ready.is_set() and threading.main_thread().is_alive()):
            if not self.__profile:
                return
            self._extract_sensors_data() # Collecting data from different sensor and saving
            # print('Data extraction time:', time.time() - t0)
            with self._mutex:
                t0 = time.time()
                process_point_cloud(self._primary_points, self._are_points_in_roi, \
                    self._roi_polar_coords, self._N_roi_pts, \
                    self._xy, self._phi, self._Npnts, *self._trimming_pars) # Y axis looks down
                lidarVector.calcLines(self.cppID)
                lidarVector.synchronize(self.cppID)
            print('Time:', time.time() - t0, 'N ROI points:', self._N_roi_pts[0], 'N output lines', self._Nlines[0])

        self.ready.clear()
        self._isFree.set()
        print('RealSense', self.RS_ID, 'thread finished.')


    def Stop(self, man_stop=True):
        ''' RealSense can be stopped in two ways:
        1. Manually, which means we command it to stop
        2. Internally by pyrealsense when exceptions are raised
        Both these ways require reinitialization of internal configurations (pipeline, config, profile, etc)'''
        # if man_stop: # this is used to differentiate manual stop from internal staop caused by pyRS. If it is manual - stop pipeline,
        #     try: 
        #         self.__pipeline.stop() # # if internal - it's already stopped and this line will cause a crush
        #         print('RealSense manual stop')
        #     except:
        #         pass
        # else:
        #     print('Realsense internal stop')
        try: 
            self.__pipeline.stop() # # if internal - it's already stopped and this line will cause a crush
            # print('RealSense manual stop')
        except:
            pass
        self._isFree.set()
        self.ready.clear()
        # Necessary part for internal RS reinitialization
        self.__pipeline = rs.pipeline()
        self.__config = rs.config()
        self.__profile = None
        RealSense.__internal_init(self) # NECESSARY
        

    def Release(self):
        print('Py releasing')
        lidarVector.release(self.cppID)


    def GetLinesXY(self, linesXY, pauseIfOld=0.0):
        with self._mutex:
            if not self.ready.is_set():
                return -1
            if self.composite_frame:
                if self.timestamp != self.composite_frame.timestamp:
                    self.timestamp = self.composite_frame.timestamp
                    return linesXY.Fill(self._linesXY, self._gapsIdxs, int(self._Nlines[0]), int(self._Ngaps[0]))
            else:
                return -1
        time.sleep(pauseIfOld)
        return 0 #the trick is - even if the pointcloud is empty there are 3 elements in linesXY






# ============== DEPRECATED FUNCTIONALITY ==========================

# It's quite obvious
'''
# while True:
#     imu.estimate_current_orientation()
#     print(np.degrees(imu.orientation), 'UNITS: degrees')
'''

# Sensors reinitialization when initial orientation is estimated with accel working alone 
'''
# # Relaunch with full set: depth, accel, gyro
# cam = RealSense(3)
# cam.orientation = accel.orientation # optionally chosen from processed and raw data analysis
# cam.visualize_single_pcd()
# cam.visualize_pcd_sequence(30, 10)
# while True:
#     cam.estimate_current_orientation()
#     print(np.degrees(cam.orientation), 'UNITS: degrees')
'''


# Waiting for different sensors complete initialization
'''
# while time.time() - t0 < 1:
    # while True:
    #     depth_comp_frame_size = depth_sensor.pipeline.poll_for_frames().size()
    #     # print(depth_comp_frame_size)
    #     if not depth_comp_frame_size:
    #         # if imu.pipeline.poll_for_frames().size():
    #         #     lin_a_acquired[n_meas], ang_w_acquired[n_meas], _ = imu.get_imu_data()
    #         #     # print(ang_w_acquired[n_meas])
    #         #     n_meas += 1
    #         #     # imu.estimate_current_orientation(imu_ang_w_prev, t_prev)
    #         depth_sensor.orientation = imu.get_current_orinetation()
    #     else:
    #         break

    # while not depth_sensor.pipeline.poll_for_frames.size():
    
    #     time.sleep(0.0001)
    
    # a = depth_sensor.getPointCloud().size
    # print(imu.get_current_position_and_orinetation_from_IMU()[0][0], imu.lin_velocity[0])
    '''

# Voxelization
'''
def maintain_voxelization(raw_points, voxel_size=0.1):
    verts = np.asarray(raw_points.get_vertices()).view(np.float32).reshape(-1, 3)  # xyz
    # using matrix multiplication increases time-efficiency (paradox) in comparison with raw data. That's why ones 3x3 matrix is used
    # for_view_verts = verts @ view_matrix # view matrix is used for visualization via Open3D library (may be deprecated but axes must be considered when PointCloud is processed)
    # for_view_pcd.points = open3d.utility.Vector3dVector(for_view_verts)
    # for_view_pcd.points = open3d.utility.Vector3dVector(verts)
    
    # downsampled_points = np.asarray(for_view_pcd.voxel_down_sample(voxel_size=voxel_size).points)
    downsampled_points = verts
    return downsampled_points
'''

# Estimation of initial cam orientation (pitch, roll) according to g-vector
'''
# Accelerometer only launch for initial orientation estimation (calibration)
accel = RealSense(1) # only accelerometer
# accel.est_initial_orientation(False) # non-processed accel measurements
# print(orientation_from_raw)
t0 = time.time()
accel.est_initial_orientation(True)
# print(orientation_from_processed)
print("Calibration time: ", time.time() - t0)
accel.pipeline.stop()
print(np.degrees(accel.orientation), 'UNITS: degrees')
'''


# Usage of IMU data collection function for faster performance
'''
# Parallelization of IMU and depth-sensor measurements 
imu = RealSense(2) # accel + gyro
imu.orientation = accel.orientation
imu_thread = threading.Thread(target=imu.imu_processing, daemon=True)
imu_thread.start()
'''


# Some data visualization
'''
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
        downpcd = for_view_pcd.voxel_down_sample(voxel_size=0.001)
        # print(np.asarray(for_view_pcd.points).shape[0], np.asarray(downpcd.points).shape[0])

        downpcd.points = open3d.utility.Vector3dVector(self.pcd_rotate(downpcd))

        open3d.visualization.draw_geometries([downpcd])
        # open3d.visualization.draw_geometries([for_view_pcd], window_name='normal')

    # Показать последовательность облаков точек. Input: время работы в секундах, "частота обновления"
    def visualize_pcd_sequence(self):
        global normal_verts

        # Create an o3d.visualizer class
        vis = open3d.visualization.Visualizer()
        vis.create_window()
        for_view_pcd = open3d.geometry.PointCloud()
        downpcd = open3d.geometry.PointCloud()
        t_prev = time.time()
        _, imu_ang_w_prev, t_prev = self.get_imu_data()
        while True:
            normal_verts = self.getPointCloud()
            for_view_verts = normal_verts @ view_matrix
            for_view_pcd.points = open3d.utility.Vector3dVector(for_view_verts)
            downpcd.clear()  # очистка облака от предыдущих вокселей
            downpcd = for_view_pcd.voxel_down_sample(voxel_size=0.001)
            imu_ang_w_prev, t_prev = self.estimate_current_orientation(imu_ang_w_prev, t_prev)
            downpcd.points = open3d.utility.Vector3dVector(self.pcd_rotate(downpcd))
            # print(np.asarray(for_view_pcd.points).shape[0], np.asarray(downpcd.points).shape[0])
            # Put point cloud into visualizer
            vis.add_geometry(downpcd)
            # Let visualizer render the point cloud
            vis.update_geometry(downpcd)
            vis.update_renderer()
            vis.poll_events()


    def visualize_depth_image_sequence(self):
        try:
            while True:
                depth_image = self.getDepthImage()
                # print(np.min(depth_image), np.max(depth_image))
                depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(depth_image, alpha=0.075), cv2.COLORMAP_JET)

                cv2.imshow('Depth image', depth_colormap)
                cv2.waitKey(1)
                time.sleep(0.1)

        except KeyboardInterrupt:
            pass

        finally:
            self.pipeline.stop()
            cv2.destroyAllWindows()


    def visualize_RGB_image_sequence(self):
        try:
            while True:
                image = self.getRGBImage()
                # depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(depth_image, alpha=0.05), cv2.COLORMAP_JET)
                cv2.imshow('RGB image', image)
                cv2.waitKey(1)
                time.sleep(0.1)

        except KeyboardInterrupt:
            pass

        finally:
            self.pipeline.stop()
            cv2.destroyAllWindows()
    '''

# IMU PROCESSING
'''
    # достаём данные IMU
    def get_imu_data(self):  
        frame = self.pipeline.wait_for_frames()
        ang_w_data = [] # for mode 1
        if len(frame) > 1:
            ang_w_data = frame[self.gyro_frame_id].as_motion_frame().get_motion_data()  # угловые скорости
        lin_a_data = frame[self.accel_frame_id].as_motion_frame().get_motion_data()  # линейные ускорения
        return lin_a_data, ang_w_data, time.time() # data + moment of the measurement

    # If avg true then real-time window averaging is maintained, else raw measurements are used
    def est_initial_orientation(self, avg): 
        # TODO: CREATE FUNCTION FOR RECALIBRATION WHEN THE BOT STOPS
        # print('Initial pitch and roll estimation. Waiting for IMU to stabilize')
        print('Maintaining initial pitch and roll estimation')
        # stabilization_meas_num = 10  # число измерений для стабилизации
        calibration_meas_num = 50  # число используемых в оценке измерений
        # x, y, z, abs_a = [], [], [], []
        x = np.zeros((calibration_meas_num), dtype=float)
        y = np.zeros((calibration_meas_num), dtype=float)
        z = np.zeros((calibration_meas_num), dtype=float)
        abs_a = np.zeros((calibration_meas_num), dtype=float)
        
        
        # th, phi = [], []

        # time.sleep(45.0)

        # for k in range(10):
        #     x_sum = 0.0
        #     y_sum = 0.0
        #     z_sum = 0.0
        #     abs_sum = 0.0
        #     for g in range(50):
        #         lin_a, _, _ = self.get_imu_data()
        #         x_sum += lin_a.x
        #         y_sum += lin_a.y
        #         z_sum += lin_a.z
        #         abs_sum += sqrt(lin_a.x ** 2 + lin_a.y ** 2 + lin_a.z ** 2)
        #         # abs_sum += 9.80665

        #     th.append(degrees(asin(z_sum / abs_sum)))  # тангаж
        #     phi.append(degrees(asin(x_sum / sqrt(x_sum ** 2 + y_sum ** 2))))  # крен
        #     x.append(x_sum/50)
        #     y.append(y_sum/50)
        #     z.append(z_sum/50)
        

        k = 10 # averaging window
        if avg:
            delta = 0.0 # full acceleration abs value deviation from abs(g)
            for i in range(calibration_meas_num):
                lin_a, _, _ = self.get_imu_data()

                if i < k-1:
                    x[i] = lin_a.x
                    y[i] = lin_a.y
                    z[i] = lin_a.z
                # Filling with window-averaged measurements and deviation correction
                else:
                    x[i] = (lin_a.x + np.sum(x[i-k+1:i])) / k + delta * x[i-1] / abs_a[i-1]
                    y[i] = (lin_a.y + np.sum(y[i-k+1:i])) / k + delta * y[i-1] / abs_a[i-1]
                    z[i] = (lin_a.z + np.sum(z[i-k+1:i])) / k + delta * z[i-1] / abs_a[i-1]

                abs_a[i] = sqrt(x[i] ** 2 + y[i] ** 2 + z[i] ** 2)
                delta = 9.80665 - abs_a[i]
        else:
            for i in range(calibration_meas_num):
                lin_a, _, _ = self.get_imu_data()
                x[i] = lin_a.x
                y[i] = lin_a.y
                z[i] = lin_a.z
                abs_a[i] = sqrt(x[i] ** 2 + y[i] ** 2 + z[i] ** 2) 

        if k:
            a_x_est = np.mean(x[-k:-1])
            a_y_est = np.mean(y[-k:-1])
            a_z_est = np.mean(z[-k:-1])
            a_est = np.mean(abs_a[-k:-1])
            self.orientation[0] = asin(a_z_est / a_est) # pitch
            self.orientation[2] = asin(a_x_est / sqrt(a_x_est ** 2 + a_y_est ** 2)) # roll

    def estimate_current_position_and_orientation_from_IMU(self, imu_lin_a_prev, imu_ang_w_prev, t_prev):

        imu_lin_a, imu_ang_w, t = self.get_imu_data()
        dt = t - t_prev
        self.orientation[0] += (imu_ang_w.x + imu_ang_w_prev.x) * dt / 2 
        self.orientation[2] -= (imu_ang_w.z + imu_ang_w_prev.z) * dt / 2 # why minus?.. Answer: due to deprecated rotation matrix


        lin_vel_prev = self.lin_velocity[0]
        # For x axis 
        self.lin_velocity[0] += (imu_lin_a_prev.x + imu_lin_a.x) * dt / 2 
        self.position[0] += (lin_vel_prev + self.lin_velocity[0]) * dt / 2

        return imu_lin_a, imu_ang_w, t


    # Target function for separate thread where IMU data is collected and orientation is estimated
    def imu_processing(self):
        self.n_meas = 0
        imu_lin_a_prev, imu_ang_w_prev, t_prev = imu.get_imu_data()
        # lin_vel_prev = self.lin_velocity # def structure for velocities?...
        while True:
            # t0 = time.time()
            imu_lin_a_prev, imu_ang_w_prev, t_prev = \
                self.estimate_current_position_and_orientation_from_IMU(imu_lin_a_prev, imu_ang_w_prev, t_prev)
            # TODO Insert lock, while all 3 axes aren't oriented
            self.n_meas += 1
            # print(imu_lin_a_prev.x, imu_ang_w_prev.x)
            # time.sleep(0.00001)

    def get_current_position_and_orinetation(self):
        self.n_meas = 0
        return self.position, self.orientation
'''

# Some functionality for 
'''
    def getRGBImage(self):  # НА ВЫХОДЕ КАКОЙ-ТО НЕ СОВСЕМ RGB ДА ЕЩЁ И ТЁМНЫЙ
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
        pc = rs.pointcloud()
        decimate = rs.decimation_filter()
        frame = self.pipeline.wait_for_frames()
        depth_frame = decimate.process(frame.get_depth_frame())
        points = pc.calculate(depth_frame)
    
        v = points.get_vertices()
        verts = np.asanyarray(v).view(np.float32).reshape(-1, 3)  # xyz
    
        return verts

    def pcd_rotate(self, pcd):
        trans_pcd = np.asarray(pcd.points).T
        pitch = self.orientation[0]
        roll = self.orientation[2]
        self.rot_matrix[:, :] = \
            np.asarray([[1, 0, 0], [0, cos(pitch), -sin(pitch)], [0, sin(pitch), cos(pitch)]], dtype=float) @ \
            np.asarray([[cos(roll), -sin(roll), 0], [sin(roll), cos(roll), 0], [0, 0, 1]], dtype=float)

        return np.transpose(self.rot_matrix @ trans_pcd)
'''

# FROM RealSense __init__()
'''
        # DEPRECATED
        # # Frames numeration control
        # pipeline_profile = self.pipeline.get_active_profile()
        # frames_num = len(pipeline_profile.get_streams())
        # # print(pipeline_profile.get_streams())
        # if frames_num == 1: # FOR INITIAL ORIENTATION ESTIMATION
        #     self.accel_frame_id = 0
        # elif  frames_num == 2: # FOR TESTS AND ANY OTHER PURPOSES
        #     self.accel_frame_id = 0
        #     self.gyro_frame_id = 1
        # elif frames_num == 3: # COMPLETE REQUIRED SET
        #     self.depth_frame_id = 0
        #     self.accel_frame_id = 1
        #     self.gyro_frame_id = 2
        # else:
        #     print("Check sensors' streams. Color image stream might be declared")
        

        if sensors_mode > 2:
            # Getting the depth sensor's depth scale (see rs-align example for explanation)
            depth_sensor = profile.get_device().first_depth_sensor()
            depth_scale = depth_sensor.get_depth_scale()
            print("Depth Scale is: ", depth_scale)
            # Create an align object
            # rs.align allows us to perform alignment of depth frames to others frames
            # The "align_to" is the stream type to which we plan to align depth frames.
            align_to = rs.stream.color
            align = rs.align(align_to)
'''