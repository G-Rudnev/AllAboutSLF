import pyrealsense2 as rs
import numpy as np
import cv2
import open3d

class RealSense:
    def __init__(self):
        # Create a pipeline
        self.pipeline = rs.pipeline()
        # Create a config and configure the pipeline to stream
        #  different resolutions of color and depth streams
        config = rs.config()
        colorizer = rs.colorizer()
        # Get device product line for setting a supporting resolution
        pipeline_wrapper = rs.pipeline_wrapper(self.pipeline)
        pipeline_profile = config.resolve(pipeline_wrapper)
        device = pipeline_profile.get_device()
        device_product_line = str(device.get_info(rs.camera_info.product_line))
        config.enable_stream(rs.stream.depth, 1280, 720, rs.format.z16, 30)
        config.enable_stream(rs.stream.color, 1280, 720, rs.format.rgb8, 30)
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

    def getRGBImage(self):
        frame = self.pipeline.wait_for_frames()
        color_frame = frame.get_color_frame()
        color_image = np.asanyarray(color_frame.get_data())
        return color_image

    def getDepthImage(self):
        frame = self.pipeline.wait_for_frames()
        depth_frame = frame.get_depth_frame()
        depth_image = np.asanyarray(depth_frame.get_data())
        return depth_image

    def getPointcloud(self):
        pc = rs.pointcloud()
        decimate = rs.decimation_filter()
        frame = self.pipeline.wait_for_frames();
        depth_frame = decimate.process(frame.get_depth_frame())
        points = pc.calculate(depth_frame)
        print(points)
        v = points.get_vertices()
        verts = np.asanyarray(v).view(np.float32).reshape(-1, 3)  # xyz

        # зададим ROI для облака точек, чтобы не тянуть лишние шумы и блики
        xmin, xmax, ymin, ymax, zmin, zmax = -0.5, 0.5, -0.5, 0.5, 0, 1
        ind1 = np.all(verts > np.array([xmin, ymin, zmin]), axis=1)
        ind2 = np.all(verts < np.array([xmax, ymax, zmax]), axis=1)

        return verts[np.logical_and(ind1, ind2)]


cam = RealSense()
color_img = cam.getRGBImage()
cv2.imshow('Color', color_img)

normal_verts = cam.getPointcloud()
cutted_verts = [[0, 0, 0]]
# print(verts)
for point in normal_verts:
    # неэффективный алгоритм в лоб. Поскольку на картинке положение тележки строго горизонтально полу,
    # и тележка стоит перед наклонной плоскостью "ровно" и сама плоскость имеет равномерный наклон (в одной плоскости)
    # Переберем все облако и выкинем ненжные точки сравнивая угол между началом СК и точкой с заданным углом.
    # z вперед х вправо у вниз

    h = 0.15  # высота камеры
    alpha = 45  # заданный угол

    p = np.array(point[1], point[2])
    a = np.array([0 - h, 0])
    b = np.array([0 - h + np.sin(np.deg2rad(alpha)), 1])

    ab = b - a
    ap = p - a
    angle = np.arccos(np.dot(ab, ap) / (np.linalg.norm(ab) * np.linalg.norm(ap)))
    angle = np.rad2deg(angle)

    #НЕУДАЧНОЕ МЕСТО, APPEND НЕ ДОПУСТИМ
    if angle - alpha > 0:
        cutted_verts.append([point[0], point[1], point[2]])
    else:
        1 + 1

#визуализация
normal_pcd = open3d.geometry.PointCloud()
normal_pcd.points = open3d.utility.Vector3dVector(normal_verts)

cutted_pcd = open3d.geometry.PointCloud()
cutted_pcd.points = open3d.utility.Vector3dVector(np.array(cutted_verts))

open3d.visualization.draw_geometries([normal_pcd], window_name='normal')
open3d.visualization.draw_geometries([cutted_pcd], window_name='cutted')
