import numpy as np
import time

__all__ = ['process_point_cloud']

# =======================================================================
# ============= POTENTIALLY FOR C++ EXTENSION MODULE PART ===============
# =======================================================================

# # TODO: verticalization
# def maintain_verticalization():

# Trimming function # TODO arrays instead of cam object
def maintain_trimming(primary_points: np.ndarray, are_points_in_roi: np.ndarray, N_roi_pts: np.ndarray, \
                    min_dist=0.5, max_dist=3.0, min_y=-0.6, max_y=0.6):
    # Add y = kz + b trimming 
    xmin, xmax, ymin, ymax, zmin, zmax = -max_dist, max_dist, min_y, max_y, min_dist, max_dist # X - right, Y - down, Z - forward (as camera looks forwards) !!!according to CS of the camera!!!
    ind1 = np.all(primary_points > np.array([[xmin], [ymin], [zmin]]), axis=0)
    ind2 = np.all(primary_points < np.array([[xmax], [ymax], [zmax]]), axis=0)
    # Determination of all the points inside the region of interest
    are_points_in_roi[:] = np.logical_and(ind1, ind2)[:]
    N_roi_pts[0] = np.sum(are_points_in_roi)

# # Projection function
# def maintain_projection(trimmed_points: np.ndarray) -> np.ndarray:
#     return np.asarray([trimmed_points[2, :], -trimmed_points[0, :]])

# Contouring function
def maintain_contouring(primary_points: np.ndarray, are_points_in_roi: np.ndarray, \
                    roi_polar_coords: np.ndarray, N_roi_pts: np.ndarray, \
                    xy: np.ndarray, phi: np.ndarray, Npnts: np.ndarray):
    roi_polar_coords[0, :N_roi_pts[0]] = np.arctan2(-primary_points[0, are_points_in_roi], \
            primary_points[2, are_points_in_roi])[:]
    roi_polar_coords[1, :N_roi_pts[0]] = np.sqrt(np.power(primary_points[2, are_points_in_roi], 2) + \
        np.power(primary_points[0, are_points_in_roi], 2))
    angle_sorted_idxs = np.flip(np.argsort(roi_polar_coords[0, :N_roi_pts[0]]))
    # For any case when angle-sorted polar frame is needed
    roi_polar_coords[0, :N_roi_pts[0]] = roi_polar_coords[0, angle_sorted_idxs]
    roi_polar_coords[1, :N_roi_pts[0]] = roi_polar_coords[1, angle_sorted_idxs]
    primary_points[::2, :N_roi_pts[0]] = primary_points[::2, are_points_in_roi][:, angle_sorted_idxs]
    

    # ''' # Line 1 (Y-coordinates) has no informational use now. Thus, it is exploited as a container for 
    # # polar angles of the projected points which may be extracted via _are_points_in_roi. 
    # # Minus for line 0 is used according to coordinate system of the camera '''
    # cam._primary_points[1, cam._are_points_in_roi] = \
    #     np.arctan2(-cam._primary_points[0, cam._are_points_in_roi], \
    #         cam._primary_points[2, cam._are_points_in_roi])[:]

    # ''' # Then line 1 (polar angles) is reverse-sorted to transform lines 0 and 2 
    # (projected points X and Z) to a clockwise sequence when only the points in roi 
    # (according to _are_points_in_roi) are accessed'''
    # # cam._primary_points[1, cam._are_points_in_roi] = np.flip(np.sort(cam._primary_points[1, cam._are_points_in_roi]))[:]
    # # cam._are_points_in_roi[] = np.flip(cam._primary_points[1, cam._are_points_in_roi].argsort())
    # idxs = np.flip(np.argsort(cam._primary_points[1, cam._are_points_in_roi]))
    # cam._primary_points[::2, cam._are_points_in_roi] = \
    #     cam._primary_points[::2, np.flip(np.argsort(cam._primary_points[1, cam._are_points_in_roi]))]
    # # print(cam._primary_points[1, cam._are_points_in_roi])

    # Contouring 
    # fov_angle = polar_frame[0, 0] - polar_frame[0, -1]
    angle = np.pi / 4.0
    delta_ang = 0.5 * np.pi / Npnts[0]
    cnt_pt_id = 0
    pt_id = 0
    slice_width = 0

    # Cyclic sectoral nearest point extraction (clockwise sequence forming)
    while True:
        t0 = time.time()
        while angle > roi_polar_coords[0, pt_id+slice_width] > angle - delta_ang:
            if pt_id + slice_width < N_roi_pts[0] - 1:
                slice_width += 1
            else: 
                break
        if slice_width == 0:
            xy[0, cnt_pt_id] = 0.0
            xy[1, cnt_pt_id] = 0.0
            phi[cnt_pt_id] = angle - delta_ang / 2
        else:
            min_d_pt_id = np.argmin(roi_polar_coords[1, pt_id:pt_id+slice_width])
                # np.argmin( \
                #     np.sqrt( \
                #         np.dot( \
                #             cam._primary_points[0, cam._are_points_in_roi][pt_id:pt_id+slice_width], \
                #                 cam._primary_points[0, cam._are_points_in_roi][pt_id:pt_id+slice_width])))
                    # np.sqrt( \
                    #     np.power(\
                    #         cam._primary_points[0, cam._are_points_in_roi][pt_id:pt_id+slice_width], 2) + \
                    #             np.power( \
                    #                 cam._primary_points[2, cam._are_points_in_roi][pt_id:pt_id+slice_width], 2)))
            
            xy[0, cnt_pt_id] = primary_points[2, pt_id+min_d_pt_id] # Same 
            xy[1, cnt_pt_id] = -primary_points[0, pt_id+min_d_pt_id] #     as projection
            phi[cnt_pt_id] = roi_polar_coords[0, pt_id+min_d_pt_id]

        angle -= delta_ang
        pt_id += slice_width
        slice_width = 0
        cnt_pt_id += 1
        if cnt_pt_id == Npnts[0]:
            break

        # print(time.time() - t0, '\n')



# Full processing: verticalization, trimming, horizontal projection, contour extraction
def process_point_cloud(primary_points: np.ndarray, are_points_in_roi: np.ndarray, \
                    roi_polar_coords: np.ndarray, N_roi_pts: np.ndarray, \
                    xy: np.ndarray, phi: np.ndarray, Npnts: np.ndarray, \
                    min_dist=0.5, max_dist=3.0, min_y=-0.6, max_y=0.6):
    t0 = time.time()
    # Assuming raw points are always vertically aligned:
    maintain_trimming(primary_points, are_points_in_roi, N_roi_pts, \
                    min_dist=0.5, max_dist=max_dist, min_y=min_y, max_y=max_y)
    maintain_contouring(primary_points, are_points_in_roi, \
                    roi_polar_coords, N_roi_pts, \
                    xy, phi, Npnts)

    # print(time.time() - t0)



# =======================================================================
# ================== END OF CALCULATION FUNCTIONS =======================
# =======================================================================