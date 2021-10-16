import open3d as o3d
import time
import numpy as np


vis = o3d.visualization.Visualizer()
vis.create_window(window_name="Lidar", width=1920, height=1080)
opt = vis.get_render_option()
opt.point_size = 1

raw_pc_for_plot = o3d.geometry.PointCloud()
pointcloud_init = np.array(([50, 50, 50], [-50, -50, -50]))
raw_pc_for_plot.points = o3d.utility.Vector3dVector(pointcloud_init[:, :3])
coordi = o3d.geometry.TriangleMesh.create_coordinate_frame(size=5.0, origin=([0, 0, 0]))
vis.add_geometry(raw_pc_for_plot)

# Log File Open
# Struct Size(xyz) : 1843873
# Before Sensor Struct : 673
xyz_list = []
intensity_list = []
file = open(log_path, "rb")
count = 1;
while True:
    end = time.time()
    # SyncTime Read
    byte = file.read(8)

    if not byte:
        break
    SyncTime = struct.unpack('d', byte)
    # Header Read
    byte = file.read(665)

    # Lidar Data Read
    byte = file.read(1843200)
    xyz, intensity = get_pointcloud(byte)

    xyz_list.append(xyz)
    intensity_list.append(intensity_list)
    print("Index : " + str(count) + ", Data Load --> SyncTime : " + str(SyncTime) + " Loading Time : " + str(time.time() - end))
    count = count + 1

    raw_pc_for_plot.points = o3d.utility.Vector3dVector(np.array(xyz))
    vis.update_geometry(raw_pc_for_plot)
    vis.poll_events()
    vis.update_renderer()
