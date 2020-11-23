import trimesh
import numpy as np

def scale_to_unit_sphere(mesh):
    if isinstance(mesh, trimesh.Scene):
        mesh = mesh.dump().sum()

    vertices = mesh.vertices - mesh.bounding_box.centroid
    distances = np.linalg.norm(vertices, axis=1)
    vertices /= np.max(distances)

    return trimesh.Trimesh(vertices=vertices, faces=mesh.faces)

def scale_to_unit_cube(mesh):
    if isinstance(mesh, trimesh.Scene):
        mesh = mesh.dump().sum()

    vertices = mesh.vertices - mesh.bounding_box.centroid
    vertices *= 2 / np.max(mesh.bounding_box.extents)

    return trimesh.Trimesh(vertices=vertices, faces=mesh.faces)

voxel_points = dict()

def get_raster_points(voxel_resolution):
    if voxel_resolution in voxel_points:
        return voxel_points[voxel_resolution]
    
    points = np.meshgrid(
        np.linspace(-1, 1, voxel_resolution),
        np.linspace(-1, 1, voxel_resolution),
        np.linspace(-1, 1, voxel_resolution)
    )
    points = np.stack(points)
    points = np.swapaxes(points, 1, 2)
    points = points.reshape(3, -1).transpose().astype(np.float32)

    voxel_points[voxel_resolution] = points
    return points

def check_voxels(voxels):
    block = voxels[:-1, :-1, :-1]
    d1 = (block - voxels[1:, :-1, :-1]).reshape(-1)
    d2 = (block - voxels[:-1, 1:, :-1]).reshape(-1)
    d3 = (block - voxels[:-1, :-1, 1:]).reshape(-1)

    max_distance = max(np.max(d1), np.max(d2), np.max(d3))
    return max_distance < 2.0 / voxels.shape[0] * 3**0.5 * 1.1

def sample_uniform_points_in_unit_sphere(amount, mesh_bounds):
    ellipsiod_dim = mesh_bounds*1.5 # bigger than the shape
    x,y,z = ellipsiod_dim
    
    sample = np.empty((0,3))
    while len(sample) < amount:
        unit_sphere_points = np.random.uniform(-ellipsiod_dim, 
                                            ellipsiod_dim, 
                                            size=(amount * 2 + 20, 3))	
                                # use ellipsoid bigger than the dimensions
        # check if sample points are inside the ellipsoid
        ellipsiod_check = unit_sphere_points[:,0]**2/x**2 + \
                        unit_sphere_points[:,1]**2/y**2 + \
                        unit_sphere_points[:,2]**2/z**2
        unit_sphere_points = unit_sphere_points[ellipsiod_check < 1]

        ## unit sphere, original
        # unit_sphere_points = np.random.uniform(-1, 1, size=(amount * 2 + 20, 3)) 
        # unit_sphere_points = unit_sphere_points[np.linalg.norm(unit_sphere_points, axis=1) < 1]

        sample = np.vstack((sample, unit_sphere_points))
    return sample[:amount]
