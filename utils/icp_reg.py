
import numpy as np
import open3d as o3d

class ICP:
    # Class to perform ICP registration between two point clouds.
    def __init__(self, events_points, model_points,   transform = np.eye(4)):
        # print("ICP")
        self.events_points = events_points
        self.model_points = model_points
        self.transform = transform
          
    def Allign_Candidate_Points(self):
        # print("Before_aligned_Model Points:", self.model_points)   
        
        points_to_be_aligned_3d = np.hstack((np.array(self.model_points), np.zeros((len(self.model_points), 1))))
        reference_points_3d = np.hstack((np.array(self.events_points), np.zeros((len(self.events_points), 1))))   
        source_cloud = o3d.geometry.PointCloud()
        target_cloud = o3d.geometry.PointCloud()
        source_cloud.points = o3d.utility.Vector3dVector(points_to_be_aligned_3d)
        target_cloud.points = o3d.utility.Vector3dVector(reference_points_3d)
           
        return source_cloud, target_cloud, points_to_be_aligned_3d
    
    def transform_patch(self):
        
        source_cloud, target_cloud, points_to_be_aligned_3d = self.Allign_Candidate_Points()
        initial_guess = self.transform                    
        icp_result = o3d.pipelines.registration.registration_icp(
            source_cloud, target_cloud, max_correspondence_distance=4,
            estimation_method=o3d.pipelines.registration.TransformationEstimationPointToPoint(),
            criteria=o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=2000)) 
        
        # aligned_points = np.asarray(icp_result.transformation @ 
        #                             np.concatenate((points_to_be_aligned_3d, np.zeros((len(points_to_be_aligned_3d), 1))), axis=1).T).T[:, :2]
        # self.model_points = aligned_points[:, :2]
 
        evaluation = o3d.pipelines.registration.evaluate_registration(source_cloud, target_cloud,  0.3, np.eye(4))
        source_point = source_cloud.transform(icp_result.transformation)

        source_point = np.asarray(source_point.points)
        
        return icp_result.transformation
