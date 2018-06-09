#include <Eigen/StdVector>
#include <Eigen/Dense>
#include <iostream>
#include <stdint.h>
#include <vector>
#include <math.h>
#include <unordered_set>

#include "g2o/core/sparse_optimizer.h"
#include "g2o/core/block_solver.h"
#include "g2o/core/solver.h"
#include "g2o/core/robust_kernel_impl.h"
#include "g2o/core/optimization_algorithm_levenberg.h"
#include "g2o/solvers/cholmod/linear_solver_cholmod.h"
#include "g2o/solvers/dense/linear_solver_dense.h"
#include "g2o/types/sba/types_six_dof_expmap.h"
#include "g2o/types/slam3d/edge_xyz_prior.h"
#include "g2o/types/slam3d/vertex_se3.h"
#include "g2o/types/slam3d/vertex_pointxyz.h"
#include "g2o/types/slam3d/edge_se3_pointxyz.h"
#include "edge_se3exp_pointxyz_prior.h"
#include "g2o/solvers/structure_only/structure_only_solver.h"


using namespace Eigen;
using namespace std;
using namespace g2o;
//using namespace cv;
static double uniform_rand(double lowerBndr, double upperBndr){
  return lowerBndr + ((double) std::rand() / (RAND_MAX + 1.0)) * (upperBndr - lowerBndr);
}

static double gauss_rand(double mean, double sigma){
  double x, y, r2;
  do {
    x = -1.0 + 2.0 * uniform_rand(0.0, 1.0);
    y = -1.0 + 2.0 * uniform_rand(0.0, 1.0);
    r2 = x * x + y * y;
  } while (r2 > 1.0 || r2 == 0.0);
  return mean + sigma * y * std::sqrt(-2.0 * log(r2) / r2);
}

int uniform(int from, int to){
  return static_cast<int>(uniform_rand(from, to));
}

double uniform(){
  return uniform_rand(0., 1.);
}

double gaussian(double sigma){
  return gauss_rand(0., sigma);
}

struct CameraPose{
    Matrix3d R;
    Vector3d t;
};

vector< CameraPose > generate_camera_poses()
{
    vector< CameraPose > poses{};
    for(int i=0;i<15;i++)
    {
        poses.push_back(CameraPose{MatrixXd::Identity(3,3),Vector3d{i*0.04-1.,0,0}});
    }
    return poses;
}

vector< Vector3d > generate_points_cloud()
{
    double unit = 3.141592653/180;
    double curr_angle = 0.0;
    vector< Vector3d > points{};
    for(int i=0;i<500;i++)
    {
        //points.push_back(Vector3d{5*cos(curr_angle),5*sin(curr_angle),i*0.5});
        points.push_back(Vector3d((uniform()-0.5)*3,
                                  uniform()-0.5,
                                  uniform()+3));
        //curr_angle += unit;
    }
    return points;
}


void test_projection()
{
    double focal_length= 1000.;
    Vector2d principal_point(320., 240.);

    SE3Quat pose{MatrixXd::Identity(3,3),Vector3d{0.0,0.0,0.0}};
    Vector3d point{1,0,10};

    CameraParameters * cam_params
        = new g2o::CameraParameters (focal_length, principal_point, 0.);

    Vector2d x = cam_params->cam_map(point);

    point[0] = 0;
    point[1] = 1;
    x = cam_params->cam_map(pose.map(point));

    point[0] = 0;
    point[1] = 0;
    x = cam_params->cam_map(pose.map(point));        

}


double compute_error(g2o::SparseOptimizer& optimizer, map<int,Vector3d>& estimation_gt)
{
    double error = 0;
    for(auto esti_gt:estimation_gt)
    {
        error += (((g2o::VertexSBAPointXYZ*)optimizer.vertex(esti_gt.first))->estimate()-esti_gt.second).norm();
    }
    return error/estimation_gt.size();
}

double compute_pose_error(vector<g2o::VertexSE3Expmap*> measure,vector<CameraPose> ground_truth)
{
    double error = 0;
    for(int i=0;i<measure.size();i++)
    {
        error += (((g2o::SE3Quat)(measure[i]->estimate())).translation() - ground_truth[i].t).norm();
    }
    return error/measure.size();
}

int main()
{
    //test_projection();
    auto poses = generate_camera_poses();
    auto points = generate_points_cloud();
    bool DENSE = false;

    g2o::SparseOptimizer optimizer;
    optimizer.setVerbose(false);
    std::unique_ptr<g2o::BlockSolver_6_3::LinearSolverType> linearSolver;
    if (DENSE) {
      linearSolver = g2o::make_unique<g2o::LinearSolverDense<g2o::BlockSolver_6_3::PoseMatrixType>>();
    } else {
      linearSolver = g2o::make_unique<g2o::LinearSolverCholmod<g2o::BlockSolver_6_3::PoseMatrixType>>();
    }

    g2o::OptimizationAlgorithmLevenberg* solver = new g2o::OptimizationAlgorithmLevenberg(
      g2o::make_unique<g2o::BlockSolver_6_3>(std::move(linearSolver))
    );
    optimizer.setAlgorithm(solver);

    double focal_length= 1000.;
    Vector2d principal_point(320., 240.);

    g2o::CameraParameters * cam_params
        = new g2o::CameraParameters (focal_length, principal_point, 0.);
    cam_params->setId(0);

    if (!optimizer.addParameter(cam_params)) {
      assert(false);
    }

    vector<g2o::VertexSE3Expmap*> se3_vertices;
    int vertex_id = 0;
    for(auto pose:poses)
    {//Add the pose vertices
        g2o::SE3Quat curr_pose(pose.R,pose.t);
        g2o::VertexSE3Expmap* vertex = new g2o::VertexSE3Expmap();
        vertex->setId(vertex_id);
        if(vertex_id<2)
            vertex->setFixed(true);
        else
            vertex->setFixed(false);

        curr_pose.setTranslation(curr_pose.translation()+Vector3d(uniform(),
                                                                  uniform(),
                                                                  uniform()));
        vertex->setEstimate(curr_pose);                      

        se3_vertices.push_back(vertex);
        optimizer.addVertex(vertex);
        vertex_id++;

        g2o::EdgeSE3ExpXYZPointPrior* gps_constrains = new g2o::EdgeSE3ExpXYZPointPrior();
        gps_constrains->setVertex(0, dynamic_cast<g2o::OptimizableGraph::Vertex*>(vertex));
        gps_constrains->setMeasurement(pose.t);
        //gps_constrains->information() = 10*Matrix3d::Identity();
        gps_constrains->setInformation(10000*Matrix3d::Identity());
        optimizer.addEdge(gps_constrains);
    }

    map<g2o::VertexSE3Expmap*,Vector2d> vertex_connections;
    map<int,Vector3d> estimation_gt;
    for(auto point:points)
    {
        g2o::VertexSBAPointXYZ* X_Vertex = new g2o::VertexSBAPointXYZ();
        X_Vertex->setId(vertex_id);
        X_Vertex->setMarginalized(true);
        X_Vertex->setEstimate(point + Vector3d(gauss_rand(0., 1),
                                          gauss_rand(0., 1),
                                          gauss_rand(0., 1)));

        vertex_connections.clear();
        for(auto pose_vertex:se3_vertices)
        {
            Vector2d image_pixel = cam_params->cam_map(pose_vertex->estimate().map(point));
            if (image_pixel[0]>=0 && image_pixel[1]>=0 && image_pixel[0]<640 && image_pixel[1]<480)
            {
                vertex_connections[pose_vertex]=image_pixel;
            }
        }

        if(vertex_connections.size()>=2)
        {//If the 3D point is visible to more than 2 poses
            optimizer.addVertex(X_Vertex);
            estimation_gt[vertex_id] = point;
            vertex_id++;
            for(auto proj:vertex_connections)
            {
                g2o::EdgeProjectXYZ2UV* edge = new g2o::EdgeProjectXYZ2UV();
                edge->setVertex(0, dynamic_cast<g2o::OptimizableGraph::Vertex*>(X_Vertex));
                edge->setVertex(1, dynamic_cast<g2o::OptimizableGraph::Vertex*>(proj.first));
                edge->setMeasurement(proj.second+
                                     Vector2d(gauss_rand(0.0,1),
                                              gauss_rand(0.0,1)));
                edge->information() = Matrix2d::Identity();
                edge->setRobustKernel(new g2o::RobustKernelHuber());
                edge->setParameterId(0,0);
                optimizer.addEdge(edge);
            }
        }
        else
        {
            delete X_Vertex;
        }
    }

    double before = compute_error(optimizer,estimation_gt);
    double pose_before = compute_pose_error(se3_vertices,poses);


    optimizer.initializeOptimization();
    optimizer.optimize(10);

    double after = compute_error(optimizer,estimation_gt);
    double pose_after = compute_pose_error(se3_vertices,poses);
    cout<<before<<","<<after<<endl;
    cout<<pose_before<<","<<pose_after<<endl;

    return 0;
}

