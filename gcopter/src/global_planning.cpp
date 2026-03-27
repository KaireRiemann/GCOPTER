#include "misc/visualizer.hpp"
#include "gcopter/spline_sfc_optimizer.hpp"
#include "gcopter/firi.hpp"
#include "gcopter/flatness.hpp"
#include "gcopter/voxel_map.hpp"
#include "gcopter/sfc_gen.hpp"
#include "gcopter/gcopter.hpp"
#include "SplineTrajectory/SplineTrajectory.hpp"
#include "NUBSTrajectory/NUBSTrajectory.hpp"
#include "gcopter/nubs_sfc_optimizer.hpp"
#include "gcopter/nubs_sfc_optimizer_zo.hpp"

#include <ros/ros.h>
#include <ros/console.h>
#include <geometry_msgs/Point.h>
#include <geometry_msgs/PoseStamped.h>
#include <sensor_msgs/PointCloud2.h>

#include <cmath>
#include <iostream>
#include <string>
#include <vector>
#include <memory>
#include <chrono>
#include <random>

struct Config
{
    std::string mapTopic;
    std::string targetTopic;
    double dilateRadius;
    double voxelWidth;
    std::vector<double> mapBound;
    double timeoutRRT;
    double maxVelMag;
    double maxBdrMag;
    double maxTiltAngle;
    double minThrust;
    double maxThrust;
    double vehicleMass;
    double gravAcc;
    double horizDrag;
    double vertDrag;
    double parasDrag;
    double speedEps;
    double weightT;
    std::vector<double> chiVec;
    double smoothingEps;
    int integralIntervs;
    double relCostTol;

    Config(const ros::NodeHandle &nh_priv)
    {
        nh_priv.getParam("MapTopic", mapTopic);
        nh_priv.getParam("TargetTopic", targetTopic);
        nh_priv.getParam("DilateRadius", dilateRadius);
        nh_priv.getParam("VoxelWidth", voxelWidth);
        nh_priv.getParam("MapBound", mapBound);
        nh_priv.getParam("TimeoutRRT", timeoutRRT);
        nh_priv.getParam("MaxVelMag", maxVelMag);
        nh_priv.getParam("MaxBdrMag", maxBdrMag);
        nh_priv.getParam("MaxTiltAngle", maxTiltAngle);
        nh_priv.getParam("MinThrust", minThrust);
        nh_priv.getParam("MaxThrust", maxThrust);
        nh_priv.getParam("VehicleMass", vehicleMass);
        nh_priv.getParam("GravAcc", gravAcc);
        nh_priv.getParam("HorizDrag", horizDrag);
        nh_priv.getParam("VertDrag", vertDrag);
        nh_priv.getParam("ParasDrag", parasDrag);
        nh_priv.getParam("SpeedEps", speedEps);
        nh_priv.getParam("WeightT", weightT);
        nh_priv.getParam("ChiVec", chiVec);
        nh_priv.getParam("SmoothingEps", smoothingEps);
        nh_priv.getParam("IntegralIntervs", integralIntervs);
        nh_priv.getParam("RelCostTol", relCostTol);
    }
};

class GlobalPlanner
{
private:
    Config config;

    ros::NodeHandle nh;
    ros::Subscriber mapSub;
    ros::Subscriber targetSub;

    bool mapInitialized;
    voxel_map::VoxelMap voxelMap;
    Visualizer visualizer;
    std::vector<Eigen::Vector3d> startGoal;


    SplineTrajectory::QuinticSpline3D spline_traj;
    nubs::NUBSTrajectory<3> nubs_traj;
    nubs::NUBSTrajectory<3> nubs_traj_zo;
    Trajectory<5> traj;
    double trajStamp;

public:
    GlobalPlanner(const Config &conf,
                  ros::NodeHandle &nh_)
        : config(conf),
          nh(nh_),
          mapInitialized(false),
          visualizer(nh)
    {
        const Eigen::Vector3i xyz((config.mapBound[1] - config.mapBound[0]) / config.voxelWidth,
                                  (config.mapBound[3] - config.mapBound[2]) / config.voxelWidth,
                                  (config.mapBound[5] - config.mapBound[4]) / config.voxelWidth);

        const Eigen::Vector3d offset(config.mapBound[0], config.mapBound[2], config.mapBound[4]);

        voxelMap = voxel_map::VoxelMap(xyz, offset, config.voxelWidth);

        mapSub = nh.subscribe(config.mapTopic, 1, &GlobalPlanner::mapCallBack, this,
                              ros::TransportHints().tcpNoDelay());

        targetSub = nh.subscribe(config.targetTopic, 1, &GlobalPlanner::targetCallBack, this,
                                 ros::TransportHints().tcpNoDelay());
    }

    inline void mapCallBack(const sensor_msgs::PointCloud2::ConstPtr &msg)
    {
        if (!mapInitialized)
        {
            size_t cur = 0;
            const size_t total = msg->data.size() / msg->point_step;
            float *fdata = (float *)(&msg->data[0]);
            for (size_t i = 0; i < total; i++)
            {
                cur = msg->point_step / sizeof(float) * i;

                if (std::isnan(fdata[cur + 0]) || std::isinf(fdata[cur + 0]) ||
                    std::isnan(fdata[cur + 1]) || std::isinf(fdata[cur + 1]) ||
                    std::isnan(fdata[cur + 2]) || std::isinf(fdata[cur + 2]))
                {
                    continue;
                }
                voxelMap.setOccupied(Eigen::Vector3d(fdata[cur + 0],
                                                     fdata[cur + 1],
                                                     fdata[cur + 2]));
            }

            voxelMap.dilate(std::ceil(config.dilateRadius / voxelMap.getScale()));

            mapInitialized = true;
        }
    }

    inline void plan()
    {
        if (startGoal.size() == 2)
        {
            std::vector<Eigen::Vector3d> route;
            sfc_gen::planPath<voxel_map::VoxelMap>(startGoal[0],
                                                   startGoal[1],
                                                   voxelMap.getOrigin(),
                                                   voxelMap.getCorner(),
                                                   &voxelMap, 0.01,
                                                   route);
            std::vector<Eigen::MatrixX4d> hPolys;
            std::vector<Eigen::Vector3d> pc;
            voxelMap.getSurf(pc);

            sfc_gen::convexCover(route,
                                 pc,
                                 voxelMap.getOrigin(),
                                 voxelMap.getCorner(),
                                 7.0,
                                 3.0,
                                 hPolys);
            sfc_gen::shortCut(hPolys);
            
            
            if (route.size() > 1)
            {
                visualizer.visualizePolytope(hPolys);

                Eigen::Matrix3d iniState;
                Eigen::Matrix3d finState;
                iniState << route.front(), Eigen::Vector3d::Zero(), Eigen::Vector3d::Zero();
                finState << route.back(), Eigen::Vector3d::Zero(), Eigen::Vector3d::Zero();

                gcopter::GCOPTER_PolytopeSFC minco_opt;
                gcopter::SplineSFCOptimizer spline_opt;
                gcopter::NUBSSFCOptimizer nubs_opt;
                gcopter::NUBSSFCOptimizerZO nubs_opt_zo;
                

                // magnitudeBounds = [v_max, omg_max, theta_max, thrust_min, thrust_max]^T
                // penaltyWeights = [pos_weight, vel_weight, omg_weight, theta_weight, thrust_weight]^T
                // physicalParams = [vehicle_mass, gravitational_acceleration, horitonral_drag_coeff,
                //                   vertical_drag_coeff, parasitic_drag_coeff, speed_smooth_factor]^T
                // initialize some constraint parameters
                Eigen::VectorXd magnitudeBounds(5);
                Eigen::VectorXd penaltyWeights(5);
                Eigen::VectorXd physicalParams(6);
                magnitudeBounds(0) = config.maxVelMag;
                magnitudeBounds(1) = config.maxBdrMag;
                magnitudeBounds(2) = config.maxTiltAngle;
                magnitudeBounds(3) = config.minThrust;
                magnitudeBounds(4) = config.maxThrust;
                penaltyWeights(0) = (config.chiVec)[0];
                penaltyWeights(1) = (config.chiVec)[1];
                penaltyWeights(2) = (config.chiVec)[2];
                penaltyWeights(3) = (config.chiVec)[3];
                penaltyWeights(4) = (config.chiVec)[4];
                physicalParams(0) = config.vehicleMass;
                physicalParams(1) = config.gravAcc;
                physicalParams(2) = config.horizDrag;
                physicalParams(3) = config.vertDrag;
                physicalParams(4) = config.parasDrag;
                physicalParams(5) = config.speedEps;
                const int quadratureRes = config.integralIntervs;

                spline_traj = SplineTrajectory::QuinticSpline3D();
                traj.clear();

                auto t_m1 = std::chrono::high_resolution_clock::now();
                if (!minco_opt.setup(config.weightT,
                                   iniState, finState,
                                   hPolys, INFINITY,
                                   config.smoothingEps,
                                   quadratureRes,
                                   magnitudeBounds,
                                   penaltyWeights,
                                   physicalParams))
                {
                    return;
                }

                if (std::isinf(minco_opt.optimize(traj, config.relCostTol)))
                {
                    return;
                }
                auto t_m2 = std::chrono::high_resolution_clock::now();
                double t_m = std::chrono::duration_cast<std::chrono::milliseconds>(t_m2-t_m1).count();
                std::cout<<"minco trajectory optimize time : "<<t_m<<" ms"<<std::endl;


                auto t1 = std::chrono::high_resolution_clock::now();
                if (!spline_opt.setup(config.weightT,
                                   iniState, finState,
                                   hPolys, INFINITY,
                                   config.smoothingEps,
                                   quadratureRes,
                                   magnitudeBounds,
                                   penaltyWeights,
                                   physicalParams))
                {
                    return;
                }

                if (std::isinf(spline_opt.optimize(spline_traj, config.relCostTol)))
                {
                    return;
                }

                auto t2 = std::chrono::high_resolution_clock::now();
                double t = std::chrono::duration_cast<std::chrono::milliseconds>(t2-t1).count();
                std::cout<<"spline trajectory optimize time : "<<t<<" ms"<<std::endl;


                auto t3 = std::chrono::high_resolution_clock::now();
                if (!nubs_opt.setup(config.weightT,
                                          iniState, finState,
                                          hPolys, INFINITY,
                                          config.smoothingEps,
                                          magnitudeBounds,
                                          penaltyWeights,
                                          physicalParams))
                {
                    ROS_WARN("NUBS Optimizer Setup Failed!");
                    return;
                }

                if (std::isinf(nubs_opt.optimize(nubs_traj, config.relCostTol)))
                {
                    ROS_WARN("NUBS Optimization Diverged/Failed!");
                    return;
                }
                auto t4 = std::chrono::high_resolution_clock::now();
                double t_b = std::chrono::duration_cast<std::chrono::milliseconds>(t4-t3).count();
                std::cout<<"nubs trajectory optimize time : "<<t_b<<" ms"<<std::endl;


                auto t7 = std::chrono::high_resolution_clock::now();
                if(!nubs_opt_zo.setup(config.weightT,iniState,finState,hPolys,INFINITY,
                                      magnitudeBounds,penaltyWeights))
                {
                    ROS_WARN("NUBS ZERO-ORDER Optimizer Setup Failed!");
                    return;
                }


                if (std::isinf(nubs_opt_zo.optimize(nubs_traj_zo)))
                {
                    ROS_WARN("NUBS ZERO-ORDER Optimization Diverged/Failed!");
                    return;
                }
                auto t8 = std::chrono::high_resolution_clock::now();
                double t_o = std::chrono::duration_cast<std::chrono::milliseconds>(t8-t7).count();
                std::cout<<"zeor-order nubs trajectory optimize time : "<<t_o<<" ms"<<std::endl;

                trajStamp = ros::Time::now().toSec();
                if (traj.getPieceNum() > 0)
                {
                    visualizer.visualize(traj, route);
                }

                if (spline_traj.isInitialized() && spline_traj.getNumSegments() > 0)
                {
                    
                    visualizer.visualize(spline_traj, route);
                }

                if(nubs_traj_zo.getPieceNum() > 0)
                {
                    visualizer.visualize(nubs_traj_zo, route, "zo", 1.0, 0.0, 1.0);
                }

                if(nubs_traj.getPieceNum() > 0)
                {
                    visualizer.visualize(nubs_traj, route, "lbfgs", 0.0, 1.0, 0.0);
                }

                
            }


        double time_minco = traj.getTotalDuration();
        double time_spline = spline_traj.getDuration();
        double time_nubs_gd = nubs_traj.getTotalDuration();
        double time_nubs_zo = nubs_traj_zo.getTotalDuration();

        std::cout<< " Trajectory Time : "<<std::endl;
        std::cout<< " MINCO Time : "<<time_minco<<" s"<<std::endl;
        std::cout<< " Spline Time : "<<time_spline<<" s"<<std::endl;
        std::cout<< " NUBS Time : "<<time_nubs_gd<<" s"<<std::endl;
        std::cout<< " NUBS ZO Time : "<<time_nubs_zo<<" s"<<std::endl;
            
        }
    }

    inline void targetCallBack(const geometry_msgs::PoseStamped::ConstPtr &msg)
    {
        if (mapInitialized)
        {
            if (startGoal.size() >= 2)
            {
                startGoal.clear();
            }
            const double zGoal = config.mapBound[4] + config.dilateRadius +
                                 fabs(msg->pose.orientation.z) *
                                     (config.mapBound[5] - config.mapBound[4] - 2 * config.dilateRadius);
            const Eigen::Vector3d goal(msg->pose.position.x, msg->pose.position.y, zGoal);
            if (voxelMap.query(goal) == 0)
            {
                visualizer.visualizeStartGoal(goal, 0.5, startGoal.size());
                startGoal.emplace_back(goal);
            }
            else
            {
                ROS_WARN("Infeasible Position Selected !!!\n");
            }

            plan();
        }
        return;
    }

    inline void process()
    {
        Eigen::VectorXd physicalParams(6);
        physicalParams << config.vehicleMass, config.gravAcc, config.horizDrag, 
                          config.vertDrag, config.parasDrag, config.speedEps;
        flatness::FlatnessMap flatmap;
        flatmap.reset(physicalParams(0), physicalParams(1), physicalParams(2),
                      physicalParams(3), physicalParams(4), physicalParams(5));

        const double delta = ros::Time::now().toSec() - trajStamp;
        if (delta <= 0.0) return;

        if (spline_traj.isInitialized() && spline_traj.getNumSegments() > 0 && delta < spline_traj.getDuration())
        {
            double thr; Eigen::Vector4d quat; Eigen::Vector3d omg;
            const auto &ppoly = spline_traj.getTrajectory();
            const Eigen::Vector3d vel = ppoly.evaluate(delta, SplineTrajectory::Deriv::Vel);
            const Eigen::Vector3d acc = ppoly.evaluate(delta, SplineTrajectory::Deriv::Acc);
            const Eigen::Vector3d jer = ppoly.evaluate(delta, SplineTrajectory::Deriv::Jerk);
            const Eigen::Vector3d pos = ppoly.evaluate(delta, SplineTrajectory::Deriv::Pos);

            flatmap.forward(vel, acc, jer, 0.0, 0.0, thr, quat, omg);
            
            std_msgs::Float64 speedMsg, thrMsg, tiltMsg, bdrMsg;
            speedMsg.data = vel.norm();
            thrMsg.data = thr;
            tiltMsg.data = acos(1.0 - 2.0 * (quat(1)*quat(1) + quat(2)*quat(2)));
            bdrMsg.data = omg.norm();
            visualizer.speedPub.publish(speedMsg);
            visualizer.thrPub.publish(thrMsg);
            visualizer.tiltPub.publish(tiltMsg);
            visualizer.bdrPub.publish(bdrMsg);

            visualizer.visualizeSphere(pos, config.dilateRadius, "spline_sphere", 0.0, 0.5, 1.0);
        }

        // 2. 处理 MINCO 轨迹运动小球 (深蓝色)
        if (traj.getPieceNum() > 0 && delta < traj.getTotalDuration()) {
            Eigen::Vector3d pos = traj.getPos(delta);
            visualizer.visualizeSphere(pos, config.dilateRadius, "minco_sphere", 0.0, 0.0, 1.0);
        }

        // 3. 处理 NUBS 一阶轨迹运动小球 (绿色)
        if (nubs_traj.getKnots().size() > 0 && delta < nubs_traj.getTotalDuration()) {
            Eigen::Vector3d pos = nubs_traj.evaluate(delta, 0);
            visualizer.visualizeSphere(pos, config.dilateRadius, "nubs_lbfgs_sphere", 0.0, 1.0, 0.0);
        }

        // 4. 处理 NUBS 零阶轨迹运动小球 (紫色)
        if (nubs_traj_zo.getKnots().size() > 0 && delta < nubs_traj_zo.getTotalDuration()) {
            Eigen::Vector3d pos = nubs_traj_zo.evaluate(delta, 0);
            visualizer.visualizeSphere(pos, config.dilateRadius, "nubs_zo_sphere", 1.0, 0.0, 1.0);
        }
    }
};

int main(int argc, char **argv)
{
    ros::init(argc, argv, "global_planning_node");
    ros::NodeHandle nh_;

    GlobalPlanner global_planner(Config(ros::NodeHandle("~")), nh_);

    ros::Rate lr(1000);
    while (ros::ok())
    {
        global_planner.process();
        ros::spinOnce();
        lr.sleep();
    }

    return 0;
}
