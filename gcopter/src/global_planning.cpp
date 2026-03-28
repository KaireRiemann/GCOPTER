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
#include <limits>
#include <random>

namespace
{
static constexpr bool kEnableVerboseSpanDiagnostics = false;

struct SimilarityStats
{
    double rms = 0.0;
    double max = 0.0;
};

struct SpanViolationReport
{
    int piece_idx = -1;
    int target_poly_idx = -1;
    double max_sample_violation = 0.0;
    double time_at_max_violation = 0.0;
    Eigen::Vector3d pos_at_max_violation = Eigen::Vector3d::Zero();
    std::vector<int> active_cp_indices;
    std::vector<double> target_poly_cp_violations;
    std::vector<int> best_poly_indices;
    std::vector<double> best_poly_violations;
    std::vector<int> common_feasible_polys;
    int best_common_poly_idx = -1;
    double best_common_poly_violation = std::numeric_limits<double>::infinity();
    std::vector<double> best_common_poly_cp_violations;
};

static inline double polytopeViolation(const Eigen::Vector3d &point,
                                       const Eigen::MatrixX4d &poly)
{
    return (poly.leftCols<3>() * point + poly.col(3)).maxCoeff();
}

static inline double unionCorridorViolation(const Eigen::Vector3d &point,
                                            const std::vector<Eigen::MatrixX4d> &h_polys)
{
    if (h_polys.empty())
    {
        return 0.0;
    }

    double best_poly_violation = std::numeric_limits<double>::infinity();
    for (const auto &poly : h_polys)
    {
        const Eigen::VectorXd violas = poly.leftCols<3>() * point + poly.col(3);
        const double poly_violation = violas.maxCoeff();
        best_poly_violation = std::min(best_poly_violation, poly_violation);
    }
    return best_poly_violation;
}

template <typename EvalFunc>
static inline double sampledTrajectoryCorridorViolation(const double duration,
                                                        EvalFunc eval,
                                                        const std::vector<Eigen::MatrixX4d> &h_polys,
                                                        int samples = 120)
{
    if (duration <= 0.0 || samples <= 0)
    {
        return 0.0;
    }

    double max_violation = -std::numeric_limits<double>::infinity();
    for (int i = 0; i < samples; ++i)
    {
        const double alpha = samples == 1 ? 0.0 : static_cast<double>(i) / static_cast<double>(samples - 1);
        const Eigen::Vector3d pos = eval(alpha * duration);
        max_violation = std::max(max_violation, unionCorridorViolation(pos, h_polys));
    }
    return max_violation;
}

template <typename EvalFuncA, typename EvalFuncB>
static inline SimilarityStats normalizedSimilarity(const double duration_a,
                                                   EvalFuncA eval_a,
                                                   const double duration_b,
                                                   EvalFuncB eval_b,
                                                   int samples = 120)
{
    SimilarityStats stats;
    if (duration_a <= 0.0 || duration_b <= 0.0 || samples <= 0)
    {
        return stats;
    }

    double sum_sq = 0.0;
    for (int i = 0; i < samples; ++i)
    {
        const double alpha = samples == 1 ? 0.0 : static_cast<double>(i) / static_cast<double>(samples - 1);
        const Eigen::Vector3d pa = eval_a(alpha * duration_a);
        const Eigen::Vector3d pb = eval_b(alpha * duration_b);
        const double dist = (pa - pb).norm();
        sum_sq += dist * dist;
        stats.max = std::max(stats.max, dist);
    }
    stats.rms = std::sqrt(sum_sq / static_cast<double>(samples));
    return stats;
}

static inline double nubsControlPointUnionViolation(const nubs::NUBSTrajectory<3> &traj,
                                                    const std::vector<Eigen::MatrixX4d> &h_polys)
{
    if (traj.getControlPoints().rows() == 0)
    {
        return 0.0;
    }

    double max_violation = -std::numeric_limits<double>::infinity();
    const auto &control_points = traj.getControlPoints();
    for (int i = 0; i < control_points.rows(); ++i)
    {
        max_violation = std::max(max_violation,
                                 unionCorridorViolation(control_points.row(i).transpose(), h_polys));
    }
    return max_violation;
}

static inline std::vector<int> collectPieceCandidatePolys(const int piece,
                                                          const int degree,
                                                          const int num_pieces,
                                                          const int num_polys)
{
    std::vector<int> candidates;
    auto add_unique = [&candidates](const int poly_idx)
    {
        if (poly_idx < 0)
        {
            return;
        }
        if (std::find(candidates.begin(), candidates.end(), poly_idx) == candidates.end())
        {
            candidates.push_back(poly_idx);
        }
    };

    add_unique(std::min(piece, num_polys - 1));
    for (int offset = 1; offset <= degree; ++offset)
    {
        if (piece - offset >= 0)
        {
            add_unique(std::min(piece - offset, num_polys - 1));
        }
        if (piece + offset < num_pieces)
        {
            add_unique(std::min(piece + offset, num_polys - 1));
        }
    }
    return candidates;
}

static inline double nubsSpanCommonPolyViolation(const nubs::NUBSTrajectory<3> &traj,
                                                 const std::vector<Eigen::MatrixX4d> &h_polys)
{
    if (traj.getPieceNum() <= 0 || traj.getControlPoints().rows() == 0 || h_polys.empty())
    {
        return 0.0;
    }

    const auto &control_points = traj.getControlPoints();
    const int p = traj.getP();
    double max_piece_violation = -std::numeric_limits<double>::infinity();

    for (int piece = 0; piece < traj.getPieceNum(); ++piece)
    {
        const int cp_begin = piece;
        const int cp_end = std::min(piece + p, static_cast<int>(control_points.rows()) - 1);
        double best_common_poly_violation = std::numeric_limits<double>::infinity();
        const std::vector<int> candidates =
            collectPieceCandidatePolys(piece, p, traj.getPieceNum(), static_cast<int>(h_polys.size()));

        for (int poly_idx : candidates)
        {
            double worst_cp_violation = -std::numeric_limits<double>::infinity();
            for (int cp_idx = cp_begin; cp_idx <= cp_end; ++cp_idx)
            {
                worst_cp_violation = std::max(worst_cp_violation,
                                              polytopeViolation(control_points.row(cp_idx).transpose(), h_polys[poly_idx]));
            }
            best_common_poly_violation = std::min(best_common_poly_violation, worst_cp_violation);
        }

        max_piece_violation = std::max(max_piece_violation, best_common_poly_violation);
    }

    return max_piece_violation;
}

static inline std::vector<SpanViolationReport> diagnoseViolatingSpans(const nubs::NUBSTrajectory<3> &traj,
                                                                      const std::vector<Eigen::MatrixX4d> &h_polys,
                                                                      double tol = 1.0e-6,
                                                                      int samples_per_piece = 8)
{
    std::vector<SpanViolationReport> reports;
    if (traj.getPieceNum() <= 0 || traj.getControlPoints().rows() == 0 || h_polys.empty())
    {
        return reports;
    }

    const auto &control_points = traj.getControlPoints();
    const auto &durations = traj.getDurations();
    const int p = traj.getP();
    double piece_start = 0.0;

    for (int piece = 0; piece < traj.getPieceNum(); ++piece)
    {
        const double piece_duration = durations(piece);
        const double piece_end = piece_start + piece_duration;
        double max_sample_violation = -std::numeric_limits<double>::infinity();
        double time_at_max_violation = piece_start;

        for (int s = 0; s <= samples_per_piece; ++s)
        {
            const double alpha = samples_per_piece == 0 ? 0.0 : static_cast<double>(s) / static_cast<double>(samples_per_piece);
            const double t = piece_start + alpha * piece_duration;
            const double violation = unionCorridorViolation(traj.evaluate(t, 0), h_polys);
            if (violation > max_sample_violation)
            {
                max_sample_violation = violation;
                time_at_max_violation = t;
            }
        }

        if (max_sample_violation > tol)
        {
            SpanViolationReport report;
            report.piece_idx = piece;
            report.target_poly_idx = std::min(piece, static_cast<int>(h_polys.size()) - 1);
            report.max_sample_violation = max_sample_violation;
            report.time_at_max_violation = time_at_max_violation;
            report.pos_at_max_violation = traj.evaluate(time_at_max_violation, 0);
            const std::vector<int> candidates =
                collectPieceCandidatePolys(piece, p, traj.getPieceNum(), static_cast<int>(h_polys.size()));

            const int cp_begin = piece;
            const int cp_end = std::min(piece + p, static_cast<int>(control_points.rows()) - 1);
            for (int cp_idx = cp_begin; cp_idx <= cp_end; ++cp_idx)
            {
                const Eigen::Vector3d cp = control_points.row(cp_idx).transpose();
                report.active_cp_indices.push_back(cp_idx);
                report.target_poly_cp_violations.push_back(
                    polytopeViolation(cp, h_polys[report.target_poly_idx]));

                int best_poly_idx = -1;
                double best_poly_violation = std::numeric_limits<double>::infinity();
                for (int poly_idx = 0; poly_idx < static_cast<int>(h_polys.size()); ++poly_idx)
                {
                    const double violation = polytopeViolation(cp, h_polys[poly_idx]);
                    if (violation < best_poly_violation)
                    {
                        best_poly_violation = violation;
                        best_poly_idx = poly_idx;
                    }
                }
                report.best_poly_indices.push_back(best_poly_idx);
                report.best_poly_violations.push_back(best_poly_violation);
            }

            for (int poly_idx : candidates)
            {
                bool all_inside = true;
                double worst_cp_violation = -std::numeric_limits<double>::infinity();
                std::vector<double> per_cp_violations;
                for (int cp_idx = cp_begin; cp_idx <= cp_end; ++cp_idx)
                {
                    const double cp_violation =
                        polytopeViolation(control_points.row(cp_idx).transpose(), h_polys[poly_idx]);
                    per_cp_violations.push_back(cp_violation);
                    worst_cp_violation = std::max(worst_cp_violation, cp_violation);
                    if (cp_violation > tol)
                    {
                        all_inside = false;
                    }
                }
                if (worst_cp_violation < report.best_common_poly_violation)
                {
                    report.best_common_poly_violation = worst_cp_violation;
                    report.best_common_poly_idx = poly_idx;
                    report.best_common_poly_cp_violations = per_cp_violations;
                }
                if (all_inside)
                {
                    report.common_feasible_polys.push_back(poly_idx);
                }
            }

            reports.push_back(report);
        }

        piece_start = piece_end;
    }
    return reports;
}

static inline void printViolatingSpanReports(const std::string &label,
                                             const nubs::NUBSTrajectory<3> &traj,
                                             const std::vector<Eigen::MatrixX4d> &h_polys,
                                             int max_reports = 4)
{
    const std::vector<SpanViolationReport> reports = diagnoseViolatingSpans(traj, h_polys);
    if (reports.empty())
    {
        return;
    }

    std::cout << label << " violating spans = " << reports.size() << std::endl;
    for (int i = 0; i < std::min(max_reports, static_cast<int>(reports.size())); ++i)
    {
        const auto &report = reports[i];
        std::cout << "  piece=" << report.piece_idx
                  << " target_poly=" << report.target_poly_idx
                  << " max_sample_violation=" << report.max_sample_violation
                  << " t=" << report.time_at_max_violation
                  << " sample_pos=[" << report.pos_at_max_violation.transpose() << "]"
                  << " best_common_poly=" << report.best_common_poly_idx
                  << ":" << report.best_common_poly_violation
                  << " common_feasible_polys=";
        if (report.common_feasible_polys.empty())
        {
            std::cout << "{}";
        }
        else
        {
            std::cout << "{";
            for (size_t k = 0; k < report.common_feasible_polys.size(); ++k)
            {
                if (k > 0) std::cout << ",";
                std::cout << report.common_feasible_polys[k];
            }
            std::cout << "}";
        }
        std::cout << std::endl;

        std::cout << "    active_cp_indices=";
        for (size_t k = 0; k < report.active_cp_indices.size(); ++k)
        {
            std::cout << (k == 0 ? "[" : ",") << report.active_cp_indices[k];
        }
        std::cout << "]" << std::endl;

        std::cout << "    target_poly_cp_violations=";
        for (size_t k = 0; k < report.target_poly_cp_violations.size(); ++k)
        {
            std::cout << (k == 0 ? "[" : ",") << report.target_poly_cp_violations[k];
        }
        std::cout << "]" << std::endl;

        std::cout << "    best_poly_assignment=";
        for (size_t k = 0; k < report.best_poly_indices.size(); ++k)
        {
            std::cout << (k == 0 ? "[" : ",")
                      << report.best_poly_indices[k] << ":" << report.best_poly_violations[k];
        }
        std::cout << "]" << std::endl;

        std::cout << "    best_common_poly_cp_violations=";
        for (size_t k = 0; k < report.best_common_poly_cp_violations.size(); ++k)
        {
            std::cout << (k == 0 ? "[" : ",") << report.best_common_poly_cp_violations[k];
        }
        std::cout << "]" << std::endl;
    }
}
}

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
    nubs::NUBSTrajectory<3> nubs_traj_zo_dpasa;
    nubs::NUBSTrajectory<3> nubs_traj_zo_abc;
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
                gcopter::NUBSSFCOptimizerZO nubs_opt_zo_dpasa(gcopter::NUBSSFCOptimizerZO::SolverBackend::DPASA);
                gcopter::NUBSSFCOptimizerZO nubs_opt_zo_abc(gcopter::NUBSSFCOptimizerZO::SolverBackend::ABC);
                

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
                                          magnitudeBounds,
                                          penaltyWeights))
                {
                    ROS_WARN("NUBS Optimizer Setup Failed!");
                    return;
                }

                if (std::isinf(nubs_opt.optimize(nubs_traj)))
                {
                    ROS_WARN("NUBS Optimization Diverged/Failed!");
                    return;
                }
                auto t4 = std::chrono::high_resolution_clock::now();
                double t_b = std::chrono::duration_cast<std::chrono::milliseconds>(t4-t3).count();
                std::cout<<"nubs trajectory optimize time : "<<t_b<<" ms"<<std::endl;


                auto t7 = std::chrono::high_resolution_clock::now();
                if(!nubs_opt_zo_dpasa.setup(config.weightT,iniState,finState,hPolys,INFINITY,
                                      magnitudeBounds,penaltyWeights))
                {
                    ROS_WARN("NUBS ZERO-ORDER DPASA Setup Failed!");
                    return;
                }


                if (std::isinf(nubs_opt_zo_dpasa.optimize(nubs_traj_zo_dpasa)))
                {
                    ROS_WARN("NUBS ZERO-ORDER DPASA Optimization Diverged/Failed!");
                    return;
                }
                auto t8 = std::chrono::high_resolution_clock::now();
                double t_o_dpasa = std::chrono::duration_cast<std::chrono::milliseconds>(t8-t7).count();
                std::cout<<"zero-order nubs DPASA optimize time : "<<t_o_dpasa<<" ms"<<std::endl;

                auto t9 = std::chrono::high_resolution_clock::now();
                if(!nubs_opt_zo_abc.setup(config.weightT,iniState,finState,hPolys,INFINITY,
                                      magnitudeBounds,penaltyWeights))
                {
                    ROS_WARN("NUBS ZERO-ORDER ABC Setup Failed!");
                    return;
                }

                if (std::isinf(nubs_opt_zo_abc.optimize(nubs_traj_zo_abc)))
                {
                    ROS_WARN("NUBS ZERO-ORDER ABC Optimization Diverged/Failed!");
                    return;
                }
                auto t10 = std::chrono::high_resolution_clock::now();
                double t_o_abc = std::chrono::duration_cast<std::chrono::milliseconds>(t10-t9).count();
                std::cout<<"zero-order nubs ABC optimize time : "<<t_o_abc<<" ms"<<std::endl;

                trajStamp = ros::Time::now().toSec();
                if (traj.getPieceNum() > 0)
                {
                    visualizer.visualize(traj, route);
                }

                if (spline_traj.isInitialized() && spline_traj.getNumSegments() > 0)
                {
                    
                    visualizer.visualize(spline_traj, route);
                }

                if(nubs_traj_zo_dpasa.getPieceNum() > 0)
                {
                    visualizer.visualize(nubs_traj_zo_dpasa, route, "zo_dpasa", 1.0, 0.0, 1.0);
                }

                if(nubs_traj_zo_abc.getPieceNum() > 0)
                {
                    visualizer.visualize(nubs_traj_zo_abc, route, "zo_abc", 1.0, 0.6, 0.0);
                }

                if(nubs_traj.getPieceNum() > 0)
                {
                    visualizer.visualize(nubs_traj, route, "lbfgs", 0.0, 1.0, 0.0);
                }

                
            }


        double time_minco = traj.getTotalDuration();
        double time_spline = spline_traj.getDuration();
        double time_nubs_gd = nubs_traj.getTotalDuration();
        double time_nubs_zo_dpasa = nubs_traj_zo_dpasa.getTotalDuration();
        double time_nubs_zo_abc = nubs_traj_zo_abc.getTotalDuration();

        const auto minco_eval = [this](double t) { return traj.getPos(t); };
        const auto spline_eval = [this](double t) { return spline_traj.getTrajectory().evaluate(t, SplineTrajectory::Deriv::Pos); };
        const auto nubs_eval = [this](double t) { return nubs_traj.evaluate(t, 0); };
        const auto zo_dpasa_eval = [this](double t) { return nubs_traj_zo_dpasa.evaluate(t, 0); };
        const auto zo_abc_eval = [this](double t) { return nubs_traj_zo_abc.evaluate(t, 0); };

        const SimilarityStats spline_sim = normalizedSimilarity(time_minco, minco_eval, time_spline, spline_eval);
        const SimilarityStats nubs_sim = normalizedSimilarity(time_minco, minco_eval, time_nubs_gd, nubs_eval);
        const SimilarityStats zo_dpasa_sim = normalizedSimilarity(time_minco, minco_eval, time_nubs_zo_dpasa, zo_dpasa_eval);
        const SimilarityStats zo_abc_sim = normalizedSimilarity(time_minco, minco_eval, time_nubs_zo_abc, zo_abc_eval);

        const double minco_sample_violation = sampledTrajectoryCorridorViolation(time_minco, minco_eval, hPolys);
        const double spline_sample_violation = sampledTrajectoryCorridorViolation(time_spline, spline_eval, hPolys);
        const double nubs_sample_violation = sampledTrajectoryCorridorViolation(time_nubs_gd, nubs_eval, hPolys);
        const double zo_dpasa_sample_violation = sampledTrajectoryCorridorViolation(time_nubs_zo_dpasa, zo_dpasa_eval, hPolys);
        const double zo_abc_sample_violation = sampledTrajectoryCorridorViolation(time_nubs_zo_abc, zo_abc_eval, hPolys);

        const double nubs_cp_violation = nubsControlPointUnionViolation(nubs_traj, hPolys);
        const double zo_dpasa_cp_violation = nubsControlPointUnionViolation(nubs_traj_zo_dpasa, hPolys);
        const double zo_abc_cp_violation = nubsControlPointUnionViolation(nubs_traj_zo_abc, hPolys);
        const double nubs_span_common_violation = nubsSpanCommonPolyViolation(nubs_traj, hPolys);
        const double zo_dpasa_span_common_violation = nubsSpanCommonPolyViolation(nubs_traj_zo_dpasa, hPolys);
        const double zo_abc_span_common_violation = nubsSpanCommonPolyViolation(nubs_traj_zo_abc, hPolys);

        std::cout<< " Trajectory Time : "<<std::endl;
        std::cout<< " MINCO Time : "<<time_minco<<" s"<<std::endl;
        std::cout<< " Spline Time : "<<time_spline<<" s"<<std::endl;
        std::cout<< " NUBS Time : "<<time_nubs_gd<<" s"<<std::endl;
        std::cout<< " NUBS ZO DPASA Time : "<<time_nubs_zo_dpasa<<" s"<<std::endl;
        std::cout<< " NUBS ZO ABC Time : "<<time_nubs_zo_abc<<" s"<<std::endl;

        std::cout<< " Similarity To MINCO (RMS / MAX) : "<<std::endl;
        std::cout<< " Spline : "<<spline_sim.rms<<" / "<<spline_sim.max<<std::endl;
        std::cout<< " NUBS : "<<nubs_sim.rms<<" / "<<nubs_sim.max<<std::endl;
        std::cout<< " NUBS ZO DPASA : "<<zo_dpasa_sim.rms<<" / "<<zo_dpasa_sim.max<<std::endl;
        std::cout<< " NUBS ZO ABC : "<<zo_abc_sim.rms<<" / "<<zo_abc_sim.max<<std::endl;

        std::cout<< " Corridor Violation (sample / cp-union / span-common-poly) : "<<std::endl;
        std::cout<< " MINCO : "<<minco_sample_violation<<" / n/a / n/a"<<std::endl;
        std::cout<< " Spline : "<<spline_sample_violation<<" / n/a / n/a"<<std::endl;
        std::cout<< " NUBS : "<<nubs_sample_violation<<" / "<<nubs_cp_violation<<" / "<<nubs_span_common_violation<<std::endl;
        std::cout<< " NUBS ZO DPASA : "<<zo_dpasa_sample_violation<<" / "<<zo_dpasa_cp_violation<<" / "<<zo_dpasa_span_common_violation<<std::endl;
        std::cout<< " NUBS ZO ABC : "<<zo_abc_sample_violation<<" / "<<zo_abc_cp_violation<<" / "<<zo_abc_span_common_violation<<std::endl;

        if (kEnableVerboseSpanDiagnostics && nubs_sample_violation > 1.0e-6)
        {
            printViolatingSpanReports("[Span Diagnostic] NUBS", nubs_traj, hPolys);
        }
        if (kEnableVerboseSpanDiagnostics && zo_dpasa_sample_violation > 1.0e-6)
        {
            printViolatingSpanReports("[Span Diagnostic] NUBS ZO DPASA", nubs_traj_zo_dpasa, hPolys);
        }
        if (kEnableVerboseSpanDiagnostics && zo_abc_sample_violation > 1.0e-6)
        {
            printViolatingSpanReports("[Span Diagnostic] NUBS ZO ABC", nubs_traj_zo_abc, hPolys);
        }
            
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
        if (nubs_traj_zo_dpasa.getKnots().size() > 0 && delta < nubs_traj_zo_dpasa.getTotalDuration()) {
            Eigen::Vector3d pos = nubs_traj_zo_dpasa.evaluate(delta, 0);
            visualizer.visualizeSphere(pos, config.dilateRadius, "nubs_zo_dpasa_sphere", 1.0, 0.0, 1.0);
        }

        if (nubs_traj_zo_abc.getKnots().size() > 0 && delta < nubs_traj_zo_abc.getTotalDuration()) {
            Eigen::Vector3d pos = nubs_traj_zo_abc.evaluate(delta, 0);
            visualizer.visualizeSphere(pos, config.dilateRadius, "nubs_zo_abc_sphere", 1.0, 0.6, 0.0);
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
