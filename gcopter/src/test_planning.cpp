#include "misc/visualizer.hpp"
#include "gcopter/firi.hpp"
#include "gcopter/flatness.hpp"
#include "gcopter/voxel_map.hpp"
#include "gcopter/sfc_gen.hpp"
#include "gcopter/gcopter.hpp"
#include "gcopter/nubs_sfc_optimizer.hpp"
#include "gcopter/nubs_sfc_optimizer_zo.hpp"
#include "gcopter/lbfgs.hpp"

#include <ros/ros.h>
#include <geometry_msgs/PoseStamped.h>
#include <sensor_msgs/PointCloud2.h>

#include <Eigen/Eigen>

#include <chrono>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <sstream>
#include <string>
#include <vector>

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

    explicit Config(const ros::NodeHandle &nh_priv)
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

namespace
{
    using NUBSTraj = nubs::NUBSTrajectory<3, 7>;

    struct GradientCheckSummary
    {
        double error_norm = 0.0;
        double rel_error = 0.0;
        double max_abs_error = 0.0;
    };

    struct ScaleScanSample
    {
        double scale = 1.0;
        gcopter::NUBSSFCOptimizer::DecisionDiagnostic diag;
    };

    struct FullObjectiveContext
    {
        gcopter::NUBSSFCOptimizer *optimizer = nullptr;
        std::string label;
        int eval_count = 0;
    };

    struct TimeOnlyContext
    {
        gcopter::NUBSSFCOptimizer *optimizer = nullptr;
        Eigen::VectorXd fixed_spatial;
        int time_dim = 0;
        std::string label;
        int eval_count = 0;
    };

    static inline lbfgs::lbfgs_parameter_t makeLbfgsParams()
    {
        lbfgs::lbfgs_parameter_t params;
        params.mem_size = 16;
        params.past = 3;
        params.g_epsilon = 1.0e-5;
        params.min_step = 1.0e-32;
        params.delta = 1.0e-4;
        params.max_iterations = 100;
        return params;
    }

    static inline void printDiagnostic(const std::string &label,
                                       const gcopter::NUBSSFCOptimizer::DecisionDiagnostic &diag)
    {
        std::cout << label
                  << " cost_finite=" << diag.cost_finite
                  << " grad_finite=" << diag.grad_finite
                  << " grad_norm=" << diag.grad_norm
                  << " min_T=" << diag.min_segment_time
                  << " total_T=" << diag.total_duration
                  << " knots_finite=" << diag.knots_finite
                  << " ctrl_finite=" << diag.control_points_finite
                  << " total=" << diag.total_cost
                  << " spatial=" << diag.spatial_penalty
                  << " energy=" << diag.energy_cost
                  << " time_cost=" << diag.time_cost
                  << " cp=" << diag.cp_cost
                  << std::endl;
    }

    static inline double safeEvaluate(gcopter::NUBSSFCOptimizer &optimizer,
                                      const Eigen::VectorXd &x,
                                      Eigen::VectorXd &grad,
                                      const std::string &label,
                                      int eval_count)
    {
        double cost = optimizer.evaluateDecision(x, grad);
        if (!std::isfinite(cost) || !grad.allFinite())
        {
            const auto diag = optimizer.diagnoseDecision(x);
            std::cout << "[safeEvaluate failure] " << label
                      << " eval=" << eval_count
                      << " cost_is_finite=" << std::isfinite(cost)
                      << " grad_all_finite=" << grad.allFinite()
                      << std::endl;
            printDiagnostic("  diagnostic:", diag);
            grad.setZero();
            return 1.0e20;
        }
        return cost;
    }

    static inline double fullObjectiveCallback(void *ptr,
                                               const Eigen::VectorXd &x,
                                               Eigen::VectorXd &grad)
    {
        auto &ctx = *static_cast<FullObjectiveContext *>(ptr);
        ++ctx.eval_count;
        return safeEvaluate(*ctx.optimizer, x, grad, ctx.label, ctx.eval_count);
    }

    static inline double timeOnlyObjectiveCallback(void *ptr,
                                                   const Eigen::VectorXd &x_time,
                                                   Eigen::VectorXd &grad_time)
    {
        auto &ctx = *static_cast<TimeOnlyContext *>(ptr);
        Eigen::VectorXd x_full(ctx.time_dim + ctx.fixed_spatial.size());
        x_full.head(ctx.time_dim) = x_time;
        x_full.tail(ctx.fixed_spatial.size()) = ctx.fixed_spatial;

        Eigen::VectorXd grad_full(x_full.size());
        ++ctx.eval_count;
        const double cost = safeEvaluate(*ctx.optimizer, x_full, grad_full, ctx.label, ctx.eval_count);
        grad_time = grad_full.head(ctx.time_dim);
        return cost;
    }

    static inline GradientCheckSummary checkTimeGradient(gcopter::NUBSSFCOptimizer &optimizer,
                                                         const Eigen::VectorXd &x,
                                                         int time_dim,
                                                         double eps = 1.0e-6)
    {
        GradientCheckSummary summary;
        Eigen::VectorXd analytical = Eigen::VectorXd::Zero(x.size());
        optimizer.evaluateDecision(x, analytical);

        Eigen::VectorXd numerical = Eigen::VectorXd::Zero(time_dim);
        Eigen::VectorXd dummy_grad = Eigen::VectorXd::Zero(x.size());
        Eigen::VectorXd x_temp = x;

        for (int i = 0; i < time_dim; ++i)
        {
            const double old_value = x_temp(i);
            x_temp(i) = old_value + eps;
            const double cost_p = optimizer.evaluateDecision(x_temp, dummy_grad);

            x_temp(i) = old_value - eps;
            const double cost_m = optimizer.evaluateDecision(x_temp, dummy_grad);

            x_temp(i) = old_value;
            numerical(i) = (cost_p - cost_m) / (2.0 * eps);
        }

        const Eigen::VectorXd diff = analytical.head(time_dim) - numerical;
        summary.error_norm = diff.norm();
        summary.max_abs_error = diff.cwiseAbs().maxCoeff();
        const double analytical_norm = analytical.head(time_dim).norm();
        summary.rel_error = analytical_norm > 1.0e-9 ? summary.error_norm / analytical_norm : summary.error_norm;
        return summary;
    }

    static inline bool runFullOptimization(gcopter::NUBSSFCOptimizer &optimizer,
                                           const Eigen::VectorXd &x_init,
                                           Eigen::VectorXd &x_best,
                                           double &cost_out,
                                           double &ms_out,
                                           int &ret_out,
                                           const std::string &label)
    {
        x_best = x_init;
        FullObjectiveContext ctx;
        ctx.optimizer = &optimizer;
        ctx.label = label;

        auto t_start = std::chrono::high_resolution_clock::now();
        ret_out = lbfgs::lbfgs_optimize(x_best, cost_out,
                                        &fullObjectiveCallback,
                                        nullptr,
                                        nullptr,
                                        &ctx,
                                        makeLbfgsParams());
        auto t_end = std::chrono::high_resolution_clock::now();
        ms_out = std::chrono::duration_cast<std::chrono::milliseconds>(t_end - t_start).count();

        if (ret_out < 0)
        {
            std::cout << label << " LBFGS ret=" << ret_out
                      << " (" << lbfgs::lbfgs_strerror(ret_out) << ")" << std::endl;
        }
        return ret_out >= 0 && std::isfinite(cost_out);
    }

    static inline bool runTimeOnlyOptimization(gcopter::NUBSSFCOptimizer &optimizer,
                                               const Eigen::VectorXd &x_init,
                                               int time_dim,
                                               Eigen::VectorXd &x_best,
                                               double &cost_out,
                                               double &ms_out,
                                               int &ret_out,
                                               const std::string &label)
    {
        Eigen::VectorXd x_time = x_init.head(time_dim);

        TimeOnlyContext ctx;
        ctx.optimizer = &optimizer;
        ctx.fixed_spatial = x_init.tail(x_init.size() - time_dim);
        ctx.time_dim = time_dim;
        ctx.label = label;

        auto t_start = std::chrono::high_resolution_clock::now();
        ret_out = lbfgs::lbfgs_optimize(x_time, cost_out,
                                        &timeOnlyObjectiveCallback,
                                        nullptr,
                                        nullptr,
                                        &ctx,
                                        makeLbfgsParams());
        auto t_end = std::chrono::high_resolution_clock::now();
        ms_out = std::chrono::duration_cast<std::chrono::milliseconds>(t_end - t_start).count();

        x_best.resize(x_init.size());
        x_best.head(time_dim) = x_time;
        x_best.tail(ctx.fixed_spatial.size()) = ctx.fixed_spatial;

        if (ret_out < 0)
        {
            std::cout << label << " LBFGS ret=" << ret_out
                      << " (" << lbfgs::lbfgs_strerror(ret_out) << ")" << std::endl;
        }
        return ret_out >= 0 && std::isfinite(cost_out);
    }

    static inline void printDurationLine(const std::string &name,
                                         double optimize_ms,
                                         double total_time,
                                         double cost)
    {
        std::cout << std::setw(24) << std::left << name
                  << " optimize = " << std::setw(8) << optimize_ms << " ms"
                  << " duration = " << std::setw(10) << total_time << " s"
                  << " cost = " << cost << std::endl;
    }

    static inline std::string makeScaleScanCsvPath()
    {
        const auto now = std::chrono::system_clock::now().time_since_epoch();
        const auto stamp = std::chrono::duration_cast<std::chrono::milliseconds>(now).count();
        std::ostringstream oss;
        oss << "/tmp/gcopter_scale_scan_" << stamp << ".csv";
        return oss.str();
    }

    static inline bool writeScaleScanCsv(const std::string &path,
                                         const std::string &label,
                                         const std::vector<ScaleScanSample> &samples)
    {
        std::ofstream ofs(path);
        if (!ofs.is_open())
        {
            return false;
        }

        ofs << "label,scale,total_cost,spatial_penalty,energy_cost,time_cost,cp_cost,grad_norm,"
               "min_segment_time,total_duration,cost_finite,grad_finite,knots_finite,control_points_finite\n";
        for (const auto &sample : samples)
        {
            const auto &d = sample.diag;
            ofs << label << ","
                << sample.scale << ","
                << d.total_cost << ","
                << d.spatial_penalty << ","
                << d.energy_cost << ","
                << d.time_cost << ","
                << d.cp_cost << ","
                << d.grad_norm << ","
                << d.min_segment_time << ","
                << d.total_duration << ","
                << d.cost_finite << ","
                << d.grad_finite << ","
                << d.knots_finite << ","
                << d.control_points_finite << "\n";
        }
        return true;
    }

    static inline std::vector<ScaleScanSample> runScaleScan(gcopter::NUBSSFCOptimizer &optimizer,
                                                            const Eigen::VectorXd &spatial_vars,
                                                            const std::vector<double> &base_times,
                                                            double scale_min,
                                                            double scale_max,
                                                            int num_samples)
    {
        std::vector<ScaleScanSample> samples;
        if (base_times.empty() || num_samples <= 1)
        {
            return samples;
        }

        samples.reserve(num_samples);
        const double denom = static_cast<double>(num_samples - 1);
        for (int i = 0; i < num_samples; ++i)
        {
            const double alpha = static_cast<double>(i) / denom;
            const double scale = scale_min + (scale_max - scale_min) * alpha;
            std::vector<double> scaled_times = base_times;
            for (double &t : scaled_times)
            {
                t *= scale;
            }

            ScaleScanSample sample;
            sample.scale = scale;
            const Eigen::VectorXd x = optimizer.composeDecision(scaled_times, spatial_vars);
            sample.diag = optimizer.diagnoseDecision(x);
            samples.push_back(sample);
        }
        return samples;
    }

    static inline void printBestScaleSample(const std::string &label,
                                            const std::vector<ScaleScanSample> &samples)
    {
        const ScaleScanSample *best_sample = nullptr;
        double best_cost = std::numeric_limits<double>::infinity();
        for (const auto &sample : samples)
        {
            if (sample.diag.cost_finite && sample.diag.total_cost < best_cost)
            {
                best_cost = sample.diag.total_cost;
                best_sample = &sample;
            }
        }
        if (best_sample == nullptr)
        {
            std::cout << label << " no finite sample found in scale scan." << std::endl;
            return;
        }

        std::cout << label
                  << " best_scale=" << best_sample->scale
                  << " total_cost=" << best_sample->diag.total_cost
                  << " total_T=" << best_sample->diag.total_duration
                  << " time_cost=" << best_sample->diag.time_cost
                  << " cp_cost=" << best_sample->diag.cp_cost
                  << " energy=" << best_sample->diag.energy_cost
                  << std::endl;
    }

    static inline const ScaleScanSample *findBestScaleSample(const std::vector<ScaleScanSample> &samples)
    {
        const ScaleScanSample *best_sample = nullptr;
        double best_cost = std::numeric_limits<double>::infinity();
        for (const auto &sample : samples)
        {
            if (sample.diag.cost_finite && sample.diag.total_cost < best_cost)
            {
                best_cost = sample.diag.total_cost;
                best_sample = &sample;
            }
        }
        return best_sample;
    }
}

class TestPlanner
{
private:
    Config config_;
    ros::NodeHandle nh_;
    ros::Subscriber map_sub_;
    ros::Subscriber target_sub_;

    bool map_initialized_ = false;
    voxel_map::VoxelMap voxel_map_;
    Visualizer visualizer_;
    std::vector<Eigen::Vector3d> start_goal_;

    Trajectory<5> minco_traj_;
    NUBSTraj nubs_spatial_only_;
    NUBSTraj nubs_zo_;
    NUBSTraj nubs_time_only_;
    NUBSTraj nubs_full_init_;
    NUBSTraj nubs_full_warm_;

public:
    TestPlanner(const Config &config, ros::NodeHandle &nh)
        : config_(config),
          nh_(nh),
          visualizer_(nh)
    {
        const Eigen::Vector3i xyz((config_.mapBound[1] - config_.mapBound[0]) / config_.voxelWidth,
                                  (config_.mapBound[3] - config_.mapBound[2]) / config_.voxelWidth,
                                  (config_.mapBound[5] - config_.mapBound[4]) / config_.voxelWidth);
        const Eigen::Vector3d offset(config_.mapBound[0], config_.mapBound[2], config_.mapBound[4]);
        voxel_map_ = voxel_map::VoxelMap(xyz, offset, config_.voxelWidth);

        map_sub_ = nh_.subscribe(config_.mapTopic, 1, &TestPlanner::mapCallback, this,
                                 ros::TransportHints().tcpNoDelay());
        target_sub_ = nh_.subscribe(config_.targetTopic, 1, &TestPlanner::targetCallback, this,
                                    ros::TransportHints().tcpNoDelay());
    }

    void mapCallback(const sensor_msgs::PointCloud2::ConstPtr &msg)
    {
        if (map_initialized_)
        {
            return;
        }

        const size_t total = msg->data.size() / msg->point_step;
        const float *fdata = reinterpret_cast<const float *>(&msg->data[0]);
        for (size_t i = 0; i < total; ++i)
        {
            const size_t cur = msg->point_step / sizeof(float) * i;
            if (std::isnan(fdata[cur + 0]) || std::isinf(fdata[cur + 0]) ||
                std::isnan(fdata[cur + 1]) || std::isinf(fdata[cur + 1]) ||
                std::isnan(fdata[cur + 2]) || std::isinf(fdata[cur + 2]))
            {
                continue;
            }
            voxel_map_.setOccupied(Eigen::Vector3d(fdata[cur + 0], fdata[cur + 1], fdata[cur + 2]));
        }

        voxel_map_.dilate(std::ceil(config_.dilateRadius / voxel_map_.getScale()));
        map_initialized_ = true;
    }

    void targetCallback(const geometry_msgs::PoseStamped::ConstPtr &msg)
    {
        if (!map_initialized_)
        {
            return;
        }

        if (start_goal_.size() >= 2)
        {
            start_goal_.clear();
        }

        const double z_goal = config_.mapBound[4] + config_.dilateRadius +
                              std::fabs(msg->pose.orientation.z) *
                                  (config_.mapBound[5] - config_.mapBound[4] - 2.0 * config_.dilateRadius);
        const Eigen::Vector3d goal(msg->pose.position.x, msg->pose.position.y, z_goal);
        if (voxel_map_.query(goal) == 0)
        {
            visualizer_.visualizeStartGoal(goal, 0.5, start_goal_.size());
            start_goal_.emplace_back(goal);
        }
        else
        {
            ROS_WARN("Infeasible Position Selected !!!");
        }

        plan();
    }

    void plan()
    {
        if (start_goal_.size() != 2)
        {
            return;
        }

        std::vector<Eigen::Vector3d> route;
        sfc_gen::planPath<voxel_map::VoxelMap>(start_goal_[0],
                                               start_goal_[1],
                                               voxel_map_.getOrigin(),
                                               voxel_map_.getCorner(),
                                               &voxel_map_,
                                               0.01,
                                               route);

        std::vector<Eigen::MatrixX4d> h_polys;
        std::vector<Eigen::Vector3d> pc;
        voxel_map_.getSurf(pc);
        sfc_gen::convexCover(route,
                             pc,
                             voxel_map_.getOrigin(),
                             voxel_map_.getCorner(),
                             7.0,
                             3.0,
                             h_polys);
        sfc_gen::shortCut(h_polys);

        if (route.size() <= 1)
        {
            return;
        }

        visualizer_.visualizePolytope(h_polys);

        Eigen::Matrix3d ini_state;
        Eigen::Matrix3d fin_state;
        ini_state << route.front(), Eigen::Vector3d::Zero(), Eigen::Vector3d::Zero();
        fin_state << route.back(), Eigen::Vector3d::Zero(), Eigen::Vector3d::Zero();

        Eigen::VectorXd magnitude_bounds(5);
        Eigen::VectorXd penalty_weights(5);
        Eigen::VectorXd physical_params(6);
        magnitude_bounds << config_.maxVelMag, config_.maxBdrMag, config_.maxTiltAngle,
            config_.minThrust, config_.maxThrust;
        penalty_weights << config_.chiVec[0], config_.chiVec[1], config_.chiVec[2],
            config_.chiVec[3], config_.chiVec[4];
        physical_params << config_.vehicleMass, config_.gravAcc, config_.horizDrag,
            config_.vertDrag, config_.parasDrag, config_.speedEps;

        gcopter::GCOPTER_PolytopeSFC minco_opt;
        gcopter::NUBSSFCOptimizer nubs_opt;
        gcopter::NUBSSFCOptimizerZO nubs_opt_zo;

        minco_traj_.clear();
        nubs_spatial_only_ = NUBSTraj();
        nubs_zo_ = NUBSTraj();
        nubs_time_only_ = NUBSTraj();
        nubs_full_init_ = NUBSTraj();
        nubs_full_warm_ = NUBSTraj();

        auto t_minco_start = std::chrono::high_resolution_clock::now();
        if (!minco_opt.setup(config_.weightT,
                             ini_state, fin_state,
                             h_polys, INFINITY,
                             config_.smoothingEps,
                             config_.integralIntervs,
                             magnitude_bounds,
                             penalty_weights,
                             physical_params))
        {
            ROS_WARN("MINCO setup failed.");
            return;
        }
        if (std::isinf(minco_opt.optimize(minco_traj_, config_.relCostTol)))
        {
            ROS_WARN("MINCO optimize failed.");
            return;
        }
        auto t_minco_end = std::chrono::high_resolution_clock::now();
        const double minco_ms =
            std::chrono::duration_cast<std::chrono::milliseconds>(t_minco_end - t_minco_start).count();

        if (!nubs_opt.setup(config_.weightT,
                            ini_state, fin_state,
                            h_polys, INFINITY,
                            magnitude_bounds,
                            penalty_weights))
        {
            ROS_WARN("Gradient NUBS setup failed.");
            return;
        }

        auto t_spatial_start = std::chrono::high_resolution_clock::now();
        if (std::isinf(nubs_opt.optimize(nubs_spatial_only_)))
        {
            ROS_WARN("Spatial-only gradient NUBS optimize failed.");
            return;
        }
        auto t_spatial_end = std::chrono::high_resolution_clock::now();
        const double spatial_ms =
            std::chrono::duration_cast<std::chrono::milliseconds>(t_spatial_end - t_spatial_start).count();

        if (!nubs_opt_zo.setup(config_.weightT,
                               ini_state, fin_state,
                               h_polys, INFINITY,
                               magnitude_bounds,
                               penalty_weights))
        {
            ROS_WARN("ZO NUBS setup failed.");
            return;
        }

        auto t_zo_start = std::chrono::high_resolution_clock::now();
        if (std::isinf(nubs_opt_zo.optimize(nubs_zo_)))
        {
            ROS_WARN("ZO NUBS optimize failed.");
            return;
        }
        auto t_zo_end = std::chrono::high_resolution_clock::now();
        const double zo_ms =
            std::chrono::duration_cast<std::chrono::milliseconds>(t_zo_end - t_zo_start).count();

        const Eigen::VectorXd x_init = nubs_opt.getFullInitialGuess();
        const int time_dim = nubs_opt.getTimeVariableDim();
        NUBSTraj nubs_init_traj;
        nubs_opt.decisionToTrajectory(x_init, nubs_init_traj);
        const auto init_diag = nubs_opt.diagnoseDecision(x_init);

        double init_cost = 0.0;
        {
            Eigen::VectorXd grad_init = Eigen::VectorXd::Zero(x_init.size());
            init_cost = nubs_opt.evaluateDecision(x_init, grad_init);
        }

        Eigen::VectorXd x_time_best;
        double time_only_cost = std::numeric_limits<double>::infinity();
        double time_only_ms = 0.0;
        int time_only_ret = 0;
        const bool time_only_ok = runTimeOnlyOptimization(nubs_opt, x_init, time_dim,
                                                          x_time_best, time_only_cost, time_only_ms,
                                                          time_only_ret, "time_only");
        if (time_only_ok)
        {
            nubs_opt.decisionToTrajectory(x_time_best, nubs_time_only_);
        }

        const GradientCheckSummary time_grad_check_init = checkTimeGradient(nubs_opt, x_init, time_dim);

        Eigen::VectorXd x_full_best;
        double full_init_cost = std::numeric_limits<double>::infinity();
        double full_init_ms = 0.0;
        int full_init_ret = 0;
        const bool full_init_ok = runFullOptimization(nubs_opt, x_init,
                                                      x_full_best, full_init_cost, full_init_ms,
                                                      full_init_ret, "full_init");
        if (full_init_ok)
        {
            nubs_opt.decisionToTrajectory(x_full_best, nubs_full_init_);
        }

        Eigen::VectorXd x_warm;
        double full_warm_cost = std::numeric_limits<double>::infinity();
        double full_warm_ms = 0.0;
        int full_warm_ret = 0;
        bool warm_ok = false;
        bool warm_constructed = false;
        GradientCheckSummary time_grad_check_warm;
        gcopter::NUBSSFCOptimizer::DecisionDiagnostic warm_diag;
        gcopter::NUBSSFCOptimizer::DecisionDiagnostic warm_roundtrip_diag;
        NUBSTraj nubs_warm_roundtrip;
        NUBSTraj nubs_scale_best_;
        std::vector<ScaleScanSample> scale_scan_samples;
        std::string scale_scan_csv_path;
        double warm_ctrl_diff_norm = std::numeric_limits<double>::quiet_NaN();
        double warm_ctrl_diff_max = std::numeric_limits<double>::quiet_NaN();
        if (nubs_opt.trajectoryToDecision(nubs_zo_, x_warm))
        {
            warm_constructed = true;
            warm_diag = nubs_opt.diagnoseDecision(x_warm);
            time_grad_check_warm = checkTimeGradient(nubs_opt, x_warm, time_dim);
            nubs_opt.decisionToTrajectory(x_warm, nubs_warm_roundtrip);
            warm_roundtrip_diag = nubs_opt.diagnoseDecision(x_warm);
            if (nubs_warm_roundtrip.getControlPoints().rows() == nubs_zo_.getControlPoints().rows())
            {
                const Eigen::MatrixXd ctrl_diff = nubs_warm_roundtrip.getControlPoints() - nubs_zo_.getControlPoints();
                warm_ctrl_diff_norm = ctrl_diff.norm();
                warm_ctrl_diff_max = ctrl_diff.cwiseAbs().maxCoeff();
            }

            const Eigen::VectorXd warm_spatial = x_warm.tail(x_warm.size() - time_dim);
            const Eigen::VectorXd &warm_durations_vec = nubs_warm_roundtrip.getDurations();
            const std::vector<double> warm_times(warm_durations_vec.data(),
                                                 warm_durations_vec.data() + warm_durations_vec.size());
            scale_scan_samples = runScaleScan(nubs_opt, warm_spatial, warm_times, 0.4, 3.0, 80);
            scale_scan_csv_path = makeScaleScanCsvPath();
            if (writeScaleScanCsv(scale_scan_csv_path, "x_warm_scale_scan", scale_scan_samples))
            {
                std::cout << "Scale scan CSV          saved to " << scale_scan_csv_path << std::endl;
            }
            else
            {
                std::cout << "Scale scan CSV          failed to save: " << scale_scan_csv_path << std::endl;
            }

            Eigen::VectorXd x_warm_seed = x_warm;
            if (const ScaleScanSample *best_scale_sample = findBestScaleSample(scale_scan_samples))
            {
                std::vector<double> scaled_times = warm_times;
                for (double &t : scaled_times)
                {
                    t *= best_scale_sample->scale;
                }
                x_warm_seed = nubs_opt.composeDecision(scaled_times, warm_spatial);
                nubs_opt.decisionToTrajectory(x_warm_seed, nubs_scale_best_);
            }

            Eigen::VectorXd x_warm_best;
            warm_ok = runFullOptimization(nubs_opt, x_warm_seed,
                                          x_warm_best, full_warm_cost, full_warm_ms,
                                          full_warm_ret, "full_warm");
            if (warm_ok)
            {
                x_warm = x_warm_best;
                nubs_opt.decisionToTrajectory(x_warm, nubs_full_warm_);
            }
        }

        std::cout << "\n========== Test Planning Report ==========\n";
        printDurationLine("MINCO baseline", minco_ms,
                          minco_traj_.getTotalDuration(), 0.0);
        printDurationLine("NUBS spatial-only", spatial_ms,
                          nubs_spatial_only_.getTotalDuration(), 0.0);
        printDurationLine("NUBS zero-order", zo_ms,
                          nubs_zo_.getTotalDuration(), 0.0);
        printDurationLine("NUBS init guess", 0.0,
                          nubs_init_traj.getTotalDuration(),
                          init_cost);
        printDiagnostic("Init diagnostic:", init_diag);

        if (time_only_ok)
        {
            printDurationLine("Time-only LBFGS", time_only_ms,
                              nubs_time_only_.getTotalDuration(), time_only_cost);
        }
        else
        {
            std::cout << "Time-only LBFGS          failed\n";
        }

        if (full_init_ok)
        {
            printDurationLine("Full LBFGS from init", full_init_ms,
                              nubs_full_init_.getTotalDuration(), full_init_cost);
        }
        else
        {
            std::cout << "Full LBFGS from init     failed\n";
        }

        if (warm_ok)
        {
            printDurationLine("Full LBFGS from ZO", full_warm_ms,
                              nubs_full_warm_.getTotalDuration(), full_warm_cost);
        }
        else
        {
            std::cout << "Full LBFGS from ZO       failed\n";
        }

        std::cout << "Time gradient @ x_init   error_norm = " << time_grad_check_init.error_norm
                  << ", rel_error = " << time_grad_check_init.rel_error
                  << ", max_abs = " << time_grad_check_init.max_abs_error << std::endl;

        if (warm_constructed)
        {
            printDiagnostic("x_warm diagnostic:", warm_diag);
            std::cout << "Time gradient @ x_warm  error_norm = " << time_grad_check_warm.error_norm
                      << ", rel_error = " << time_grad_check_warm.rel_error
                      << ", max_abs = " << time_grad_check_warm.max_abs_error << std::endl;
            std::cout << "Warm round-trip         zo_duration = " << nubs_zo_.getTotalDuration()
                      << " s, reconstructed = " << nubs_warm_roundtrip.getTotalDuration()
                      << " s, ctrl_diff_norm = " << warm_ctrl_diff_norm
                      << ", ctrl_diff_max = " << warm_ctrl_diff_max << std::endl;
            printDiagnostic("Round-trip diagnostic:", warm_roundtrip_diag);
            printBestScaleSample("Scale scan summary:", scale_scan_samples);
            if (nubs_scale_best_.getPieceNum() > 0)
            {
                printDurationLine("Scale-best seed", 0.0,
                                  nubs_scale_best_.getTotalDuration(),
                                  findBestScaleSample(scale_scan_samples)->diag.total_cost);
            }
        }

        if (warm_ok)
        {
            std::cout << "Warm start duration      before = " << nubs_zo_.getTotalDuration()
                      << " s, after = " << nubs_full_warm_.getTotalDuration() << " s" << std::endl;
        }
        std::cout << "=========================================\n" << std::endl;

        visualizer_.visualize(minco_traj_, route);
        visualizer_.visualize(nubs_spatial_only_, route, "spatial_only", 0.0, 1.0, 0.0);
        visualizer_.visualize(nubs_zo_, route, "zo", 1.0, 0.0, 1.0);
        if (time_only_ok)
        {
            visualizer_.visualize(nubs_time_only_, route, "time_only", 1.0, 0.75, 0.0);
        }
        if (full_init_ok)
        {
            visualizer_.visualize(nubs_full_init_, route, "full_init", 0.0, 1.0, 1.0);
        }
        if (warm_ok)
        {
            visualizer_.visualize(nubs_full_warm_, route, "full_warm", 1.0, 0.2, 0.2);
        }
        if (nubs_scale_best_.getPieceNum() > 0)
        {
            visualizer_.visualize(nubs_scale_best_, route, "scale_best", 1.0, 1.0, 0.0);
        }
    }
};

int main(int argc, char **argv)
{
    ros::init(argc, argv, "test_planning_node");
    ros::NodeHandle nh;

    TestPlanner planner(Config(ros::NodeHandle("~")), nh);

    ros::Rate loop_rate(100);
    while (ros::ok())
    {
        ros::spinOnce();
        loop_rate.sleep();
    }

    return 0;
}
