#ifndef NUBS_OPTIMIZER_ZO_HPP
#define NUBS_OPTIMIZER_ZO_HPP

#include "NUBSTrajectory/NUBSTrajectory.hpp" 
#include <Eigen/Dense>
#include <vector>
#include <cmath>
#include <memory>
#include <type_traits>
#include <iostream>
#include "TrajectoryOptComponents/PolytopeSpatialMapZO.hpp"

namespace nubs
{
    namespace TypeTraitsZO 
    {
        template <typename...> using void_t = void;
        template <typename T, typename = void> struct HasTimeMapInterface : std::false_type {};
        template <typename T> struct HasTimeMapInterface<T, void_t<
            decltype(static_cast<double>(std::declval<T>().toTime(std::declval<double>()))),
            decltype(static_cast<double>(std::declval<T>().toTau(std::declval<double>())))
        >> : std::true_type {};

        template <typename T, int DIM, typename = void> struct HasSpatialMapInterface : std::false_type {};
        template <typename T, int DIM> struct HasSpatialMapInterface<T, DIM, void_t<
            decltype(static_cast<int>(std::declval<T>().getUnconstrainedDim(std::declval<int>()))),
            decltype(std::declval<T>().toPhysical(std::declval<Eigen::VectorXd>(), std::declval<int>())),
            decltype(std::declval<T>().toUnconstrained(std::declval<Eigen::Matrix<double, DIM, 1>>(), std::declval<int>())),
            decltype(static_cast<double>(std::declval<T>().getNormPenalty(std::declval<Eigen::VectorXd>())))
        >> : std::true_type {};

        template <typename T, typename = void> struct HasTimeCostInterface : std::false_type {};
        template <typename T> struct HasTimeCostInterface<T, void_t<
            decltype(static_cast<double>(std::declval<T>()(std::declval<const std::vector<double>&>())))
        >> : std::true_type {};
    }

    struct QuadInvTimeMapZO
    {
        inline double toTime(double tau) const { return tau > 0 ? ((0.5 * tau + 1.0) * tau + 1.0) : (1.0 / ((0.5 * tau - 1.0) * tau + 1.0)); }
        inline double toTau(double T) const { return T > 1.0 ? (std::sqrt(2.0 * T - 1.0) - 1.0) : (1.0 - std::sqrt(2.0 / T - 1.0)); }
    };

    template <int DIM>
    class NUBSZeroOrderEvaluator
    {
    public:
        template <typename UserCostFunc>
        static double evaluateCost(const Eigen::MatrixXd& C, const Eigen::VectorXd& u, int p, UserCostFunc&& user_cost_func)
        {
            const int n = C.rows();
            if (n < 2) return 0.0;
            Eigen::MatrixXd V(std::max(0, n - 1), DIM);
            Eigen::MatrixXd A(std::max(0, n - 2), DIM);
            Eigen::MatrixXd J(std::max(0, n - 3), DIM);

            for (int i = 0; i < n - 1; ++i) {
                double dt = u(i + p + 1) - u(i + 1);
                if (dt > 1e-9) V.row(i) = (p / dt) * (C.row(i + 1) - C.row(i)); else V.row(i).setZero();
            }
            for (int i = 0; i < n - 2; ++i) {
                double dt = u(i + p + 1) - u(i + 2);
                if (dt > 1e-9) A.row(i) = ((p - 1) / dt) * (V.row(i + 1) - V.row(i)); else A.row(i).setZero();
            }
            for (int i = 0; i < n - 3; ++i) {
                double dt = u(i + p + 1) - u(i + 3);
                if (dt > 1e-9) J.row(i) = ((p - 2) / dt) * (A.row(i + 1) - A.row(i)); else J.row(i).setZero();
            }
            return user_cost_func(C, V, A, J);
        }
    };

    template <int DIM, int MAX_P = 7, typename TimeMap = nubs::QuadInvTimeMapZO, typename SpatialMap = gcopter::PolytopeSpatialMapZO>
    class NUBSZeroOrderOptimizer
    {
        static_assert(TypeTraitsZO::HasTimeMapInterface<TimeMap>::value, "TimeMapZO failed");
        static_assert(TypeTraitsZO::HasSpatialMapInterface<SpatialMap, DIM>::value, "SpatialMapZO failed");

    public:
        using TrajType = NUBSTrajectory<DIM, MAX_P>;
        using MatrixType = Eigen::Matrix<double, Eigen::Dynamic, DIM>;

        struct Workspace {
            EIGEN_MAKE_ALIGNED_OPERATOR_NEW
            Eigen::VectorXd cache_T;
            MatrixType cache_C_free;
            Eigen::MatrixXd headState;
            Eigen::MatrixXd tailState;
            Eigen::MatrixXd P_full; 
        };

        struct CostBreakdown {
            double total_cost = 0.0, spatial_mapping_penalty = 0.0, energy_cost = 0.0, time_cost = 0.0, cp_cost = 0.0;
        };

    private:
        TrajType traj_;
        int num_segments_ = 0;
        std::vector<double> ref_times_;
        MatrixType ref_waypoints_;
        TimeMap active_time_map_;
        SpatialMap* active_spatial_map_ = nullptr;
        mutable std::unique_ptr<Workspace> ws_;
        double rho_energy_ = 1.0; 

    public:
        EIGEN_MAKE_ALIGNED_OPERATOR_NEW
        NUBSZeroOrderOptimizer(int sys_order = 3) : traj_(sys_order) { ws_ = std::make_unique<Workspace>(); }

        void setEnergyWeights(double rho) { rho_energy_ = rho; }
        void setSpatialMap(SpatialMap* sm) { active_spatial_map_ = sm; }

        bool setInitState(const std::vector<double> &time_segments, const MatrixType &waypoints, const Eigen::MatrixXd &head, const Eigen::MatrixXd &tail) {
            num_segments_ = time_segments.size();
            ref_times_ = time_segments; ref_waypoints_ = waypoints;
            ws_->cache_T.resize(num_segments_);
            if (num_segments_ > 1) ws_->cache_C_free.resize(num_segments_ - 1, DIM);
            ws_->headState = head; ws_->tailState = tail;
            return true;
        }

        Eigen::VectorXd generateInitialGuess() const {
            int dim_P = 0;
            for (int i = 1; i < num_segments_; ++i) dim_P += active_spatial_map_->getUnconstrainedDim(i);
            Eigen::VectorXd x(num_segments_ + dim_P);
            for (int i = 0; i < num_segments_; ++i) x(i) = active_time_map_.toTau(ref_times_[i]);
            int offset = num_segments_;
            for (int i = 1; i < num_segments_; ++i) {
                int dof = active_spatial_map_->getUnconstrainedDim(i);
                x.segment(offset, dof) = active_spatial_map_->toUnconstrained(ref_waypoints_.row(i).transpose(), i);
                offset += dof;
            }
            return x;
        }

        template <typename TimeCostFunc, typename ControlPointCostFunc>
        double evaluate(const Eigen::Ref<const Eigen::VectorXd>& x, TimeCostFunc &&time_cost_func, ControlPointCostFunc &&cp_cost_func)
        {
            double total_cost = 0.0;
            for (int i = 0; i < num_segments_; ++i) ws_->cache_T(i) = active_time_map_.toTime(x(i));
            int offset = num_segments_;
            for (int i = 1; i < num_segments_; ++i) {
                int dof = active_spatial_map_->getUnconstrainedDim(i);
                Eigen::VectorXd xi = x.segment(offset, dof);
                ws_->cache_C_free.row(i - 1) = active_spatial_map_->toPhysical(xi, i).transpose();
                total_cost += active_spatial_map_->getNormPenalty(xi);
                offset += dof;
            }
            traj_.generateFromFreeControlPoints(ws_->cache_C_free, ws_->headState, ws_->tailState, ws_->cache_T);
            std::vector<double> T_vec(ws_->cache_T.data(), ws_->cache_T.data() + ws_->cache_T.size());
            total_cost += time_cost_func(T_vec);
            total_cost += NUBSZeroOrderEvaluator<DIM>::evaluateCost(traj_.getControlPoints(), traj_.getKnots(), traj_.getP(), std::forward<ControlPointCostFunc>(cp_cost_func));
            return total_cost;
        }

        const TrajType& getTrajectory() const { return traj_; }
    };
} 
#endif