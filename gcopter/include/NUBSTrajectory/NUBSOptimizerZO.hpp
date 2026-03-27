#ifndef NUBS_OPTIMIZER_ZERO_ORDER_HPP
#define NUBS_OPTIMIZER_ZERO_ORDER_HPP

#include "NUBSTrajectory.hpp" 
#include <Eigen/Dense>
#include <vector>
#include <cmath>
#include <memory>
#include <type_traits>
#include <iostream>
#include <TrajectoryOptComponents/PolytopeSpatialMap.hpp>
#include <TrajectoryOptComponents/IdentitySpatialMap.hpp>

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

    struct IdentityTimeMapZO
    {
        inline double toTime(double tau) const { return tau; }
        inline double toTau(double T) const { return T; }
    };

    struct QuadInvTimeMapZO
    {
        inline double toTime(double tau) const {
            return tau > 0 ? ((0.5 * tau + 1.0) * tau + 1.0) : (1.0 / ((0.5 * tau - 1.0) * tau + 1.0));
        }
        inline double toTau(double T) const {
            return T > 1.0 ? (std::sqrt(2.0 * T - 1.0) - 1.0) : (1.0 - std::sqrt(2.0 / T - 1.0));
        }
    };

    // =========================================================================
    //  NUBS Control Point Evaluator (Forward kinematics only, NO chain rule)
    // =========================================================================
    template <int DIM>
    class NUBSZeroOrderEvaluator
    {
    public:
        /**
         * @brief 仅计算控制点 C 的各阶导数并评估代价
         */
        template <typename UserCostFunc>
        static double evaluateCost(const Eigen::MatrixXd& C, 
                                   const Eigen::VectorXd& u, 
                                   int p, 
                                   UserCostFunc&& user_cost_func)
        {
            const int n = C.rows();
            if (n < 2) return 0.0;

            Eigen::MatrixXd V(std::max(0, n - 1), DIM);
            Eigen::MatrixXd A(std::max(0, n - 2), DIM);
            Eigen::MatrixXd J(std::max(0, n - 3), DIM);

            // Velocity (C -> V)
            for (int i = 0; i < n - 1; ++i) {
                double dt = u(i + p + 1) - u(i + 1);
                if (dt > 1e-9) V.row(i) = (p / dt) * (C.row(i + 1) - C.row(i));
                else V.row(i).setZero();
            }
            // Acceleration (V -> A)
            for (int i = 0; i < n - 2; ++i) {
                double dt = u(i + p + 1) - u(i + 2);
                if (dt > 1e-9) A.row(i) = ((p - 1) / dt) * (V.row(i + 1) - V.row(i));
                else A.row(i).setZero();
            }
            // Jerk (A -> J)
            for (int i = 0; i < n - 3; ++i) {
                double dt = u(i + p + 1) - u(i + 3);
                if (dt > 1e-9) J.row(i) = ((p - 2) / dt) * (A.row(i + 1) - A.row(i));
                else J.row(i).setZero();
            }

            return user_cost_func(C, V, A, J);
        }
    };

    // =========================================================================
    //  通用 NUBS 零阶求解器引擎 (Designed for CMA-ES, ABC, etc.)
    // =========================================================================
    template <int DIM, int MAX_P = 7,
              typename TimeMap = nubs::QuadInvTimeMapZO,            
              typename SpatialMap = gcopter::PolytopeSpatialMapZO> 
    class NUBSZeroOrderOptimizer
    {
        static_assert(TypeTraitsZO::HasTimeMapInterface<TimeMap>::value, "TimeMap does not satisfy zero-order requirements.");
        static_assert(TypeTraitsZO::HasSpatialMapInterface<SpatialMap, DIM>::value, "SpatialMap does not satisfy zero-order requirements.");

    public:
        using TrajType = NUBSTrajectory<DIM, MAX_P>;
        using MatrixType = Eigen::Matrix<double, Eigen::Dynamic, DIM>;

        struct Workspace
        {
            EIGEN_MAKE_ALIGNED_OPERATOR_NEW
            Eigen::VectorXd cache_T;
            MatrixType cache_P_inner;
            Eigen::MatrixXd headState;
            Eigen::MatrixXd tailState;
            Eigen::MatrixXd P_full; 
        };

    private:
        TrajType traj_;
        int num_segments_ = 0;
        int s_ = 0;
        
        std::vector<double> ref_times_;
        MatrixType ref_waypoints_;
        
        TimeMap default_time_map_;
        SpatialMap default_spatial_map_;
        
        const TimeMap* active_time_map_ = nullptr;
        const SpatialMap* active_spatial_map_ = nullptr;
        
        mutable std::unique_ptr<Workspace> ws_;
        double rho_energy_ = 1.0; 

    public:
        EIGEN_MAKE_ALIGNED_OPERATOR_NEW

        NUBSZeroOrderOptimizer(int sys_order = 3) : traj_(sys_order), s_(sys_order) 
        {
            ws_ = std::make_unique<Workspace>();
        }

        void setEnergyWeights(double rho) { rho_energy_ = rho; }
        void setTimeMap(const TimeMap* tm) { active_time_map_ = (tm != nullptr) ? tm : &default_time_map_; }
        void setSpatialMap(const SpatialMap* sm) { active_spatial_map_ = (sm != nullptr) ? sm : &default_spatial_map_; }

        bool setInitState(const std::vector<double> &time_segments,
                          const MatrixType &waypoints,
                          const Eigen::MatrixXd &boundary_head, 
                          const Eigen::MatrixXd &boundary_tail) 
        {
            num_segments_ = time_segments.size();
            ref_times_ = time_segments;
            ref_waypoints_ = waypoints;

            ws_->cache_T.resize(num_segments_);
            if (num_segments_ > 1) {
                ws_->cache_P_inner.resize(num_segments_ - 1, DIM);
            }
            ws_->headState = boundary_head;
            ws_->tailState = boundary_tail;

            return true;
        }

        Eigen::VectorXd generateInitialGuess() const
        {
            int dim_T = num_segments_;
            int dim_P = 0;
            for (int i = 1; i < num_segments_; ++i) {
                dim_P += active_spatial_map_->getUnconstrainedDim(i);
            }

            Eigen::VectorXd x(dim_T + dim_P);

            for (int i = 0; i < num_segments_; ++i) {
                x(i) = active_time_map_->toTau(ref_times_[i]);
            }

            int offset = dim_T;
            for (int i = 1; i < num_segments_; ++i) {
                int dof = active_spatial_map_->getUnconstrainedDim(i);
                x.segment(offset, dof) = active_spatial_map_->toUnconstrained(ref_waypoints_.row(i).transpose(), i);
                offset += dof;
            }

            return x;
        }

        /**
         * @brief 零阶评估函数：纯前向传递，输入参数向量 x，直接输出标量 Cost。
         */
        template <typename TimeCostFunc, typename ControlPointCostFunc>
        double evaluate(const Eigen::Ref<const Eigen::VectorXd>& x, 
                        TimeCostFunc &&time_cost_func,
                        ControlPointCostFunc &&cp_cost_func)
        {
            static_assert(TypeTraitsZO::HasTimeCostInterface<typename std::decay<TimeCostFunc>::type>::value, 
                          "TimeCostFunc does not satisfy zero-order requirements.");

            double total_cost = 0.0;

            // 1. Time Mapping (无约束 -> 物理正时间)
            for (int i = 0; i < num_segments_; ++i) 
            {
                ws_->cache_T(i) = active_time_map_->toTime(x(i));
            }

            // 2. Spatial Mapping (无约束 -> 物理空间点)
            int offset = num_segments_;
            for (int i = 1; i < num_segments_; ++i) 
            {
                int dof = active_spatial_map_->getUnconstrainedDim(i);
                Eigen::VectorXd xi = x.segment(offset, dof);
                ws_->cache_P_inner.row(i - 1) = active_spatial_map_->toPhysical(xi, i).transpose();
                
                // 累计映射本身的惩罚代价（如软约束越界）
                total_cost += active_spatial_map_->getNormPenalty(xi);
                offset += dof;
            }

            // 3. Generate Trajectory 
            traj_.generate(ws_->cache_P_inner, ws_->headState, ws_->tailState, ws_->cache_T, ws_->P_full);

            // 4. Energy Cost 
            if (rho_energy_ > 0) 
            {
                total_cost += rho_energy_ * traj_.getEnergy(); 
            }

            // 5. User Time Cost
            std::vector<double> T_vec(ws_->cache_T.data(), ws_->cache_T.data() + ws_->cache_T.size());
            total_cost += time_cost_func(T_vec);

            // 6. User Control Point Cost (动力学超限、避障等)
            double cp_cost = NUBSZeroOrderEvaluator<DIM>::evaluateCost(
                traj_.getControlPoints(), 
                traj_.getKnots(), 
                traj_.getP(), 
                std::forward<ControlPointCostFunc>(cp_cost_func)
            );
            
            total_cost += cp_cost;

            return total_cost;
        }

        const TrajType& getTrajectory() const { return traj_; }

        // =========================================================================
        //  代价透视分析工具 (Cost Debugger)
        // =========================================================================
        struct CostBreakdown
        {
            double total_cost = 0.0;
            double spatial_mapping_penalty = 0.0;
            double energy_cost = 0.0;
            double time_cost = 0.0;
            double cp_cost = 0.0;

            void print(const std::string& prefix = "") const {
                std::cout << "\n========== " << prefix << " Cost Breakdown ==========\n";
                std::cout << " 1. Spatial Map Penalty : " << spatial_mapping_penalty << "\n";
                std::cout << " 2. Energy (Smoothness) : " << energy_cost << "\n";
                std::cout << " 3. Time Duration Cost  : " << time_cost << "\n";
                std::cout << " 4. Control Point Cost  : " << cp_cost << "  <-- (Check details below)\n";
                std::cout << "-------------------------------------------\n";
                std::cout << " TOTAL AGGREGATE COST   : " << total_cost << "\n";
                std::cout << "===========================================\n";
            }
        };

        /**
         * @brief 仅用于调试：完全重演一次 evaluate，但拆分记录每一项的代价
         */
        template <typename TimeCostFunc, typename ControlPointCostFunc>
        CostBreakdown debugEvaluate(const Eigen::Ref<const Eigen::VectorXd>& x, 
                                    TimeCostFunc &&time_cost_func,
                                    ControlPointCostFunc &&cp_cost_func)
        {
            CostBreakdown brk;

            // 1. Time Mapping
            for (int i = 0; i < num_segments_; ++i) {
                ws_->cache_T(i) = active_time_map_->toTime(x(i));
            }

            // 2. Spatial Mapping & Penalty
            int offset = num_segments_;
            for (int i = 1; i < num_segments_; ++i) {
                int dof = active_spatial_map_->getUnconstrainedDim(i);
                Eigen::VectorXd xi = x.segment(offset, dof);
                ws_->cache_P_inner.row(i - 1) = active_spatial_map_->toPhysical(xi, i).transpose();
                brk.spatial_mapping_penalty += active_spatial_map_->getNormPenalty(xi);
                offset += dof;
            }

            // 3. Generate Trajectory
            traj_.generate(ws_->cache_P_inner, ws_->headState, ws_->tailState, ws_->cache_T, ws_->P_full);

            // 4. Energy
            if (rho_energy_ > 0) {
                brk.energy_cost = rho_energy_ * traj_.getEnergy();
            }

            // 5. Time
            std::vector<double> T_vec(ws_->cache_T.data(), ws_->cache_T.data() + ws_->cache_T.size());
            brk.time_cost = time_cost_func(T_vec);

            // 6. Control Points
            brk.cp_cost = NUBSZeroOrderEvaluator<DIM>::evaluateCost(
                traj_.getControlPoints(), traj_.getKnots(), traj_.getP(), 
                std::forward<ControlPointCostFunc>(cp_cost_func));

            brk.total_cost = brk.spatial_mapping_penalty + brk.energy_cost + brk.time_cost + brk.cp_cost;
            return brk;
        }
    };

} // namespace nubs_zo

#endif // NUBS_OPTIMIZER_ZERO_ORDER_HPP