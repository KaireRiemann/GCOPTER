#ifndef NUBS_SFC_OPTIMIZER_ZO_HPP
#define NUBS_SFC_OPTIMIZER_ZO_HPP

#include "NUBSTrajectory/NUBSOptimizerZO.hpp" 
#include "TrajectoryOptComponents/LinearTimeCost.hpp"
#include "TrajectoryOptComponents/SFCControlPointsCostsZO.hpp"
#include "TrajectoryOptComponents/PolytopeSpatialMapZO.hpp"
#include "gcopter/abc_solver.hpp"
#include "gcopter/dpasa_solver.hpp"
#include "gcopter/geo_utils.hpp"
#include "gcopter/lbfgs.hpp" 

#include <Eigen/Eigen>
#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <limits>
#include <vector>
#include <iostream>

namespace gcopter
{
    class NUBSSFCOptimizerZO
    {
    public:
        enum class SolverBackend
        {
            ABC,
            DPASA
        };

        using TrajType = nubs::NUBSTrajectory<3, 7>;
        using OptimizerType = nubs::NUBSZeroOrderOptimizer<3, 7, nubs::QuadInvTimeMapZO, gcopter::PolytopeSpatialMapZO>;
        typedef Eigen::Matrix3Xd PolyhedronV;
        typedef Eigen::MatrixX4d PolyhedronH;
        typedef std::vector<PolyhedronV> PolyhedraV;
        typedef std::vector<PolyhedronH> PolyhedraH;

    private:
        OptimizerType optimizer_;
        gcopter::PolytopeSpatialMapZO spatial_map_;
        NUBSTimeCostZO time_cost_;
        SFCControlPointCostZO cp_cost_;

        Eigen::Matrix3d headState_, tailState_;
        PolyhedraV vPolytopes_; PolyhedraH hPolytopes_;
        Eigen::Matrix3Xd shortPath_;
        Eigen::VectorXi pieceIdx_, vPolyIdx_, hPolyIdx_;
        int polyN_ = 0, pieceN_ = 0;
        Eigen::VectorXd magnitudeBd_, penaltyWt_;
        double allocSpeed_ = 0.0;
        SolverBackend backend_ = SolverBackend::DPASA;

    public:
        explicit NUBSSFCOptimizerZO(SolverBackend backend = SolverBackend::DPASA)
            : backend_(backend)
        {
        }

        void setSolverBackend(SolverBackend backend)
        {
            backend_ = backend;
        }

    private:
        struct ObjectiveSummary
        {
            double total_cost = std::numeric_limits<double>::infinity();
            double base_cost = std::numeric_limits<double>::infinity();
            double sampled_corridor_cost = 0.0;
            double sampled_speed_cost = 0.0;
            double sampled_thrust_cost = 0.0;
            double max_corridor_violation = 0.0;
            double max_speed_violation = 0.0;
            double max_thrust_violation = 0.0;
            bool finite = false;
            bool feasible = false;
        };

        static inline double costDistance(void *ptr,
                                          const Eigen::VectorXd &xi,
                                          Eigen::VectorXd &gradXi)
        {
            void **dataPtrs = (void **)ptr;
            const double &dEps = *((const double *)(dataPtrs[0]));
            const Eigen::Vector3d &ini = *((const Eigen::Vector3d *)(dataPtrs[1]));
            const Eigen::Vector3d &fin = *((const Eigen::Vector3d *)(dataPtrs[2]));
            const PolyhedraV &vPolys = *((PolyhedraV *)(dataPtrs[3]));

            double cost = 0.0;
            const int overlaps = vPolys.size() / 2;

            Eigen::Matrix3Xd gradP = Eigen::Matrix3Xd::Zero(3, overlaps);
            Eigen::Vector3d a, b, d;
            Eigen::VectorXd r;
            double smoothedDistance;
            for (int i = 0, j = 0, k = 0; i <= overlaps; i++, j += k)
            {
                a = i == 0 ? ini : b;
                if (i < overlaps)
                {
                    k = vPolys[2 * i + 1].cols();
                    Eigen::Map<const Eigen::VectorXd> q(xi.data() + j, k);
                    r = q.normalized().head(k - 1);
                    b = vPolys[2 * i + 1].rightCols(k - 1) * r.cwiseProduct(r) +
                        vPolys[2 * i + 1].col(0);
                }
                else
                {
                    b = fin;
                }

                d = b - a;
                smoothedDistance = sqrt(d.squaredNorm() + dEps);
                cost += smoothedDistance;

                if (i < overlaps)
                {
                    gradP.col(i) += d / smoothedDistance;
                }
                if (i > 0)
                {
                    gradP.col(i - 1) -= d / smoothedDistance;
                }
            }

            Eigen::VectorXd unitQ;
            double sqrNormQ, invNormQ, sqrNormViolation, c, dc;
            for (int i = 0, j = 0, k; i < overlaps; i++, j += k)
            {
                k = vPolys[2 * i + 1].cols();
                Eigen::Map<const Eigen::VectorXd> q(xi.data() + j, k);
                Eigen::Map<Eigen::VectorXd> gradQ(gradXi.data() + j, k);
                sqrNormQ = q.squaredNorm();
                invNormQ = 1.0 / sqrt(sqrNormQ);
                unitQ = q * invNormQ;
                gradQ.head(k - 1) = (vPolys[2 * i + 1].rightCols(k - 1).transpose() * gradP.col(i)).array() *
                                    unitQ.head(k - 1).array() * 2.0;
                gradQ(k - 1) = 0.0;
                gradQ = (gradQ - unitQ * unitQ.dot(gradQ)).eval() * invNormQ;

                sqrNormViolation = sqrNormQ - 1.0;
                if (sqrNormViolation > 0.0)
                {
                    c = sqrNormViolation * sqrNormViolation;
                    dc = 3.0 * c;
                    c *= sqrNormViolation;
                    cost += c;
                    gradQ += dc * 2.0 * q;
                }
            }

            return cost;
        }

        static inline void getShortestPath(const Eigen::Vector3d &ini,
                                           const Eigen::Vector3d &fin,
                                           const PolyhedraV &vPolys,
                                           const double &smoothD,
                                           Eigen::Matrix3Xd &path)
        {
            const int overlaps = vPolys.size() / 2;
            Eigen::VectorXi vSizes(overlaps);
            for (int i = 0; i < overlaps; i++)
            {
                vSizes(i) = vPolys[2 * i + 1].cols();
            }
            Eigen::VectorXd xi(vSizes.sum());
            for (int i = 0, j = 0; i < overlaps; i++)
            {
                xi.segment(j, vSizes(i)).setConstant(sqrt(1.0 / vSizes(i)));
                j += vSizes(i);
            }

            double minDistance;
            void *dataPtrs[4];
            dataPtrs[0] = (void *)(&smoothD);
            dataPtrs[1] = (void *)(&ini);
            dataPtrs[2] = (void *)(&fin);
            dataPtrs[3] = (void *)(&vPolys);
            lbfgs::lbfgs_parameter_t shortest_path_params;
            shortest_path_params.past = 3;
            shortest_path_params.delta = 1.0e-3;
            shortest_path_params.g_epsilon = 1.0e-5;

            lbfgs::lbfgs_optimize(xi,
                                  minDistance,
                                  &NUBSSFCOptimizerZO::costDistance,
                                  nullptr,
                                  nullptr,
                                  dataPtrs,
                                  shortest_path_params);

            path.resize(3, overlaps + 2);
            path.leftCols<1>() = ini;
            path.rightCols<1>() = fin;
            Eigen::VectorXd r;
            for (int i = 0, j = 0, k; i < overlaps; i++, j += k)
            {
                k = vPolys[2 * i + 1].cols();
                Eigen::Map<const Eigen::VectorXd> q(xi.data() + j, k);
                r = q.normalized().head(k - 1);
                path.col(i + 1) = vPolys[2 * i + 1].rightCols(k - 1) * r.cwiseProduct(r) +
                                  vPolys[2 * i + 1].col(0);
            }

            return;
        }

        static inline bool processCorridor(const PolyhedraH &hPs,
                                           PolyhedraV &vPs)
        {
            const int sizeCorridor = hPs.size() - 1;

            vPs.clear();
            vPs.reserve(2 * sizeCorridor + 1);

            int nv;
            PolyhedronH curIH;
            PolyhedronV curIV, curIOB;
            for (int i = 0; i < sizeCorridor; i++)
            {
                if (!geo_utils::enumerateVs(hPs[i], curIV))
                {
                    return false;
                }
                nv = curIV.cols();
                curIOB.resize(3, nv);
                curIOB.col(0) = curIV.col(0);
                curIOB.rightCols(nv - 1) = curIV.rightCols(nv - 1).colwise() - curIV.col(0);
                vPs.push_back(curIOB);

                curIH.resize(hPs[i].rows() + hPs[i + 1].rows(), 4);
                curIH.topRows(hPs[i].rows()) = hPs[i];
                curIH.bottomRows(hPs[i + 1].rows()) = hPs[i + 1];
                if (!geo_utils::enumerateVs(curIH, curIV))
                {
                    return false;
                }
                nv = curIV.cols();
                curIOB.resize(3, nv);
                curIOB.col(0) = curIV.col(0);
                curIOB.rightCols(nv - 1) = curIV.rightCols(nv - 1).colwise() - curIV.col(0);
                vPs.push_back(curIOB);
            }

            if (!geo_utils::enumerateVs(hPs.back(), curIV))
            {
                return false;
            }
            nv = curIV.cols();
            curIOB.resize(3, nv);
            curIOB.col(0) = curIV.col(0);
            curIOB.rightCols(nv - 1) = curIV.rightCols(nv - 1).colwise() - curIV.col(0);
            vPs.push_back(curIOB);

            return true;
        }

        static inline void setInitial(const Eigen::Matrix3Xd &path,
                                      const double &speed,
                                      const Eigen::VectorXi &intervalNs,
                                      Eigen::Matrix3Xd &innerPoints,
                                      Eigen::VectorXd &timeAlloc)
        {
            const int sizeM = intervalNs.size();
            const int sizeN = intervalNs.sum();
            innerPoints.resize(3, sizeN - 1);
            timeAlloc.resize(sizeN);

            Eigen::Vector3d a, b, c;
            for (int i = 0, j = 0, k = 0, l; i < sizeM; i++)
            {
                l = intervalNs(i);
                a = path.col(i);
                b = path.col(i + 1);
                c = (b - a) / l;
                timeAlloc.segment(j, l).setConstant(c.norm() / speed);
                j += l;
                for (int m = 0; m < l; m++)
                {
                    if (i > 0 || m > 0)
                    {
                        innerPoints.col(k++) = a + c * m;
                    }
                }
            }
        }

        static inline double positiveToPhysicalTime(const double tau)
        {
            return nubs::QuadInvTimeMapZO().toTime(tau);
        }

        static inline double physicalTimeToPositive(const double time)
        {
            return nubs::QuadInvTimeMapZO().toTau(std::max(time, 1.0e-8));
        }

        static inline Eigen::VectorXd scaleProfileToPhysicalTimes(const Eigen::VectorXd &encoded)
        {
            const int num_segments = encoded.size();
            Eigen::VectorXd times(num_segments);
            if (num_segments <= 0)
            {
                return times;
            }

            const double total_time = positiveToPhysicalTime(encoded(0));
            if (num_segments == 1)
            {
                times(0) = total_time;
                return times;
            }

            Eigen::VectorXd logits(num_segments);
            logits.head(num_segments - 1) = encoded.tail(num_segments - 1);
            logits(num_segments - 1) = 0.0;

            const double max_logit = logits.maxCoeff();
            Eigen::VectorXd exp_logits = (logits.array() - max_logit).exp();
            const double exp_sum = std::max(exp_logits.sum(), 1.0e-8);
            const Eigen::VectorXd alpha = exp_logits / exp_sum;
            times = total_time * alpha;
            return times;
        }

        static inline Eigen::VectorXd physicalTimesToScaleProfile(const Eigen::VectorXd &times)
        {
            const int num_segments = times.size();
            Eigen::VectorXd encoded = Eigen::VectorXd::Zero(num_segments);
            if (num_segments <= 0)
            {
                return encoded;
            }

            const double total_time = std::max(times.sum(), 1.0e-8);
            encoded(0) = physicalTimeToPositive(total_time);

            if (num_segments == 1)
            {
                return encoded;
            }

            Eigen::VectorXd alpha(num_segments);
            for (int i = 0; i < num_segments; ++i)
            {
                alpha(i) = std::max(times(i) / total_time, 1.0e-8);
            }
            alpha /= alpha.sum();

            const double anchor = std::log(alpha(num_segments - 1));
            for (int i = 0; i < num_segments - 1; ++i)
            {
                encoded(1 + i) = std::log(alpha(i)) - anchor;
            }
            return encoded;
        }

        Eigen::RowVectorXd evaluatorToDecision(const Eigen::VectorXd &x_eval) const
        {
            const int time_dim = pieceN_;
            const int spatial_dim = x_eval.size() - time_dim;
            Eigen::RowVectorXd theta(x_eval.size());

            if (spatial_dim > 0)
            {
                theta.head(spatial_dim) = x_eval.tail(spatial_dim).transpose();
            }

            Eigen::VectorXd times(time_dim);
            for (int i = 0; i < time_dim; ++i)
            {
                times(i) = positiveToPhysicalTime(x_eval(i));
            }
            theta.tail(time_dim) = physicalTimesToScaleProfile(times).transpose();
            return theta;
        }

        Eigen::VectorXd decisionToEvaluator(const Eigen::RowVectorXd &theta) const
        {
            const int time_dim = pieceN_;
            const int spatial_dim = theta.size() - time_dim;
            Eigen::VectorXd x_eval(theta.size());

            Eigen::VectorXd times = scaleProfileToPhysicalTimes(theta.tail(time_dim).transpose());
            for (int i = 0; i < time_dim; ++i)
            {
                x_eval(i) = physicalTimeToPositive(times(i));
            }
            if (spatial_dim > 0)
            {
                x_eval.tail(spatial_dim) = theta.head(spatial_dim).transpose();
            }
            return x_eval;
        }

        double evaluateDecision(const Eigen::RowVectorXd &theta,
                                const NUBSTimeCostZO &time_cost,
                                const SFCControlPointCostZO &cp_cost)
        {
            return evaluateObjective(theta, time_cost, cp_cost).total_cost;
        }

        ObjectiveSummary evaluateObjective(const Eigen::RowVectorXd &theta,
                                           const NUBSTimeCostZO &time_cost,
                                           const SFCControlPointCostZO &cp_cost)
        {
            ObjectiveSummary summary;
            const Eigen::VectorXd x_eval = decisionToEvaluator(theta);
            summary.base_cost = optimizer_.evaluate(x_eval, time_cost, cp_cost);
            summary.total_cost = summary.base_cost;
            summary.finite = std::isfinite(summary.base_cost);
            if (summary.finite)
            {
                const auto &traj = optimizer_.getTrajectory();
                const auto &C = traj.getControlPoints();
                const auto &u = traj.getKnots();
                const int p = traj.getP();
                const int n = static_cast<int>(C.rows());

                Eigen::MatrixXd V(std::max(0, n - 1), 3);
                Eigen::MatrixXd A(std::max(0, n - 2), 3);
                for (int i = 0; i < n - 1; ++i)
                {
                    const double dt = u(i + p + 1) - u(i + 1);
                    if (dt > 1.0e-9)
                    {
                        V.row(i) = (p / dt) * (C.row(i + 1) - C.row(i));
                    }
                    else
                    {
                        V.row(i).setZero();
                    }
                }
                for (int i = 0; i < n - 2; ++i)
                {
                    const double dt = u(i + p + 1) - u(i + 2);
                    if (dt > 1.0e-9)
                    {
                        A.row(i) = ((p - 1) / dt) * (V.row(i + 1) - V.row(i));
                    }
                    else
                    {
                        A.row(i).setZero();
                    }
                }

                summary.max_corridor_violation = cp_cost.maxPieceHullViolation(C);
                summary.max_speed_violation = cp_cost.maxSpeedViolation(V);
                summary.max_thrust_violation = cp_cost.maxThrustViolation(A);
            }
            summary.feasible = summary.finite &&
                               summary.max_corridor_violation <= 1.0e-6 &&
                               summary.max_speed_violation <= 1.0e-6 &&
                               summary.max_thrust_violation <= 1.0e-6;
            if (!summary.finite)
            {
                summary.total_cost = 1.0e20;
            }
            return summary;
        }

        void buildDecisionBounds(const Eigen::RowVectorXd &theta0,
                                 Eigen::VectorXd &lb,
                                 Eigen::VectorXd &ub) const
        {
            const int dim = theta0.size();
            const int time_dim = pieceN_;
            const int spatial_dim = dim - time_dim;

            lb.resize(dim);
            ub.resize(dim);

            for (int i = 0; i < spatial_dim; ++i)
            {
                lb(i) = theta0(i) - 1.0;
                ub(i) = theta0(i) + 1.0;
            }

            const Eigen::VectorXd initial_times = scaleProfileToPhysicalTimes(theta0.tail(time_dim).transpose());
            const double initial_total_time = std::max(initial_times.sum(), 1.0e-3);
            lb(spatial_dim) = physicalTimeToPositive(initial_total_time * 0.85);
            ub(spatial_dim) = physicalTimeToPositive(initial_total_time * 2.20);

            for (int i = 0; i < time_dim - 1; ++i)
            {
                const int idx = spatial_dim + 1 + i;
                lb(idx) = theta0(idx) - 2.5;
                ub(idx) = theta0(idx) + 2.5;
            }
        }

        double optimizeWithABC(const Eigen::RowVectorXd &theta0,
                               const Eigen::VectorXd &lb,
                               const Eigen::VectorXd &ub,
                               TrajType &spline)
        {
            auto obj_func = [this](const Eigen::RowVectorXd &theta) -> std::pair<double, bool>
            {
                const ObjectiveSummary summary = evaluateObjective(theta, time_cost_, cp_cost_);
                return {summary.total_cost, summary.feasible};
            };

            ABC abc;
            const int dim = theta0.size();
            const int n_pop = 24;
            const int max_it = 36;
            abc.initializeWithPriority(n_pop, max_it, ub, lb, dim,
                                       n_pop / 2, 0.08, theta0);
            const auto [best_cost_unused, best_theta] = abc.optimize(n_pop, max_it, ub, lb, dim, obj_func);
            (void)best_cost_unused;

            const double final_cost = evaluateObjective(best_theta, time_cost_, cp_cost_).total_cost;
            spline = optimizer_.getTrajectory();
            return final_cost;
        }

        double optimizeWithDPASA(const Eigen::RowVectorXd &theta0,
                                 const Eigen::VectorXd &lb,
                                 const Eigen::VectorXd &ub,
                                 TrajType &spline)
        {
            const int dim = theta0.size();
            const int time_dim = pieceN_;
            const int spatial_dim = dim - time_dim;

            gcopter::DPASABlockLayout layout;
            layout.spatial = {0, spatial_dim};
            layout.scale = {spatial_dim, 1};
            layout.profile = {spatial_dim + 1, std::max(0, time_dim - 1)};

            NUBSTimeCostZO feasibility_time_cost = time_cost_;
            SFCControlPointCostZO feasibility_cp_cost = cp_cost_;
            feasibility_time_cost.weight *= 0.80;
            feasibility_cp_cost.weight_v *= 1.50;
            feasibility_cp_cost.weight_thrust *= 1.50;
            feasibility_cp_cost.weight_pos *= 1.25;

            NUBSTimeCostZO refinement_time_cost = time_cost_;
            SFCControlPointCostZO refinement_cp_cost = cp_cost_;
            refinement_time_cost.weight *= 1.35;
            refinement_cp_cost.weight_v *= 0.65;
            refinement_cp_cost.weight_thrust *= 0.65;

            auto makeObjective = [this](const NUBSTimeCostZO &time_cost,
                                        const SFCControlPointCostZO &cp_cost)
            {
                return [this, time_cost, cp_cost](const Eigen::RowVectorXd &theta) -> std::pair<double, bool>
                {
                    const ObjectiveSummary summary = evaluateObjective(theta, time_cost, cp_cost);
                    return {summary.total_cost, summary.feasible};
                };
            };

            gcopter::DPASASolverOptions stage1_options;
            stage1_options.n_candidates = 16;
            stage1_options.n_strategies = 6;
            stage1_options.max_iterations = 14;
            stage1_options.elite_size = 5;
            stage1_options.strategy_sample_size = 3;
            stage1_options.top_strategy_count = 2;
            stage1_options.convergence_window = 4;
            stage1_options.convergence_tol = 5.0e-4;

            gcopter::DPASASolverOptions stage2_options = stage1_options;
            stage2_options.n_candidates = 18;
            stage2_options.max_iterations = 18;
            stage2_options.init_spatial_radius = 0.05;
            stage2_options.init_time_radius = 0.08;
            stage2_options.seed += 17u;

            gcopter::DPASASolver stage1_solver(stage1_options);
            const gcopter::DPASASolverResult stage1_result =
                stage1_solver.optimize(ub, lb, layout, theta0, makeObjective(feasibility_time_cost, feasibility_cp_cost));

            const Eigen::RowVectorXd stage2_seed =
                stage1_result.best_position.size() == dim ? stage1_result.best_position : theta0;
            gcopter::DPASASolver stage2_solver(stage2_options);
            const gcopter::DPASASolverResult stage2_result =
                stage2_solver.optimize(ub, lb, layout, stage2_seed, makeObjective(refinement_time_cost, refinement_cp_cost));

            Eigen::RowVectorXd best_theta = theta0;
            double final_cost = evaluateObjective(theta0, time_cost_, cp_cost_).total_cost;

            auto try_update_best = [this, &best_theta, &final_cost](const Eigen::RowVectorXd &theta)
            {
                if (theta.size() == 0)
                {
                    return;
                }
                const double candidate_cost = evaluateObjective(theta, time_cost_, cp_cost_).total_cost;
                if (std::isfinite(candidate_cost) && candidate_cost < final_cost)
                {
                    final_cost = candidate_cost;
                    best_theta = theta;
                }
            };

            try_update_best(stage1_result.best_position);
            try_update_best(stage2_result.best_position);

            final_cost = evaluateObjective(best_theta, time_cost_, cp_cost_).total_cost;
            spline = optimizer_.getTrajectory();
            return final_cost;
        }

    public:
        bool setup(const double &timeWeight, const Eigen::Matrix3d &initialPVA, const Eigen::Matrix3d &terminalPVA,
                   const PolyhedraH &safeCorridor, const double &lengthPerPiece,
                   const Eigen::VectorXd &magnitudeBounds, const Eigen::VectorXd &penaltyWeights) 
        {
            headState_ = initialPVA; tailState_ = terminalPVA; hPolytopes_ = safeCorridor;
            for (size_t i = 0; i < hPolytopes_.size(); i++) {
                const Eigen::ArrayXd norms = hPolytopes_[i].leftCols<3>().rowwise().norm();
                hPolytopes_[i].array().colwise() /= norms;
            }
            if (!processCorridor(hPolytopes_, vPolytopes_)) return false;

            polyN_ = hPolytopes_.size();
            magnitudeBd_ = magnitudeBounds; penaltyWt_ = penaltyWeights;
            
            allocSpeed_ = magnitudeBd_(0) * 3.0; 

            getShortestPath(headState_.col(0), tailState_.col(0), vPolytopes_, 0.1, shortPath_);
            const Eigen::Matrix3Xd deltas = shortPath_.rightCols(polyN_) - shortPath_.leftCols(polyN_);
            pieceIdx_ = (deltas.colwise().norm() / lengthPerPiece).cast<int>().transpose();
            pieceIdx_.array() += 1; pieceN_ = pieceIdx_.sum();

            vPolyIdx_.resize(pieceN_ - 1); hPolyIdx_.resize(pieceN_);
            for (int i = 0, j = 0, k; i < polyN_; i++) {
                k = pieceIdx_(i);
                for (int l = 0; l < k; l++, j++) {
                    if (l < k - 1) vPolyIdx_(j) = 2 * i; else if (i < polyN_ - 1) vPolyIdx_(j) = 2 * i + 1;
                    hPolyIdx_(j) = i;
                }
            }

            spatial_map_.reset(&vPolytopes_, &vPolyIdx_, pieceN_);
            optimizer_.setSpatialMap(&spatial_map_);
            optimizer_.setEnergyWeights(1.0); 

            Eigen::MatrixXd waypoints(pieceN_ + 1, 3);
            waypoints.row(0) = headState_.col(0).transpose();
            Eigen::Matrix3Xd innerPoints; Eigen::VectorXd timeAlloc;
            setInitial(shortPath_, allocSpeed_, pieceIdx_, innerPoints, timeAlloc);
            for (int i = 0; i < innerPoints.cols(); ++i) waypoints.row(i + 1) = innerPoints.col(i).transpose();
            waypoints.row(pieceN_) = tailState_.col(0).transpose();

            if (!optimizer_.setInitState(std::vector<double>(timeAlloc.data(), timeAlloc.data() + timeAlloc.size()), waypoints, headState_, tailState_)) return false;

            time_cost_.weight = timeWeight;
            // 恢复真实的物理上限（不再放宽）
            cp_cost_.max_v = magnitudeBd_(0) * 1.2;
            cp_cost_.max_thrust = magnitudeBd_(1) * 1.2;
            cp_cost_.weight_pos = penaltyWt_.size() > 0 ? penaltyWt_(0) : 10000.0;
            cp_cost_.weight_v = penaltyWt_.size() > 1 ? penaltyWt_(1) : 10000.0;
            cp_cost_.weight_thrust = penaltyWt_.size() > 4 ? penaltyWt_(4) : 100000.0;
            cp_cost_.hPolys = &hPolytopes_;
            
            int p = optimizer_.getTrajectory().getP();
            cp_cost_.spline_degree = p;
            std::vector<std::vector<int>> piece_mapping(pieceN_);
            for (int piece = 0; piece < pieceN_; ++piece)
            {
                auto add_unique = [&](int poly_idx)
                {
                    if (poly_idx < 0)
                    {
                        return;
                    }
                    if (std::find(piece_mapping[piece].begin(),
                                  piece_mapping[piece].end(),
                                  poly_idx) == piece_mapping[piece].end())
                    {
                        piece_mapping[piece].push_back(poly_idx);
                    }
                };

                add_unique(hPolyIdx_(piece));
                for (int offset = 1; offset <= p; ++offset)
                {
                    if (piece - offset >= 0)
                    {
                        add_unique(hPolyIdx_(piece - offset));
                    }
                    if (piece + offset < pieceN_)
                    {
                        add_unique(hPolyIdx_(piece + offset));
                    }
                }
            }
            cp_cost_.piece_to_polys = piece_mapping;
            cp_cost_.cp_to_polys.clear();
            return true;
        }

        double optimize(TrajType &spline)
        {
            Eigen::VectorXd x0 = optimizer_.generateInitialGuess();
            const Eigen::RowVectorXd theta0 = evaluatorToDecision(x0);

            Eigen::VectorXd lb;
            Eigen::VectorXd ub;
            buildDecisionBounds(theta0, lb, ub);

            if (backend_ == SolverBackend::ABC)
            {
                return optimizeWithABC(theta0, lb, ub, spline);
            }
            return optimizeWithDPASA(theta0, lb, ub, spline);
        }
    };
} 
#endif
