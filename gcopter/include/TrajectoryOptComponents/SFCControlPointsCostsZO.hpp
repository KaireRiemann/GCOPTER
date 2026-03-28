#ifndef SFC_CONTROL_POINTS_COSTS_ZO_HPP
#define SFC_CONTROL_POINTS_COSTS_ZO_HPP

#include <cmath>
#include <limits>
#include <vector>
#include <Eigen/Eigen>

namespace gcopter
{
    struct SFCControlPointCostZO
    {
        double max_v, max_thrust;
        double weight_v, weight_thrust, weight_pos;
        Eigen::Vector3d gravity;
        
        using PolyhedraH = std::vector<Eigen::MatrixX4d>;
        const PolyhedraH* hPolys; 
        std::vector<std::vector<int>> cp_to_polys; 
        std::vector<std::vector<int>> piece_to_polys;
        int spline_degree;

        SFCControlPointCostZO() : 
            max_v(0), max_thrust(0), weight_v(0), weight_thrust(0), 
            weight_pos(10000.0), gravity(0.0, 0.0, 9.81), hPolys(nullptr), spline_degree(-1) {}
        
        inline double penaltySq(double violation) const {
            if (violation <= 0.0) return 0.0;
            double threshold = 0.5; 
            if (violation > threshold) return threshold * threshold + 2.0 * threshold * (violation - threshold);
            return violation * violation;
        }

        inline double polytopeViolation(const Eigen::Vector3d &pos,
                                        const Eigen::MatrixX4d &poly) const
        {
            return (poly.leftCols<3>() * pos + poly.col(3)).maxCoeff();
        }

        inline int inferDegree(const Eigen::MatrixXd &C) const
        {
            if (spline_degree >= 0)
            {
                return spline_degree;
            }
            if (!piece_to_polys.empty())
            {
                return std::max(0, static_cast<int>(C.rows()) - static_cast<int>(piece_to_polys.size()));
            }
            return 0;
        }

        inline double computePiecePolyCost(const Eigen::MatrixXd &C,
                                           const int cp_begin,
                                           const int cp_end,
                                           const Eigen::MatrixX4d &poly) const
        {
            double piece_cost = 0.0;
            for (int cp_idx = cp_begin; cp_idx <= cp_end; ++cp_idx)
            {
                const Eigen::Vector3d pos = C.row(cp_idx).transpose();
                const Eigen::VectorXd violas = poly.leftCols<3>() * pos + poly.col(3);
                for (int k = 0; k < violas.size(); ++k)
                {
                    piece_cost += weight_pos * penaltySq(violas(k));
                }
            }
            return piece_cost;
        }

        inline double pieceHullCost(const Eigen::MatrixXd &C) const
        {
            if (hPolys == nullptr || piece_to_polys.empty() || C.rows() <= 0)
            {
                return 0.0;
            }

            double cost = 0.0;
            const int p = inferDegree(C);
            const int num_ctrl_pts = static_cast<int>(C.rows());
            for (int piece = 0; piece < static_cast<int>(piece_to_polys.size()); ++piece)
            {
                const int cp_begin = std::min(piece, num_ctrl_pts - 1);
                const int cp_end = std::min(piece + p, num_ctrl_pts - 1);
                if (cp_begin > cp_end)
                {
                    continue;
                }

                const std::vector<int> &candidates = piece_to_polys[piece];
                double best_cost = std::numeric_limits<double>::infinity();

                if (candidates.empty())
                {
                    for (int poly_idx = 0; poly_idx < static_cast<int>(hPolys->size()); ++poly_idx)
                    {
                        best_cost = std::min(best_cost,
                                             computePiecePolyCost(C, cp_begin, cp_end, (*hPolys)[poly_idx]));
                    }
                }
                else
                {
                    for (int poly_idx : candidates)
                    {
                        if (poly_idx < 0 || poly_idx >= static_cast<int>(hPolys->size()))
                        {
                            continue;
                        }
                        best_cost = std::min(best_cost,
                                             computePiecePolyCost(C, cp_begin, cp_end, (*hPolys)[poly_idx]));
                    }
                }

                if (std::isfinite(best_cost))
                {
                    cost += best_cost;
                }
            }
            return cost;
        }

        inline double maxPieceHullViolation(const Eigen::MatrixXd &C) const
        {
            if (hPolys == nullptr || piece_to_polys.empty() || C.rows() <= 0)
            {
                return 0.0;
            }

            double max_piece_violation = -std::numeric_limits<double>::infinity();
            const int p = inferDegree(C);
            const int num_ctrl_pts = static_cast<int>(C.rows());
            for (int piece = 0; piece < static_cast<int>(piece_to_polys.size()); ++piece)
            {
                const int cp_begin = std::min(piece, num_ctrl_pts - 1);
                const int cp_end = std::min(piece + p, num_ctrl_pts - 1);
                if (cp_begin > cp_end)
                {
                    continue;
                }

                double best_common_violation = std::numeric_limits<double>::infinity();
                const std::vector<int> &candidates = piece_to_polys[piece];
                if (candidates.empty())
                {
                    for (int poly_idx = 0; poly_idx < static_cast<int>(hPolys->size()); ++poly_idx)
                    {
                        double worst_cp_violation = -std::numeric_limits<double>::infinity();
                        for (int cp_idx = cp_begin; cp_idx <= cp_end; ++cp_idx)
                        {
                            worst_cp_violation = std::max(
                                worst_cp_violation,
                                polytopeViolation(C.row(cp_idx).transpose(), (*hPolys)[poly_idx]));
                        }
                        best_common_violation = std::min(best_common_violation, worst_cp_violation);
                    }
                }
                else
                {
                    for (int poly_idx : candidates)
                    {
                        if (poly_idx < 0 || poly_idx >= static_cast<int>(hPolys->size()))
                        {
                            continue;
                        }
                        double worst_cp_violation = -std::numeric_limits<double>::infinity();
                        for (int cp_idx = cp_begin; cp_idx <= cp_end; ++cp_idx)
                        {
                            worst_cp_violation = std::max(
                                worst_cp_violation,
                                polytopeViolation(C.row(cp_idx).transpose(), (*hPolys)[poly_idx]));
                        }
                        best_common_violation = std::min(best_common_violation, worst_cp_violation);
                    }
                }

                if (std::isfinite(best_common_violation))
                {
                    max_piece_violation = std::max(max_piece_violation, best_common_violation);
                }
            }

            return std::isfinite(max_piece_violation) ? max_piece_violation : 0.0;
        }

        inline double maxSpeedViolation(const Eigen::MatrixXd &V) const
        {
            double max_violation = 0.0;
            for (int i = 0; i < V.rows(); ++i)
            {
                max_violation = std::max(max_violation, V.row(i).norm() - max_v);
            }
            return std::max(0.0, max_violation);
        }

        inline double maxThrustViolation(const Eigen::MatrixXd &A) const
        {
            double max_violation = 0.0;
            for (int i = 0; i < A.rows(); ++i)
            {
                const Eigen::Vector3d thr = A.row(i).transpose() + gravity;
                max_violation = std::max(max_violation, thr.norm() - max_thrust);
            }
            return std::max(0.0, max_violation);
        }

        inline double operator()(const Eigen::MatrixXd &C, const Eigen::MatrixXd &V, 
                                 const Eigen::MatrixXd &A, const Eigen::MatrixXd &J) const
        {
            double cost = 0.0;
            double max_v_sq = max_v * max_v;
            double max_thrust_sq = max_thrust * max_thrust;

            if (hPolys != nullptr && !piece_to_polys.empty()) {
                cost += pieceHullCost(C);
            }
            else if (hPolys != nullptr && !cp_to_polys.empty()) {
                int max_i = std::min((int)C.rows(), (int)cp_to_polys.size());
                for (int i = 0; i < max_i; ++i) {
                    Eigen::Vector3d pos = C.row(i).transpose();
                    for (int L : cp_to_polys[i]) {
                        if (L < 0 || L >= (int)hPolys->size()) continue;
                        Eigen::VectorXd violas = (*hPolys)[L].leftCols<3>() * pos + (*hPolys)[L].col(3);
                        for (int k = 0; k < violas.size(); ++k) {
                            if (violas(k) > 0.0) cost += weight_pos * penaltySq(violas(k)); 
                        }
                    }
                }
            }

            if (V.rows() > 0) {
                Eigen::VectorXd v_sqnorm = V.rowwise().squaredNorm();
                for (int i = 0; i < v_sqnorm.size(); ++i) {
                    if (v_sqnorm(i) > max_v_sq) cost += weight_v * penaltySq(std::sqrt(v_sqnorm(i)) - max_v);
                }
            }

            if (A.rows() > 0) {
                Eigen::MatrixXd A_total = A.rowwise() + gravity.transpose();
                Eigen::VectorXd a_sqnorm = A_total.rowwise().squaredNorm();
                for (int i = 0; i < a_sqnorm.size(); ++i) {
                    if (a_sqnorm(i) > max_thrust_sq) cost += weight_thrust * penaltySq(std::sqrt(a_sqnorm(i)) - max_thrust);
                }
            }
            return cost;
        }
    };
} 
#endif
