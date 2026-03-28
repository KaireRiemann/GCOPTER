#ifndef SFC_CONTROL_POINTS_COSTS_HPP
#define SFC_CONTROL_POINTS_COSTS_HPP

#include<cmath>
#include<limits>
#include<Eigen/Eigen>
#include<TrajectoryOptComponents/SFCCommonTypes.hpp>

namespace gcopter
{
    /**
     * @brief 基于控制点的碰撞惩罚以及动力学限幅惩罚代价 
     */
    struct SFCControlPointCost
    {
        double max_v, max_thrust;
        double weight_v, weight_thrust;
        double smooth_eps;
        Eigen::Vector3d gravity;
        using PolyhedraH = std::vector<Eigen::MatrixX4d>;

        double weight_pos;
        const PolyhedraH* hPolys; 
        std::vector<std::vector<int>> cp_to_polys; 
        std::vector<std::vector<int>> piece_to_polys;
        int spline_degree;

        SFCControlPointCost()
            : max_v(0.0),
              max_thrust(0.0),
              weight_v(0.0),
              weight_thrust(0.0),
              smooth_eps(0.1),
              gravity(0.0, 0.0, 9.81),
              weight_pos(10000.0),
              hPolys(nullptr),
              spline_degree(-1)
        {
        }
        
        inline bool smoothedL1(double x, double& pena, double& penaD) const
        {
            if (x <= 0.0) return false;
            if (x > smooth_eps) {
                pena = x - smooth_eps / 2.0;
                penaD = 1.0;
            } else {
                pena = x * x / (2.0 * smooth_eps);
                penaD = x / smooth_eps;
            }
            return true;
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

        inline double piecePolyCost(const Eigen::MatrixXd &C,
                                    const int cp_begin,
                                    const int cp_end,
                                    const Eigen::MatrixX4d &poly) const
        {
            double piece_cost = 0.0;
            double pena = 0.0;
            double penaD = 0.0;
            for (int cp_idx = cp_begin; cp_idx <= cp_end; ++cp_idx)
            {
                const Eigen::Vector3d pos = C.row(cp_idx).transpose();
                for (int k = 0; k < poly.rows(); ++k)
                {
                    const Eigen::Vector3d outer_normal = poly.block<1, 3>(k, 0).transpose();
                    const double viola_pos = outer_normal.dot(pos) + poly(k, 3);
                    if (smoothedL1(viola_pos, pena, penaD))
                    {
                        piece_cost += weight_pos * pena;
                    }
                }
            }
            return piece_cost;
        }

        inline void addPiecePolyCostAndGrad(const Eigen::MatrixXd &C,
                                            const int cp_begin,
                                            const int cp_end,
                                            const Eigen::MatrixX4d &poly,
                                            double &cost,
                                            Eigen::MatrixXd &gdC) const
        {
            double pena = 0.0;
            double penaD = 0.0;
            for (int cp_idx = cp_begin; cp_idx <= cp_end; ++cp_idx)
            {
                const Eigen::Vector3d pos = C.row(cp_idx).transpose();
                for (int k = 0; k < poly.rows(); ++k)
                {
                    const Eigen::Vector3d outer_normal = poly.block<1, 3>(k, 0).transpose();
                    const double viola_pos = outer_normal.dot(pos) + poly(k, 3);
                    if (smoothedL1(viola_pos, pena, penaD))
                    {
                        cost += weight_pos * pena;
                        gdC.row(cp_idx) += weight_pos * penaD * outer_normal.transpose();
                    }
                }
            }
        }

        void operator()(const Eigen::MatrixXd &C,
                        const Eigen::MatrixXd &V, 
                        const Eigen::MatrixXd &A, 
                        const Eigen::MatrixXd &J,
                        double &cost, 
                        Eigen::MatrixXd &gdC,
                        Eigen::MatrixXd &gdV, 
                        Eigen::MatrixXd &gdA, 
                        Eigen::MatrixXd &gdJ) const
        {
            // 1. 走廊位置约束惩罚
            if (hPolys != nullptr && !piece_to_polys.empty() && C.rows() > 0)
            {
                const int p = inferDegree(C);
                const int num_ctrl_pts = C.rows();
                for (int piece = 0; piece < static_cast<int>(piece_to_polys.size()); ++piece)
                {
                    const int cp_begin = std::min(piece, num_ctrl_pts - 1);
                    const int cp_end = std::min(piece + p, num_ctrl_pts - 1);
                    if (cp_begin > cp_end)
                    {
                        continue;
                    }

                    const std::vector<int> &candidates = piece_to_polys[piece];
                    int best_poly_idx = -1;
                    double best_piece_cost = std::numeric_limits<double>::infinity();

                    if (candidates.empty())
                    {
                        for (int poly_idx = 0; poly_idx < static_cast<int>(hPolys->size()); ++poly_idx)
                        {
                            const double poly_cost = piecePolyCost(C, cp_begin, cp_end, (*hPolys)[poly_idx]);
                            if (poly_cost < best_piece_cost)
                            {
                                best_piece_cost = poly_cost;
                                best_poly_idx = poly_idx;
                            }
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
                            const double poly_cost = piecePolyCost(C, cp_begin, cp_end, (*hPolys)[poly_idx]);
                            if (poly_cost < best_piece_cost)
                            {
                                best_piece_cost = poly_cost;
                                best_poly_idx = poly_idx;
                            }
                        }
                    }

                    if (best_poly_idx >= 0)
                    {
                        addPiecePolyCostAndGrad(C, cp_begin, cp_end, (*hPolys)[best_poly_idx], cost, gdC);
                    }
                }
            }
            else if (hPolys != nullptr && !cp_to_polys.empty())
            {
                for (int i = 0; i < C.rows(); ++i)
                {
                    if (i >= static_cast<int>(cp_to_polys.size())) break;
                    Eigen::Vector3d pos = C.row(i).transpose();
                    
                    for (int L : cp_to_polys[i])
                    {
                        if (L < 0 || L >= static_cast<int>(hPolys->size())) continue;
                        const auto& poly = (*hPolys)[L];
                        int K = poly.rows();
                        
                        for (int k = 0; k < K; ++k)
                        {
                            Eigen::Vector3d outerNormal = poly.block<1, 3>(k, 0).transpose();
                            double d = poly(k, 3);
                            double violaPos = outerNormal.dot(pos) + d;
                            
                            double pena = 0.0, penaD = 0.0;
                            if (smoothedL1(violaPos, pena, penaD))
                            {
                                cost += weight_pos * pena; 
                                gdC.row(i) += weight_pos * penaD * outerNormal.transpose();
                            }
                        }
                    }
                }
            }

            // 2. 速度限幅惩罚
            for (int i = 0; i < V.rows(); ++i)
            {
                double v_norm = V.row(i).norm();
                if (v_norm > max_v)
                {
                    double diff = v_norm - max_v;
                    double pena = 0.0, penaD = 0.0;
                    if (smoothedL1(diff, pena, penaD))
                    {
                        cost += weight_v * pena; 
                        gdV.row(i) += weight_v * penaD * (V.row(i) / v_norm);
                    }
                }
            }

            // 3. 推力限幅惩罚
            for (int i = 0; i < A.rows(); ++i)
            {
                Eigen::Vector3d thr = A.row(i).transpose() + gravity;
                double t_norm = thr.norm();
                if (t_norm > max_thrust)
                {
                    double diff = t_norm - max_thrust;
                    double pena = 0.0, penaD = 0.0;
                    if (smoothedL1(diff, pena, penaD))
                    {
                        cost += weight_thrust * pena;
                        gdA.row(i) += weight_thrust * penaD * (thr.transpose() / t_norm);
                    }
                }
            }
        }
    };

}//namespace gcopter


#endif
