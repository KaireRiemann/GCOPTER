#ifndef SFC_CONTROL_POINTS_COSTS_HPP
#define SFC_CONTROL_POINTS_COSTS_HPP

#include<cmath>
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

        SFCControlPointCost() : gravity(0.0, 0.0, 9.81), hPolys(nullptr), weight_pos(10000.0), smooth_eps(0.1) {}
        
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
            if (hPolys != nullptr && !cp_to_polys.empty())
            {
                for (int i = 0; i < C.rows(); ++i)
                {
                    if (i >= cp_to_polys.size()) break;
                    Eigen::Vector3d pos = C.row(i).transpose();
                    
                    for (int L : cp_to_polys[i])
                    {
                        if (L < 0 || L >= hPolys->size()) continue;
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