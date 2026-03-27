#ifndef SFC_CONTROL_POINTS_COSTS_ZO_HPP
#define SFC_CONTROL_POINTS_COSTS_ZO_HPP

#include <cmath>
#include <vector>
#include <Eigen/Eigen>

namespace gcopter
{
    /**
     * @brief 基于控制点的零阶碰撞与动力学惩罚代价 
     */
    struct SFCControlPointCostZO
    {
        double max_v, max_thrust;
        double weight_v, weight_thrust;
        double weight_pos;
        Eigen::Vector3d gravity;
        
        using PolyhedraH = std::vector<Eigen::MatrixX4d>;
        const PolyhedraH* hPolys; 
        std::vector<std::vector<int>> cp_to_polys; 

        /**
         * @brief 调试专用：将总的物理代价拆分为 位置(走廊)、速度、推力 三项打印
         */
        void debugPrint(const Eigen::MatrixXd &C, const Eigen::MatrixXd &V, 
                        const Eigen::MatrixXd &A, const Eigen::MatrixXd &J) const
        {
            double pos_cost = 0.0;
            double vel_cost = 0.0;
            double acc_cost = 0.0;

            double max_v_sq = max_v * max_v;
            double max_thrust_sq = max_thrust * max_thrust;

            // 1. 走廊代价计算
            if (hPolys != nullptr && !cp_to_polys.empty()) {
                int max_i = std::min((int)C.rows(), (int)cp_to_polys.size());
                for (int i = 0; i < max_i; ++i) {
                    Eigen::Vector3d pos = C.row(i).transpose();
                    for (int L : cp_to_polys[i]) {
                        if (L < 0 || L >= (int)hPolys->size()) continue;
                        Eigen::VectorXd violas = (*hPolys)[L].leftCols<3>() * pos + (*hPolys)[L].col(3);
                        for (int k = 0; k < violas.size(); ++k) {
                            if (violas(k) > 0.0) pos_cost += weight_pos * penaltySq(violas(k));
                        }
                    }
                }
            }

            // 2. 速度代价
            if (V.rows() > 0) {
                Eigen::VectorXd v_sqnorm = V.rowwise().squaredNorm();
                for (int i = 0; i < v_sqnorm.size(); ++i) {
                    if (v_sqnorm(i) > max_v_sq) vel_cost += weight_v * penaltySq(std::sqrt(v_sqnorm(i)) - max_v);
                }
            }

            // 3. 推力代价
            if (A.rows() > 0) {
                Eigen::MatrixXd A_total = A.rowwise() + gravity.transpose();
                Eigen::VectorXd a_sqnorm = A_total.rowwise().squaredNorm();
                for (int i = 0; i < a_sqnorm.size(); ++i) {
                    if (a_sqnorm(i) > max_thrust_sq) acc_cost += weight_thrust * penaltySq(std::sqrt(a_sqnorm(i)) - max_thrust);
                }
            }

            std::cout << "    [CP Component Detail]:\n";
            std::cout << "      -> Corridor (Pos) Cost : " << pos_cost << "\n";
            std::cout << "      -> Velocity Limit Cost : " << vel_cost << "\n";
            std::cout << "      -> Thrust Limit Cost   : " << acc_cost << "\n";
        }

        SFCControlPointCostZO() : 
            max_v(0), max_thrust(0), 
            weight_v(0), weight_thrust(0), 
            weight_pos(10000.0), 
            gravity(0.0, 0.0, 9.81), 
            hPolys(nullptr) {}
        
        inline double penaltySq(double violation) const
        {
            double threshold = 0.5; 
            if (violation > threshold) {
                return threshold * threshold + 2.0 * threshold * (violation - threshold);
            }
            return violation * violation;
        }

        inline double operator()(const Eigen::MatrixXd &C,
                                 const Eigen::MatrixXd &V, 
                                 const Eigen::MatrixXd &A, 
                                 const Eigen::MatrixXd &J) const
        {
            double cost = 0.0;
            double max_v_sq = max_v * max_v;
            double max_thrust_sq = max_thrust * max_thrust;

            if (hPolys != nullptr && !cp_to_polys.empty())
            {
                int max_i = std::min((int)C.rows(), (int)cp_to_polys.size());
                for (int i = 0; i < max_i; ++i)
                {
                    Eigen::Vector3d pos = C.row(i).transpose();
                    
                    for (int L : cp_to_polys[i])
                    {
                        if (L < 0 || L >= (int)hPolys->size()) continue;
                        const auto& poly = (*hPolys)[L];
                        
                        Eigen::VectorXd violas = poly.leftCols<3>() * pos + poly.col(3);
                        
                        for (int k = 0; k < violas.size(); ++k)
                        {
                            if (violas(k) > 0.0) {
                                cost += weight_pos * penaltySq(violas(k)); 
                            }
                        }
                    }
                }
            }

            if (V.rows() > 0) 
            {
                Eigen::VectorXd v_sqnorm = V.rowwise().squaredNorm();
                for (int i = 0; i < v_sqnorm.size(); ++i)
                {
                    if (v_sqnorm(i) > max_v_sq)
                    {
                        cost += weight_v * penaltySq(std::sqrt(v_sqnorm(i)) - max_v);
                    }
                }
            }

            if (A.rows() > 0) 
            {
                Eigen::MatrixXd A_total = A.rowwise() + gravity.transpose();
                Eigen::VectorXd a_sqnorm = A_total.rowwise().squaredNorm();
                for (int i = 0; i < a_sqnorm.size(); ++i)
                {
                    if (a_sqnorm(i) > max_thrust_sq)
                    {
                        cost += weight_thrust * penaltySq(std::sqrt(a_sqnorm(i)) - max_thrust);
                    }
                }
            }

            return cost;
        }
    };

} // namespace gcopter

#endif