#ifndef SFC_CONTROL_POINTS_COSTS_ZO_HPP
#define SFC_CONTROL_POINTS_COSTS_ZO_HPP

#include <cmath>
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

        SFCControlPointCostZO() : 
            max_v(0), max_thrust(0), weight_v(0), weight_thrust(0), 
            weight_pos(10000.0), gravity(0.0, 0.0, 9.81), hPolys(nullptr) {}
        
        inline double penaltySq(double violation) const {
            if (violation <= 0.0) return 0.0;
            double threshold = 0.5; 
            if (violation > threshold) return threshold * threshold + 2.0 * threshold * (violation - threshold);
            return violation * violation;
        }

        inline double operator()(const Eigen::MatrixXd &C, const Eigen::MatrixXd &V, 
                                 const Eigen::MatrixXd &A, const Eigen::MatrixXd &J) const
        {
            double cost = 0.0;
            double max_v_sq = max_v * max_v;
            double max_thrust_sq = max_thrust * max_thrust;

            if (hPolys != nullptr && !cp_to_polys.empty()) {
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