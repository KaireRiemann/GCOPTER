#ifndef POLYTOPE_SPATIAL_MAP_ZO_HPP
#define POLYTOPE_SPATIAL_MAP_ZO_HPP

#include "TrajectoryOptComponents/SFCCommonTypes.hpp"
#include "gcopter/lbfgs.hpp" 
#include <cmath>
#include <cfloat>

namespace gcopter 
{
struct PolytopeSpatialMapZO
{
    using VectorType = Eigen::Vector3d;

    const PolyhedraV *v_polys = nullptr;
    const Eigen::VectorXi *v_poly_idx = nullptr;
    int num_segments = 0;

    void reset(const PolyhedraV *polys, const Eigen::VectorXi *indices, int segments) {
        v_polys = polys; v_poly_idx = indices; num_segments = segments;
    }

    inline int getUnconstrainedDim(int index) const {
        if (!v_polys || !v_poly_idx || index <= 0 || index >= num_segments) return 3;
        const int poly_id = (*v_poly_idx)(index - 1);
        return (*v_polys)[poly_id].cols();
    }
    
    inline VectorType toPhysical(Eigen::VectorXd xi, int index) const 
    {
        if (!v_polys || !v_poly_idx || index <= 0 || index >= num_segments) return xi.head<3>();

        const int poly_id = (*v_poly_idx)(index - 1);
        const PolyhedronV &poly = (*v_polys)[poly_id];
        const int k = poly.cols();
        
        double norm = xi.norm();
        if (norm > 1.0) {
            xi = xi / norm; // 强行拉回边界
            norm = 1.0;
        } else if (norm < 1e-12) {
            return poly.col(0);
        }

        const Eigen::VectorXd unit_xi = xi / norm;
        const Eigen::VectorXd r = unit_xi.head(k - 1);
        return poly.rightCols(k - 1) * r.cwiseProduct(r) * (norm * norm) + poly.col(0);
    }

    static inline double costTinyNLS(void *ptr, const Eigen::VectorXd &xi, Eigen::VectorXd &gradXi) {
        const Eigen::Matrix3Xd &ov_poly = *(Eigen::Matrix3Xd *)ptr;
        const int k = xi.size();
        const double sqrNormXi = xi.squaredNorm();
        const double invNormXi = 1.0 / std::sqrt(sqrNormXi);
        const Eigen::VectorXd unitXi = xi * invNormXi;
        const Eigen::VectorXd r = unitXi.head(k - 1);
        const Eigen::Vector3d delta = ov_poly.rightCols(k - 1) * r.cwiseProduct(r) + ov_poly.col(1) - ov_poly.col(0);

        double cost = delta.squaredNorm();
        gradXi.head(k - 1) = (ov_poly.rightCols(k - 1).transpose() * (2 * delta)).array() * r.array() * 2.0;
        gradXi(k - 1) = 0.0;
        gradXi = (gradXi - unitXi.dot(gradXi) * unitXi).eval() * invNormXi;

        const double sqrNormViolation = sqrNormXi - 1.0;
        if (sqrNormViolation > 0.0) {
            double c = sqrNormViolation * sqrNormViolation;
            cost += c * sqrNormViolation;
            gradXi += (3.0 * c) * 2.0 * xi;
        }
        return cost;
    }

    Eigen::VectorXd toUnconstrained(const Eigen::VectorXd &p, int index) const {
        if (!v_polys || !v_poly_idx || index <= 0 || index >= num_segments) return p;
        const int poly_id = (*v_poly_idx)(index - 1);
        const PolyhedronV &poly = (*v_polys)[poly_id];
        const int k = poly.cols();

        Eigen::Matrix3Xd ov_poly(3, k + 1);
        ov_poly.col(0) = p; ov_poly.rightCols(k) = poly;
        Eigen::VectorXd xi(k); xi.setConstant(std::sqrt(1.0 / static_cast<double>(k)));

        lbfgs::lbfgs_parameter_t params;
        params.past = 0; params.delta = 1.0e-5; params.g_epsilon = FLT_EPSILON; params.max_iterations = 128;
        double min_cost = 0.0;
        lbfgs::lbfgs_optimize(xi, min_cost, &PolytopeSpatialMapZO::costTinyNLS, nullptr, nullptr, &ov_poly, params);
        return xi;
    }

    inline double getNormPenalty(const Eigen::VectorXd& xi) const { return 0.0; }
};
} // namespace gcopter
#endif