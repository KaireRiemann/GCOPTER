#ifndef SPLINE_TRAJECTORY_POLYTOPE_SPATIAL_MAP_HPP
#define SPLINE_TRAJECTORY_POLYTOPE_SPATIAL_MAP_HPP

#include "TrajectoryOptComponents/SFCCommonTypes.hpp"
#include "gcopter/lbfgs.hpp"

#include <cmath>
#include <cfloat>

namespace gcopter
{
struct PolytopeSpatialMap
{
    using VectorType = Eigen::Vector3d;

    const PolyhedraV *v_polys = nullptr;
    const Eigen::VectorXi *v_poly_idx = nullptr;
    int num_segments = 0;

    void reset(const PolyhedraV *polys, const Eigen::VectorXi *indices, int segments)
    {
        v_polys = polys;
        v_poly_idx = indices;
        num_segments = segments;
    }

    int getUnconstrainedDim(int index) const
    {
        if (!v_polys || !v_poly_idx || index <= 0 || index >= num_segments)
            return 3;
        const int poly_id = (*v_poly_idx)(index - 1);
        return (*v_polys)[poly_id].cols();
    }

    VectorType toPhysical(const Eigen::VectorXd &xi, int index) const
    {
        if (!v_polys || !v_poly_idx || index <= 0 || index >= num_segments)
            return xi.head<3>();

        const int poly_id = (*v_poly_idx)(index - 1);
        const PolyhedronV &poly = (*v_polys)[poly_id];
        const int k = poly.cols();

        const double norm = xi.norm();
        if (norm < 1e-12)
            return poly.col(0);

        const Eigen::VectorXd unit_xi = xi / norm;
        const Eigen::VectorXd r = unit_xi.head(k - 1);
        return poly.rightCols(k - 1) * r.cwiseProduct(r) + poly.col(0);
    }

    Eigen::VectorXd toUnconstrained(const Eigen::VectorXd &p, int index) const
    {
        if (!v_polys || !v_poly_idx || index <= 0 || index >= num_segments)
            return p;

        const int poly_id = (*v_poly_idx)(index - 1);
        const PolyhedronV &poly = (*v_polys)[poly_id];
        const int k = poly.cols();

        Eigen::Matrix3Xd ov_poly(3, k + 1);
        ov_poly.col(0) = p;
        ov_poly.rightCols(k) = poly;

        Eigen::VectorXd xi(k);
        xi.setConstant(std::sqrt(1.0 / static_cast<double>(k)));

        lbfgs::lbfgs_parameter_t params;
        params.past = 0;
        params.delta = 1.0e-5;
        params.g_epsilon = FLT_EPSILON;
        params.max_iterations = 128;

        double min_cost = 0.0;
        lbfgs::lbfgs_optimize(xi, min_cost, &PolytopeSpatialMap::costTinyNLS,
                              nullptr, nullptr, &ov_poly, params);

        return xi;
    }

    Eigen::VectorXd backwardGrad(const Eigen::VectorXd &xi,
                                 const Eigen::VectorXd &grad_p,
                                 int index) const
    {
        if (!v_polys || !v_poly_idx || index <= 0 || index >= num_segments)
            return grad_p;

        const int poly_id = (*v_poly_idx)(index - 1);
        const PolyhedronV &poly = (*v_polys)[poly_id];
        const int k = poly.cols();

        Eigen::VectorXd grad_xi = Eigen::VectorXd::Zero(k);
        const double norm = xi.norm();
        if (norm < 1e-12)
            return grad_xi;

        const double norm_inv = 1.0 / norm;
        const Eigen::VectorXd unit_xi = xi * norm_inv;
        Eigen::VectorXd grad_q(k);
        grad_q.head(k - 1) = (poly.rightCols(k - 1).transpose() * grad_p).array() *
                             unit_xi.head(k - 1).array() * 2.0;
        grad_q(k - 1) = 0.0;
        grad_xi = (grad_q - unit_xi * unit_xi.dot(grad_q)) * norm_inv;
        return grad_xi;
    }

    void addNormPenalty(const Eigen::VectorXd& xi, double& cost, Eigen::VectorXd& grad_xi) const 
    {
        double sqrNormViolation = xi.squaredNorm() - 1.0;
        if (sqrNormViolation > 0.0) {
            double c = sqrNormViolation * sqrNormViolation;
            cost += c * sqrNormViolation; 
            grad_xi += (3.0 * c) * 2.0 * xi;
        }
    }

private:
    static inline double costTinyNLS(void *ptr,
                                     const Eigen::VectorXd &xi,
                                     Eigen::VectorXd &gradXi)
    {
        const int n = xi.size();
        const Eigen::Matrix3Xd &ov_poly = *(Eigen::Matrix3Xd *)ptr;

        const double sqrNormXi = xi.squaredNorm();
        const double invNormXi = 1.0 / std::sqrt(sqrNormXi);
        const Eigen::VectorXd unitXi = xi * invNormXi;
        const Eigen::VectorXd r = unitXi.head(n - 1);
        const Eigen::Vector3d delta = ov_poly.rightCols(n - 1) * r.cwiseProduct(r) +
                                      ov_poly.col(1) - ov_poly.col(0);

        double cost = delta.squaredNorm();
        gradXi.head(n - 1) = (ov_poly.rightCols(n - 1).transpose() * (2 * delta)).array() *
                             r.array() * 2.0;
        gradXi(n - 1) = 0.0;
        gradXi = (gradXi - unitXi.dot(gradXi) * unitXi).eval() * invNormXi;

        const double sqrNormViolation = sqrNormXi - 1.0;
        if (sqrNormViolation > 0.0)
        {
            double c = sqrNormViolation * sqrNormViolation;
            const double dc = 3.0 * c;
            c *= sqrNormViolation;
            cost += c;
            gradXi += dc * 2.0 * xi;
        }

        return cost;
    }
};

struct PolytopeSpatialMapZO
{
    using VectorType = Eigen::Vector3d;

    const PolyhedraV *v_polys = nullptr;
    const Eigen::VectorXi *v_poly_idx = nullptr;
    int num_segments = 0;

    void reset(const PolyhedraV *polys, const Eigen::VectorXi *indices, int segments)
    {
        v_polys = polys;
        v_poly_idx = indices;
        num_segments = segments;
    }

    inline int getUnconstrainedDim(int index) const
    {
        if (!v_polys || !v_poly_idx || index <= 0 || index >= num_segments) return 3;
        const int poly_id = (*v_poly_idx)(index - 1);
        return (*v_polys)[poly_id].cols();
    }

    inline VectorType toPhysical(const Eigen::VectorXd &xi, int index) const
    {
        if (!v_polys || !v_poly_idx || index <= 0 || index >= num_segments) return xi.head<3>();

        const int poly_id = (*v_poly_idx)(index - 1);
        const PolyhedronV &poly = (*v_polys)[poly_id];
        const int k = poly.cols();
        double norm = xi.norm(); 
        if (norm < 1e-12) 
        {
            return poly.col(0);
        }

        const Eigen::VectorXd unit_xi = xi / norm;
        const Eigen::VectorXd r = unit_xi.head(k - 1);
        return poly.rightCols(k - 1) * r.cwiseProduct(r) + poly.col(0);
    }

    // ========== 将 costTinyNLS 设为 public ==========
    static inline double costTinyNLS(void *ptr, const Eigen::VectorXd &xi, Eigen::VectorXd &gradXi)
    {
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
        if (sqrNormViolation > 0.0)
        {
            double c = sqrNormViolation * sqrNormViolation;
            const double dc = 3.0 * c;
            c *= sqrNormViolation;
            cost += c;
            gradXi += dc * 2.0 * xi;
        }
        return cost;
    }

    Eigen::VectorXd toUnconstrained(const Eigen::VectorXd &p, int index) const
    {
        if (!v_polys || !v_poly_idx || index <= 0 || index >= num_segments) return p;

        const int poly_id = (*v_poly_idx)(index - 1);
        const PolyhedronV &poly = (*v_polys)[poly_id];
        const int k = poly.cols();

        Eigen::Matrix3Xd ov_poly(3, k + 1);
        ov_poly.col(0) = p;
        ov_poly.rightCols(k) = poly;

        Eigen::VectorXd xi(k);
        xi.setConstant(std::sqrt(1.0 / static_cast<double>(k)));

        lbfgs::lbfgs_parameter_t params;
        params.past = 0;
        params.delta = 1.0e-5;
        params.g_epsilon = FLT_EPSILON;
        params.max_iterations = 128;

        double min_cost = 0.0;
        // ==== 核心修复：匹配 gcopter Eigen 版 L-BFGS 签名 ====
        lbfgs::lbfgs_optimize(xi, min_cost, &PolytopeSpatialMapZO::costTinyNLS, nullptr, nullptr, &ov_poly, params);

        return xi;
    }

    // 零阶特有接口：直接返回标量惩罚
    inline double getNormPenalty(const Eigen::VectorXd& xi) const 
    {
        double cost = 0.0;
        double sqrNormViolation = xi.squaredNorm() - 1.0;
        if (sqrNormViolation > 0.0) {
            double c = sqrNormViolation * sqrNormViolation;
            cost = c * sqrNormViolation; 
        }
        return cost;
    }
};
}

#endif
