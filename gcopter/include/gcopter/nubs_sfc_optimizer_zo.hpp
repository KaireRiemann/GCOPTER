#ifndef NUBS_SFC_OPTIMIZER_ZO_HPP
#define NUBS_SFC_OPTIMIZER_ZO_HPP

#include "NUBSTrajectory/NUBSOptimizerZO.hpp" 
#include "TrajectoryOptComponents/LinearTimeCost.hpp"
#include "TrajectoryOptComponents/SFCControlPointsCostsZO.hpp"
#include "TrajectoryOptComponents/PolytopeSpatialMapZO.hpp"
#include "gcopter/abc_solver.hpp" 
#include "gcopter/geo_utils.hpp"
#include "gcopter/lbfgs.hpp" 

#include <Eigen/Eigen>
#include <cmath>
#include <cstdlib>
#include <vector>
#include <iostream>

namespace gcopter
{
    class NUBSSFCOptimizerZO
    {
    public:
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

    private:
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
            cp_cost_.weight_v = penaltyWt_(0);
            cp_cost_.weight_thrust = penaltyWt_(1);
            cp_cost_.weight_pos = penaltyWt_.size() > 2 ? penaltyWt_(2) : 10000.0; 
            cp_cost_.hPolys = &hPolytopes_;
            
            int p = optimizer_.getTrajectory().getP();
            int nc = optimizer_.getTrajectory().getCtrlPtNum(pieceN_);
            std::vector<std::vector<int>> mapping(nc);
            for (int j = 0; j < nc; ++j) {
                int primary_piece = std::max(0, std::min(pieceN_ - 1, j - p / 2));
                mapping[j].push_back(hPolyIdx_(primary_piece));
            }
            cp_cost_.cp_to_polys = mapping;
            return true;
        }

        double optimize(TrajType &spline)
        {
            Eigen::VectorXd x0 = optimizer_.generateInitialGuess();
            int dim = x0.size();

            Eigen::VectorXd lb(dim);
            Eigen::VectorXd ub(dim);
            
            // =========================================================
            // 核心修改 3：物理锁死 ABC 的时间缩放边界
            // =========================================================
            lb(0) = 0.5;  // 允许时间最多压缩到 50%
            ub(0) = 2.0;  // 允许时间最多拉长到 200%（杜绝拉到几十秒）

            // 空间边界放宽，给平滑算法留下空间
            for(int i = 1; i < dim; ++i) { 
                lb(i) = x0(i) - 1.0; 
                ub(i) = x0(i) + 1.0;
            }

            auto objFunc = [this](const Eigen::RowVectorXd &x_row) -> std::pair<double, bool> {
                Eigen::VectorXd x = x_row.transpose();
                double cost = optimizer_.evaluate(x, time_cost_, cp_cost_);
                return {std::isfinite(cost) ? cost : 1.0e10, std::isfinite(cost) && cost < 100000.0};
            };

            ABC abc;
            // 降维后 ABC 压力骤减，只需极少的蜜蜂和代数
            int nPop = 20;   
            int MaxIt = 30;  
            int priority_count = nPop / 2;
            
            abc.initializeWithPriority(nPop, MaxIt, ub, lb, dim, priority_count, 0.1, x0.transpose());
            auto [bestCost, bestPos] = abc.optimize(nPop, MaxIt, ub, lb, dim, objFunc);

            Eigen::VectorXd best_x = bestPos.transpose();
            double final_cost = optimizer_.evaluate(best_x, time_cost_, cp_cost_);
            spline = optimizer_.getTrajectory();

            return final_cost;
        }
    };
} 
#endif