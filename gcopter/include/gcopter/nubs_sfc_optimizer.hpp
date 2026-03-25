#ifndef NUBS_SFC_OPTIMIZER_HPP
#define NUBS_SFC_OPTIMIZER_HPP

#include "NUBSTrajectory/NUBSOptimizer.hpp" 
#include "TrajectoryOptComponents/SFCCommonTypes.hpp"
#include "gcopter/geo_utils.hpp"
#include "gcopter/lbfgs.hpp"

#include <Eigen/Eigen>

#include <cmath>
#include <cstdlib>
#include <cfloat>
#include <vector>

namespace gcopter
{
    struct NUBSTimeCost
    {
        double weight = 1.0;

        double operator()(const std::vector<double> &T, Eigen::VectorXd &gdT) const
        {
            double cost = 0.0;
            gdT.setZero(T.size());
            for (size_t i = 0; i < T.size(); ++i)
            {
                cost += weight * T[i];
                gdT(i) = weight;
            }
            return cost;
        }
    };

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

        // 平滑 L1 惩罚：大数值时呈现线性，防止梯度爆炸
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

    class NUBSSFCOptimizer
    {
    public:
        using TrajType = nubs::NUBSTrajectory<3, 7>;
        using OptimizerType = nubs::NUBSOptimizer<3, 7,
                                                  nubs::QuadInvTimeMap,
                                                  gcopter::PolytopeSpatialMap>;

        typedef Eigen::Matrix3Xd PolyhedronV;
        typedef Eigen::MatrixX4d PolyhedronH;
        typedef std::vector<PolyhedronV> PolyhedraV;
        typedef std::vector<PolyhedronH> PolyhedraH;

    private:
        OptimizerType optimizer_;
        gcopter::PolytopeSpatialMap spatial_map_;
        NUBSTimeCost time_cost_;
        SFCControlPointCost cp_cost_;

        Eigen::Matrix3d headState_; // [Pos, Vel, Acc]
        Eigen::Matrix3d tailState_; // [Pos, Vel, Acc]

        PolyhedraV vPolytopes_;
        PolyhedraH hPolytopes_;
        Eigen::Matrix3Xd shortPath_;

        Eigen::VectorXi pieceIdx_;
        Eigen::VectorXi vPolyIdx_;
        Eigen::VectorXi hPolyIdx_;

        int polyN_ = 0;
        int pieceN_ = 0;

        double smoothEps_ = 0.0;
        Eigen::VectorXd magnitudeBd_;
        Eigen::VectorXd penaltyWt_;
        double allocSpeed_ = 0.0;

        lbfgs::lbfgs_parameter_t lbfgs_params_{};

        std::vector<double> ref_times_;
        Eigen::MatrixXd ref_waypoints_;

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
                                  &NUBSSFCOptimizer::costDistance,
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

        /**
         * @brief L-BFGS 调用的静态目标函数回调
         */
        static inline double costFunctional(void *ptr,
                                            const Eigen::VectorXd &x,
                                            Eigen::VectorXd &g)
        {
            auto &obj = *(NUBSSFCOptimizer *)ptr;
            // 核心调用：将 x 映射、求轨、算物理代价、反传梯
            double cost = obj.optimizer_.evaluate(x, g, obj.time_cost_, obj.cp_cost_);

            if (!std::isfinite(cost) || !g.allFinite())
            {
                g.setZero();
                return 1.0e20;
            }
            
            return cost;
        }

    public:
        /**
         * @brief 初始化环境、提取初值、配置优化器参数
         */
        bool setup(const double &timeWeight,
                   const Eigen::Matrix3d &initialPVA,
                   const Eigen::Matrix3d &terminalPVA,
                   const PolyhedraH &safeCorridor,
                   const double &lengthPerPiece,
                   const double &smoothingFactor,
                   const Eigen::VectorXd &magnitudeBounds,
                   const Eigen::VectorXd &penaltyWeights,
                   const Eigen::VectorXd &physicalParams) 
        {
                       headState_ = initialPVA;
            tailState_ = terminalPVA;

            hPolytopes_ = safeCorridor;
            for (size_t i = 0; i < hPolytopes_.size(); i++)
            {
                const Eigen::ArrayXd norms = hPolytopes_[i].leftCols<3>().rowwise().norm();
                hPolytopes_[i].array().colwise() /= norms;
            }
            if (!processCorridor(hPolytopes_, vPolytopes_)) return false;

            polyN_ = hPolytopes_.size();
            smoothEps_ = smoothingFactor;
            magnitudeBd_ = magnitudeBounds; // [max_vel, max_thrust, ...]
            penaltyWt_ = penaltyWeights;    // [wt_vel, wt_thrust, ...]
            allocSpeed_ = magnitudeBd_(0) * 3.0; 

    
            getShortestPath(headState_.col(0), tailState_.col(0), vPolytopes_, smoothEps_, shortPath_);
            
            const Eigen::Matrix3Xd deltas = shortPath_.rightCols(polyN_) - shortPath_.leftCols(polyN_);
            pieceIdx_ = (deltas.colwise().norm() / lengthPerPiece).cast<int>().transpose();
            pieceIdx_.array() += 1;
            pieceN_ = pieceIdx_.sum();

            vPolyIdx_.resize(pieceN_ - 1);
            hPolyIdx_.resize(pieceN_);
            for (int i = 0, j = 0, k; i < polyN_; i++) {
                k = pieceIdx_(i);
                for (int l = 0; l < k; l++, j++) {
                    if (l < k - 1) vPolyIdx_(j) = 2 * i;
                    else if (i < polyN_ - 1) vPolyIdx_(j) = 2 * i + 1;
                    hPolyIdx_(j) = i;
                }
            }

            spatial_map_.reset(&vPolytopes_, &vPolyIdx_, pieceN_);
            optimizer_.setSpatialMap(spatial_map_);
            
            // 设置底层轨迹的能量代价权重 (例如 minimum jerk 的权重)
            optimizer_.setEnergyWeights(1.0); 

            // 4. 初值提取与参数配置
            Eigen::MatrixXd waypoints(pieceN_ + 1, 3);
            waypoints.row(0) = headState_.col(0).transpose();

            Eigen::Matrix3Xd innerPoints;
            Eigen::VectorXd timeAlloc;
            setInitial(shortPath_, allocSpeed_, pieceIdx_, innerPoints, timeAlloc);
            for (int i = 0; i < innerPoints.cols(); ++i) {
                waypoints.row(i + 1) = innerPoints.col(i).transpose();
            }
            waypoints.row(pieceN_) = tailState_.col(0).transpose();

            // 注入到优化器中
            if (!optimizer_.setInitState(std::vector<double>(timeAlloc.data(), timeAlloc.data() + timeAlloc.size()),
                                         waypoints,
                                         headState_,
                                         tailState_))
            {
                return false;
            }

            time_cost_.weight = timeWeight;
            
            cp_cost_.max_v = magnitudeBd_(0) * 1.2;
            cp_cost_.max_thrust = magnitudeBd_(1) * 1.2 ;
            cp_cost_.weight_v = penaltyWt_(0);
            cp_cost_.weight_thrust = penaltyWt_(1);
            cp_cost_.weight_pos = penaltyWt_.size() > 2 ? penaltyWt_(2) : 10000.0; 
            cp_cost_.smooth_eps = smoothEps_;
            cp_cost_.hPolys = &hPolytopes_;
            
            int p = optimizer_.getTrajectory().getP();
            int nc = optimizer_.getTrajectory().getCtrlPtNum(pieceN_);
            std::vector<std::vector<int>> mapping(nc);
            
            for (int j = 0; j < nc; ++j) 
            {
                int primary_piece = j - p / 2;
                
                primary_piece = std::max(0, std::min(pieceN_ - 1, primary_piece));
                
                mapping[j].push_back(hPolyIdx_(primary_piece));
            }
            cp_cost_.cp_to_polys = mapping;

            return true;
        }

        /**
         * @brief 执行主优化逻辑
         */
        double optimize(TrajType &spline, const double &relCostTol)
        {
            Eigen::VectorXd x = optimizer_.generateInitialGuess();

            double minCostFunctional = 0.0;
            lbfgs_params_.mem_size = 256;
            lbfgs_params_.past = 3;
            lbfgs_params_.min_step = 1.0e-32;
            lbfgs_params_.g_epsilon = 0.0;
            lbfgs_params_.delta = relCostTol;

            const char *grad_check_env = std::getenv("GCOPTER_GRAD_CHECK");
            if (grad_check_env && std::string(grad_check_env) == "1")
            {
                struct ZeroTimeCost {
                    double operator()(const std::vector<double> &/*Ts*/, Eigen::VectorXd &grad) const {
                        grad.setZero();
                        return 0.0;
                    }
                };

                auto zero_cp = [](const Eigen::MatrixXd &/*C*/,const Eigen::MatrixXd &/*V*/, const Eigen::MatrixXd &/*A*/, const Eigen::MatrixXd &/*J*/,
                                  double &cost, Eigen::MatrixXd &/*gdC*/, Eigen::MatrixXd &/*gdV*/, Eigen::MatrixXd &/*gdA*/, Eigen::MatrixXd &/*gdJ*/)
                {
                    cost = 0.0;
                };

                auto run_check = [&](const char *tag,
                                     auto &&time_func,
                                     auto &&cp_func,
                                     double energy_weight)
                {
                    optimizer_.setEnergyWeights(energy_weight);
                    auto check = optimizer_.checkGradients(
                        x,
                        std::forward<decltype(time_func)>(time_func),
                        std::forward<decltype(cp_func)>(cp_func));
                        
                    std::cerr << "[GradCheck] " << tag << " -> " << check.makeReport();
                    std::cerr << "[GradCheck] " << tag << " rel error: " << check.rel_error
                              << " | norm: " << check.error_norm << std::endl;
                              
                    if (check.analytical.size() == check.numerical.size() && check.analytical.size() > 0)
                    {
                        const int time_dim = pieceN_;
                        const int total_dim = static_cast<int>(check.analytical.size());
                        const int spatial_dim = std::max(0, total_dim - time_dim);

                        if (time_dim > 0 && total_dim >= time_dim) {
                            Eigen::VectorXd diff = check.analytical - check.numerical;
                            double time_err = diff.head(time_dim).norm();
                            double time_norm = check.analytical.head(time_dim).norm();
                            double time_rel = (time_norm > 1e-9) ? (time_err / time_norm) : time_err;
                            std::cerr << "[GradCheck] " << tag << " time rel: " << time_rel << " | time norm: " << time_err << std::endl;
                        }

                        if (spatial_dim > 0) {
                            Eigen::VectorXd diff = check.analytical - check.numerical;
                            double spatial_err = diff.tail(spatial_dim).norm();
                            double spatial_norm = check.analytical.tail(spatial_dim).norm();
                            double spatial_rel = (spatial_norm > 1e-9) ? (spatial_err / spatial_norm) : spatial_err;
                            std::cerr << "[GradCheck] " << tag << " spatial rel: " << spatial_rel << " | spatial norm: " << spatial_err << std::endl;
                        }
                    }
                    std::cerr.flush();
                };

                run_check("ALL (time+cp+energy)", time_cost_, cp_cost_, 1.0);
                run_check("TIME ONLY", time_cost_, zero_cp, 0.0);
                run_check("CP ONLY (Penalty)", ZeroTimeCost(), cp_cost_, 0.0);
                run_check("ENERGY ONLY (Min-Jerk)", ZeroTimeCost(), zero_cp, 1.0);

                optimizer_.setEnergyWeights(1.0);
            }

            const int ret = lbfgs::lbfgs_optimize(x,
                                                  minCostFunctional,
                                                  &NUBSSFCOptimizer::costFunctional,
                                                  nullptr,
                                                  nullptr,
                                                  this,
                                                  lbfgs_params_);

            if (ret >= 0)
            {
                Eigen::VectorXd grad(x.size());
                // 确保在最优参数处，提取一次最终轨迹
                minCostFunctional = optimizer_.evaluate(x, grad, time_cost_, cp_cost_);
                spline = optimizer_.getTrajectory();
            }
            else
            {
                spline = TrajType();
                minCostFunctional = INFINITY;
                std::cout << "NUBS Optimization Failed: " << lbfgs::lbfgs_strerror(ret) << std::endl;
            }

            return minCostFunctional;
        }
    };

} // namespace gcopter

#endif // NUBS_SFC_OPTIMIZER_HPP