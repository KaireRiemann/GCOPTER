#ifndef NUBS_SFC_OPTIMIZER_HPP
#define NUBS_SFC_OPTIMIZER_HPP

#include "NUBSTrajectory/NUBSOptimizer.hpp" 
#include "TrajectoryOptComponents/LinearTimeCost.hpp"
#include "TrajectoryOptComponents/SFCCommonTypes.hpp"
#include "TrajectoryOptComponents/SFCControlPointsCosts.hpp"
#include "TrajectoryOptComponents/IdentitySpatialMap.hpp"
#include "gcopter/geo_utils.hpp"
#include "gcopter/lbfgs.hpp"

#include <Eigen/Eigen>

#include <cmath>
#include <cstdlib>
#include <cfloat>
#include <algorithm>
#include <limits>
#include <vector>

namespace gcopter
{
    class NUBSSFCOptimizer
    {
    public:
        using TrajType = nubs::NUBSTrajectory<3, 7>;
        using OptimizerType = nubs::NUBSOptimizer<3, 7,
                                                  nubs::ScaleProfileTimeMap,
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
        std::vector<double> base_time_alloc_;
        Eigen::VectorXd base_time_vars_;
        int time_var_dim_ = 0;
        double time_weight_balance_gain_ = 1.0;

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
        bool setup(const double &timeWeight, const Eigen::Matrix3d &initialPVA, const Eigen::Matrix3d &terminalPVA,
                   const PolyhedraH &safeCorridor, const double &lengthPerPiece,
                   const Eigen::VectorXd &magnitudeBounds, const Eigen::VectorXd &penaltyWeights) 
        {
            headState_ = initialPVA; tailState_ = terminalPVA; hPolytopes_ = safeCorridor;
            for (size_t i = 0; i < hPolytopes_.size(); i++) {
                hPolytopes_[i].array().colwise() /= hPolytopes_[i].leftCols<3>().rowwise().norm().array();
            }
            if (!processCorridor(hPolytopes_, vPolytopes_)) return false;

            polyN_ = hPolytopes_.size();
            magnitudeBd_ = magnitudeBounds; penaltyWt_ = penaltyWeights;
            

            double allocSpeed = magnitudeBd_(0) * 3.0; 

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
            setInitial(shortPath_, allocSpeed, pieceIdx_, innerPoints, timeAlloc);
            for (int i = 0; i < innerPoints.cols(); ++i) waypoints.row(i + 1) = innerPoints.col(i).transpose();
            waypoints.row(pieceN_) = tailState_.col(0).transpose();

            // 缓存基准时间分配
            base_time_alloc_ = std::vector<double>(timeAlloc.data(), timeAlloc.data() + timeAlloc.size());

            if (!optimizer_.setInitState(base_time_alloc_, waypoints, headState_, tailState_)) return false;
            base_time_vars_ = optimizer_.encodeTimeVariables(base_time_alloc_);
            time_var_dim_ = base_time_vars_.size();

            // 与 MINCO / ZO 版本保持一致：总代价中显式包含总时间正则项，
            // 同时恢复速度/推力惩罚，避免目标只剩下“拉长时间压低能量/控制点罚项”。
            time_cost_.weight = timeWeight;
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

            // 当前 NUBS surrogate 的 cp/energy 项通常比线性时间项大数百到数千倍。
            // 这里用参考初值做一次量级校准，让时间项在起点附近至少进入同一数量级，
            // 否则总目标会在很大时间范围内近似单调下降，难以形成像 MINCO 那样的最优时间尺度。
            time_weight_balance_gain_ = 1.0;
            {
                const Eigen::VectorXd x_ref = optimizer_.generateInitialGuess();
                const DecisionDiagnostic ref_diag = diagnoseDecision(x_ref);
                const double raw_time_cost = std::max(ref_diag.time_cost, 1.0e-6);
                const double target_time_cost = std::max(1.0e3, 0.25 * (std::max(0.0, ref_diag.cp_cost) +
                                                                        std::max(0.0, ref_diag.energy_cost)));
                time_weight_balance_gain_ = std::clamp(target_time_cost / raw_time_cost, 1.0, 2000.0);
                time_cost_.weight = timeWeight * time_weight_balance_gain_;
            }
            return true;
        }

        // 供 L-BFGS 调用的目标函数：彻底剔除时间的联合优化
        static inline double costSpatialOnly(void *ptr, const Eigen::VectorXd &x_spatial, Eigen::VectorXd &grad_spatial)
        {
            auto *obj = static_cast<NUBSSFCOptimizer*>(ptr);
            
            // 构造全量 x 向量（将缓存的固定时间拼接在前面，欺骗底层 evaluater）
            Eigen::VectorXd x_full(obj->time_var_dim_ + x_spatial.size());
            x_full.head(obj->time_var_dim_) = obj->base_time_vars_;
            x_full.tail(x_spatial.size()) = x_spatial;

            Eigen::VectorXd grad_full(x_full.size());
            // 调用底层一阶优化器的代价与梯度计算
            double cost = obj->optimizer_.evaluate(x_full, grad_full, obj->time_cost_, obj->cp_cost_);
            
            // 截断梯度，只将空间变量的梯度返回给 L-BFGS
            grad_spatial = grad_full.tail(x_spatial.size());
            return cost;
        }

        Eigen::VectorXd buildDecisionFromScaledTimes(const Eigen::VectorXd &x_spatial,
                                                     const std::vector<double> &times,
                                                     double scale) const
        {
            std::vector<double> scaled_times = times;
            for (double &t : scaled_times)
            {
                t *= scale;
            }
            return composeDecision(scaled_times, x_spatial);
        }

        double optimize(TrajType &spline)
        {
            // 1. 获取初始猜想，分离出纯空间变量
            Eigen::VectorXd x0_full = optimizer_.generateInitialGuess();
            Eigen::VectorXd x_spatial = x0_full.tail(x0_full.size() - time_var_dim_);


            lbfgs::lbfgs_parameter_t lbfgs_params;
            lbfgs_params.mem_size = 16;
            lbfgs_params.past = 3;
            lbfgs_params.g_epsilon = 1.0e-5;
            lbfgs_params.min_step = 1.0e-32;
            lbfgs_params.delta = 1.0e-4;
            lbfgs_params.max_iterations = 400; // 在强凸地形下，通常 10 步内收敛
            
            double min_cost = 0.0;
            int ret = lbfgs::lbfgs_optimize(x_spatial, min_cost, &NUBSSFCOptimizer::costSpatialOnly, nullptr, nullptr, this, lbfgs_params);
            (void)ret;

            // 2. 固定空间变量，对全局时间尺度做一维扫描，找到更合理的联合优化初值
            Eigen::VectorXd x_full_seed = composeDecision(base_time_alloc_, x_spatial);
            DecisionDiagnostic best_diag = diagnoseDecision(x_full_seed);
            constexpr int kScaleSamples = 80;
            constexpr double kScaleMin = 0.4;
            constexpr double kScaleMax = 3.0;
            for (int i = 0; i < kScaleSamples; ++i)
            {
                const double alpha = static_cast<double>(i) / static_cast<double>(kScaleSamples - 1);
                const double scale = kScaleMin + (kScaleMax - kScaleMin) * alpha;
                Eigen::VectorXd x_candidate = buildDecisionFromScaledTimes(x_spatial, base_time_alloc_, scale);
                const DecisionDiagnostic diag = diagnoseDecision(x_candidate);
                if (diag.cost_finite && (!best_diag.cost_finite || diag.total_cost < best_diag.total_cost))
                {
                    best_diag = diag;
                    x_full_seed = x_candidate;
                }
            }

            // 3. 从 scale-scan 的最佳种子出发做时空联合优化
            Eigen::VectorXd x_full_best = x_full_seed;
            double final_cost = best_diag.total_cost;
            const int full_ret = lbfgs::lbfgs_optimize(x_full_best,
                                                       final_cost,
                                                       &NUBSSFCOptimizer::costFunctional,
                                                       nullptr,
                                                       nullptr,
                                                       this,
                                                       lbfgs_params);

            if (full_ret >= 0 && std::isfinite(final_cost))
            {
                Eigen::VectorXd dummy_grad = Eigen::VectorXd::Zero(x_full_best.size());
                optimizer_.evaluate(x_full_best, dummy_grad, time_cost_, cp_cost_);
            }
            else
            {
                Eigen::VectorXd dummy_grad = Eigen::VectorXd::Zero(x_full_seed.size());
                optimizer_.evaluate(x_full_seed, dummy_grad, time_cost_, cp_cost_);
                final_cost = best_diag.total_cost;
            }

            // 获取联合优化后的轨迹
            spline = optimizer_.getTrajectory();

            // =========================================================
            // 4. 后端解析时间重缩放 (Analytical Time Rescaling)
            // =========================================================
            const auto& C = spline.getControlPoints();
            const auto& u = spline.getKnots();
            int p = spline.getP();
            int n = C.rows();

            double max_v_sq_actual = 0.0;
            double max_a_sq_actual = 0.0;

            // 计算该曲线产生的最大控制点速度和加速度
            Eigen::MatrixXd V(std::max(0, n - 1), 3);
            for (int i = 0; i < n - 1; ++i) {
                double dt = u(i + p + 1) - u(i + 1);
                if (dt > 1e-9) V.row(i) = (p / dt) * (C.row(i + 1) - C.row(i)); else V.row(i).setZero();
                max_v_sq_actual = std::max(max_v_sq_actual, V.row(i).squaredNorm());
            }

            for (int i = 0; i < n - 2; ++i) {
                double dt = u(i + p + 1) - u(i + 2);
                Eigen::Vector3d A_i = Eigen::Vector3d::Zero();
                if (dt > 1e-9) A_i = ((p - 1) / dt) * (V.row(i + 1).transpose() - V.row(i).transpose()); 
                Eigen::Vector3d total_thrust = A_i + cp_cost_.gravity;
                max_a_sq_actual = std::max(max_a_sq_actual, total_thrust.squaredNorm());
            }

            // 计算时间缩放因子
            double v_limit = magnitudeBd_(0);
            double a_limit = magnitudeBd_(1);
            
            double k_v = std::sqrt(max_v_sq_actual) / v_limit;
            double k_a = std::sqrt(std::sqrt(max_a_sq_actual) / a_limit); // 注意推力与时间的关系是平方

            // 获取确保安全的最大缩放因子
            double k_time = std::max({1.0, k_v, k_a});

            // 如果发生动力学越界，执行解析缩放
            if (k_time > 1.0) {
                const Eigen::VectorXd x_spatial_final = x_full_best.tail(x_full_best.size() - time_var_dim_);
                const Eigen::VectorXd &durations = spline.getDurations();
                std::vector<double> final_times(durations.data(), durations.data() + durations.size());
                Eigen::VectorXd x_final_full = buildDecisionFromScaledTimes(x_spatial_final, final_times, k_time);
                
                // 仅为了更新内部样条状态，调用一次 evaluate
                Eigen::VectorXd dummy_grad(x_final_full.size());
                optimizer_.evaluate(x_final_full, dummy_grad, time_cost_, cp_cost_);
                spline = optimizer_.getTrajectory();
            }

            return final_cost;
        }

        struct DecisionDiagnostic
        {
            double total_cost = std::numeric_limits<double>::quiet_NaN();
            double spatial_penalty = std::numeric_limits<double>::quiet_NaN();
            double energy_cost = std::numeric_limits<double>::quiet_NaN();
            double time_cost = std::numeric_limits<double>::quiet_NaN();
            double cp_cost = std::numeric_limits<double>::quiet_NaN();
            double grad_norm = std::numeric_limits<double>::quiet_NaN();
            double min_segment_time = std::numeric_limits<double>::quiet_NaN();
            double total_duration = std::numeric_limits<double>::quiet_NaN();
            bool cost_finite = false;
            bool grad_finite = false;
            bool knots_finite = false;
            bool control_points_finite = false;
        };

        int getTimeVariableDim() const
        {
            return time_var_dim_;
        }

        Eigen::VectorXd getFullInitialGuess() const
        {
            return optimizer_.generateInitialGuess();
        }

        Eigen::VectorXd encodeTimeDecision(const std::vector<double> &times) const
        {
            return optimizer_.encodeTimeVariables(times);
        }

        Eigen::VectorXd composeDecision(const std::vector<double> &times,
                                        const Eigen::VectorXd &spatial_vars) const
        {
            Eigen::VectorXd x(time_var_dim_ + spatial_vars.size());
            x.head(time_var_dim_) = optimizer_.encodeTimeVariables(times);
            x.tail(spatial_vars.size()) = spatial_vars;
            return x;
        }

        double evaluateDecision(const Eigen::VectorXd &x, Eigen::VectorXd &grad)
        {
            return optimizer_.evaluate(x, grad, time_cost_, cp_cost_);
        }

        DecisionDiagnostic diagnoseDecision(const Eigen::VectorXd &x)
        {
            DecisionDiagnostic diag;
            Eigen::VectorXd grad = Eigen::VectorXd::Zero(x.size());
            diag.total_cost = optimizer_.evaluate(x, grad, time_cost_, cp_cost_);
            diag.cost_finite = std::isfinite(diag.total_cost);
            diag.grad_finite = grad.allFinite();
            if (diag.grad_finite)
            {
                diag.grad_norm = grad.norm();
            }

            const auto &traj = optimizer_.getTrajectory();
            const auto &durations = traj.getDurations();
            if (durations.size() > 0 && durations.allFinite())
            {
                diag.min_segment_time = durations.minCoeff();
                diag.total_duration = durations.sum();
            }

            diag.knots_finite = traj.getKnots().allFinite();
            diag.control_points_finite = traj.getControlPoints().allFinite();
            diag.energy_cost = traj.getEnergy();

            std::vector<double> Ts(durations.data(), durations.data() + durations.size());
            Eigen::VectorXd dummy_grad_t = Eigen::VectorXd::Zero(Ts.size());
            diag.time_cost = time_cost_(Ts, dummy_grad_t);

            diag.spatial_penalty = 0.0;
            int offset = time_var_dim_;
            for (int i = 1; i < pieceN_; ++i)
            {
                const int dof = spatial_map_.getUnconstrainedDim(i);
                Eigen::VectorXd xi = x.segment(offset, dof);
                Eigen::VectorXd dummy_grad_xi = Eigen::VectorXd::Zero(dof);
                spatial_map_.addNormPenalty(xi, diag.spatial_penalty, dummy_grad_xi);
                offset += dof;
            }

            if (diag.cost_finite && std::isfinite(diag.energy_cost) &&
                std::isfinite(diag.time_cost) && std::isfinite(diag.spatial_penalty))
            {
                diag.cp_cost = diag.total_cost - diag.energy_cost - diag.time_cost - diag.spatial_penalty;
            }

            return diag;
        }

        void decisionToTrajectory(const Eigen::VectorXd &x, TrajType &spline)
        {
            Eigen::VectorXd dummy_grad = Eigen::VectorXd::Zero(x.size());
            optimizer_.evaluate(x, dummy_grad, time_cost_, cp_cost_);
            spline = optimizer_.getTrajectory();
        }

        bool trajectoryToDecision(const TrajType &traj, Eigen::VectorXd &x) const
        {
            if (traj.getPieceNum() != pieceN_)
            {
                return false;
            }

            const Eigen::VectorXd &durations = traj.getDurations();
            std::vector<double> times(durations.data(), durations.data() + durations.size());

            const int dim_T = time_var_dim_;
            int dim_P = 0;
            for (int i = 1; i < pieceN_; ++i)
            {
                dim_P += spatial_map_.getUnconstrainedDim(i);
            }

            x.resize(dim_T + dim_P);
            x.head(dim_T) = optimizer_.encodeTimeVariables(times);

            double t_cum = 0.0;
            int offset = dim_T;
            for (int i = 1; i < pieceN_; ++i)
            {
                t_cum += durations(i - 1);
                const Eigen::Vector3d waypoint = traj.evaluate(t_cum, 0);
                const int dof = spatial_map_.getUnconstrainedDim(i);
                x.segment(offset, dof) = spatial_map_.toUnconstrained(waypoint, i);
                offset += dof;
            }

            return true;
        }
    };

} // namespace gcopter

#endif // NUBS_SFC_OPTIMIZER_HPP
