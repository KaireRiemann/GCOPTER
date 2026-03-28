#ifndef GCOPTER_DPASA_SOLVER_HPP
#define GCOPTER_DPASA_SOLVER_HPP

#include <Eigen/Dense>

#include <algorithm>
#include <cmath>
#include <functional>
#include <limits>
#include <numeric>
#include <random>
#include <utility>
#include <vector>

namespace gcopter
{
    constexpr double kDpasaPi = 3.14159265358979323846;

    struct DPASABlockSpan
    {
        int offset = 0;
        int dim = 0;
    };

    struct DPASABlockLayout
    {
        DPASABlockSpan spatial;
        DPASABlockSpan scale;
        DPASABlockSpan profile;
    };

    struct DPASASolverOptions
    {
        int n_candidates = 24;
        int n_strategies = 8;
        int max_iterations = 30;
        int elite_size = 6;
        int strategy_sample_size = 4;
        int top_strategy_count = 3;
        int negation_period = 5;
        int stagnation_threshold = 7;
        double init_seed_fraction = 0.65;
        double init_spatial_radius = 0.08;
        double init_time_radius = 0.12;
        double step_scale = 0.28;
        double step_decay = 0.985;
        double noise_floor = 0.01;
        int convergence_window = 5;
        double convergence_tol = 1.0e-3;
        unsigned int seed = std::random_device{}();
    };

    struct DPASASolverResult
    {
        double best_cost = std::numeric_limits<double>::infinity();
        bool best_valid = false;
        Eigen::RowVectorXd best_position;
        std::vector<double> convergence_curve;
    };

    class DPASASolver
    {
    public:
        using Objective = std::function<std::pair<double, bool>(const Eigen::RowVectorXd &)>;

    private:
        struct Candidate
        {
            Eigen::RowVectorXd position;
            double cost = std::numeric_limits<double>::infinity();
            bool valid = false;
        };

        struct StrategyBlock
        {
            Eigen::VectorXd a;
            Eigen::VectorXd b;
            Eigen::VectorXd c;
            Eigen::VectorXd d;
        };

        struct Strategy
        {
            StrategyBlock spatial;
            StrategyBlock scale;
            StrategyBlock profile;
            double score = std::numeric_limits<double>::infinity();
            int stagnation = 0;
        };

        DPASASolverOptions options_;
        DPASABlockLayout layout_;
        Eigen::RowVectorXd lb_;
        Eigen::RowVectorXd ub_;
        Eigen::RowVectorXd range_;
        Objective objective_;
        std::mt19937 gen_;

    public:
        explicit DPASASolver(const DPASASolverOptions &options = DPASASolverOptions())
            : options_(options), gen_(options.seed)
        {
        }

        DPASASolverResult optimize(const Eigen::VectorXd &ub,
                                   const Eigen::VectorXd &lb,
                                   const DPASABlockLayout &layout,
                                   const Eigen::RowVectorXd &priority_seed,
                                   const Objective &objective)
        {
            layout_ = layout;
            objective_ = objective;
            lb_ = lb.transpose();
            ub_ = ub.transpose();
            range_ = (ub_ - lb_).cwiseMax(1.0e-8);

            std::vector<Candidate> candidates = initializeCandidates(priority_seed);
            std::vector<Strategy> strategies = initializeStrategies();
            std::vector<Candidate> elite_archive;
            Candidate best;
            best.position = priority_seed;
            best.cost = std::numeric_limits<double>::infinity();
            best.valid = false;

            for (Candidate &candidate : candidates)
            {
                evaluateCandidate(candidate);
                updateBest(candidate, best);
                updateEliteArchive(candidate, elite_archive);
            }

            DPASASolverResult result;
            result.best_position = best.position;
            result.best_cost = best.cost;
            result.best_valid = best.valid;
            result.convergence_curve.reserve(options_.max_iterations);

            double step_scale = options_.step_scale;
            std::vector<double> best_history;
            best_history.reserve(std::max(1, options_.convergence_window));
            for (int iter = 0; iter < options_.max_iterations; ++iter)
            {
                std::vector<int> ranked_indices = rankStrategies(strategies, candidates, best, elite_archive, step_scale);
                const int elite_strategy_count = std::max(1, std::min(options_.top_strategy_count, static_cast<int>(ranked_indices.size())));

                learnStrategies(strategies, ranked_indices, elite_strategy_count);

                for (Candidate &candidate : candidates)
                {
                    const Strategy &strategy = strategies[selectStrategyIndex(ranked_indices, elite_strategy_count)];
                    const Candidate &elite_ref = selectEliteReference(elite_archive, best);

                    Candidate trial;
                    trial.position = guideCandidate(candidate.position, strategy, best.position, elite_ref.position, step_scale);
                    evaluateCandidate(trial);

                    if (isBetter(trial, candidate))
                    {
                        candidate = trial;
                        updateBest(candidate, best);
                        updateEliteArchive(candidate, elite_archive);
                    }
                }

                if (options_.negation_period > 0 && ((iter + 1) % options_.negation_period == 0))
                {
                    applyNegation(strategies, candidates, best, elite_archive, step_scale);
                }

                step_scale = std::max(options_.noise_floor, step_scale * options_.step_decay);
                result.best_cost = best.cost;
                result.best_valid = best.valid;
                result.best_position = best.position;
                result.convergence_curve.push_back(best.cost);

                if (best.valid)
                {
                    best_history.push_back(best.cost);
                    if (static_cast<int>(best_history.size()) > options_.convergence_window)
                    {
                        best_history.erase(best_history.begin());
                    }

                    if (options_.convergence_window > 1 &&
                        static_cast<int>(best_history.size()) == options_.convergence_window)
                    {
                        double avg_abs_change = 0.0;
                        for (int i = 1; i < options_.convergence_window; ++i)
                        {
                            avg_abs_change += std::abs(best_history[i] - best_history[i - 1]);
                        }
                        avg_abs_change /= static_cast<double>(options_.convergence_window - 1);
                        const double scale = std::max(1.0, std::abs(best_history.back()));
                        if (avg_abs_change / scale < options_.convergence_tol)
                        {
                            break;
                        }
                    }
                }
            }

            return result;
        }

    private:
        std::vector<Candidate> initializeCandidates(const Eigen::RowVectorXd &seed)
        {
            std::vector<Candidate> candidates(options_.n_candidates);
            const int seed_count = std::max(1, static_cast<int>(std::round(options_.init_seed_fraction * options_.n_candidates)));

            for (int i = 0; i < options_.n_candidates; ++i)
            {
                candidates[i].position.resize(seed.size());
                if (i == 0)
                {
                    candidates[i].position = repair(seed);
                    continue;
                }

                if (i < seed_count)
                {
                    candidates[i].position = seed;
                    perturbBlock(candidates[i].position, layout_.spatial, options_.init_spatial_radius);
                    perturbBlock(candidates[i].position, layout_.scale, options_.init_time_radius);
                    perturbBlock(candidates[i].position, layout_.profile, options_.init_time_radius);
                    candidates[i].position = repair(candidates[i].position);
                }
                else
                {
                    candidates[i].position = randomUniform(lb_, ub_);
                }
            }

            return candidates;
        }

        std::vector<Strategy> initializeStrategies()
        {
            std::vector<Strategy> strategies(options_.n_strategies);
            for (Strategy &strategy : strategies)
            {
                strategy.spatial = randomStrategyBlock(layout_.spatial.dim, 0.35, 0.70);
                strategy.scale = randomStrategyBlock(layout_.scale.dim, 0.20, 0.55);
                strategy.profile = randomStrategyBlock(layout_.profile.dim, 0.25, 0.60);
            }
            return strategies;
        }

        StrategyBlock randomStrategyBlock(const int dim, const double amplitude, const double drift)
        {
            StrategyBlock block;
            block.a.resize(dim);
            block.b.resize(dim);
            block.c.resize(dim);
            block.d.resize(dim);

            std::uniform_real_distribution<double> unit(-1.0, 1.0);
            std::uniform_real_distribution<double> freq(0.5, 2.5);
            std::uniform_real_distribution<double> phase(-kDpasaPi, kDpasaPi);

            for (int i = 0; i < dim; ++i)
            {
                block.a(i) = amplitude * unit(gen_);
                block.b(i) = freq(gen_);
                block.c(i) = phase(gen_);
                block.d(i) = drift * unit(gen_);
            }
            return block;
        }

        void perturbBlock(Eigen::RowVectorXd &position, const DPASABlockSpan &span, const double radius)
        {
            if (span.dim <= 0)
            {
                return;
            }

            std::normal_distribution<double> normal(0.0, 1.0);
            for (int i = 0; i < span.dim; ++i)
            {
                const int idx = span.offset + i;
                position(idx) += normal(gen_) * radius * range_(idx);
            }
        }

        Eigen::RowVectorXd guideCandidate(const Eigen::RowVectorXd &x,
                                          const Strategy &strategy,
                                          const Eigen::RowVectorXd &best,
                                          const Eigen::RowVectorXd &elite,
                                          const double step_scale)
        {
            Eigen::RowVectorXd guided = x;
            applyGuidance(guided, x, best, elite, layout_.spatial, strategy.spatial, step_scale, 1.00);
            applyGuidance(guided, x, best, elite, layout_.scale, strategy.scale, step_scale, 0.65);
            applyGuidance(guided, x, best, elite, layout_.profile, strategy.profile, step_scale, 0.80);
            return repair(guided);
        }

        void applyGuidance(Eigen::RowVectorXd &out,
                           const Eigen::RowVectorXd &x,
                           const Eigen::RowVectorXd &best,
                           const Eigen::RowVectorXd &elite,
                           const DPASABlockSpan &span,
                           const StrategyBlock &strategy,
                           const double step_scale,
                           const double block_gain)
        {
            if (span.dim <= 0)
            {
                return;
            }

            Eigen::ArrayXd x_block = x.segment(span.offset, span.dim).array();
            Eigen::ArrayXd best_block = best.segment(span.offset, span.dim).array();
            Eigen::ArrayXd elite_block = elite.segment(span.offset, span.dim).array();
            Eigen::ArrayXd local_range = range_.segment(span.offset, span.dim).array();

            Eigen::ArrayXd denom = local_range.cwiseMax(1.0e-8);
            Eigen::ArrayXd normalized_best = (best_block - x_block) / denom;
            Eigen::ArrayXd normalized_elite = (elite_block - x_block) / denom;
            Eigen::ArrayXd sine_term = strategy.a.array() * (strategy.b.array() * normalized_best + strategy.c.array()).sin();
            Eigen::ArrayXd drift_term = strategy.d.array() * normalized_elite;

            std::normal_distribution<double> normal(0.0, 1.0);
            Eigen::ArrayXd noise(span.dim);
            for (int i = 0; i < span.dim; ++i)
            {
                noise(i) = normal(gen_);
            }

            const Eigen::ArrayXd update = block_gain * step_scale * local_range * (sine_term + drift_term) +
                                          0.15 * step_scale * local_range * noise;
            out.segment(span.offset, span.dim) = (x_block + update).matrix().transpose();
        }

        std::vector<int> rankStrategies(std::vector<Strategy> &strategies,
                                        const std::vector<Candidate> &candidates,
                                        const Candidate &best,
                                        const std::vector<Candidate> &elite_archive,
                                        const double step_scale)
        {
            std::vector<int> ranked_indices(strategies.size());
            std::iota(ranked_indices.begin(), ranked_indices.end(), 0);

            const int sample_count = std::max(1, std::min(options_.strategy_sample_size, static_cast<int>(candidates.size())));
            const Candidate &elite_ref = selectEliteReference(elite_archive, best);
            const double invalid_bias = 1.0e12;

            for (int i = 0; i < static_cast<int>(strategies.size()); ++i)
            {
                Strategy &strategy = strategies[i];
                double score_sum = 0.0;
                for (int k = 0; k < sample_count; ++k)
                {
                    const Candidate &base = candidates[randomIndex(static_cast<int>(candidates.size()))];
                    Candidate trial;
                    trial.position = guideCandidate(base.position, strategy, best.position, elite_ref.position, step_scale);
                    evaluateCandidate(trial);
                    const double base_score = base.cost + (base.valid ? 0.0 : invalid_bias);
                    const double trial_score = trial.cost + (trial.valid ? 0.0 : invalid_bias);
                    score_sum += (trial_score - base_score);
                }
                const double new_score = score_sum / sample_count;
                if (new_score + 1.0e-9 < strategy.score)
                {
                    strategy.stagnation = 0;
                }
                else
                {
                    strategy.stagnation += 1;
                }
                strategy.score = new_score;
            }

            std::sort(ranked_indices.begin(), ranked_indices.end(),
                      [&strategies](const int lhs, const int rhs)
                      {
                          return strategies[lhs].score < strategies[rhs].score;
                      });
            return ranked_indices;
        }

        void learnStrategies(std::vector<Strategy> &strategies,
                             const std::vector<int> &ranked_indices,
                             const int elite_strategy_count)
        {
            if (ranked_indices.empty())
            {
                return;
            }

            const Strategy best_strategy = strategies[ranked_indices.front()];
            for (int rank = elite_strategy_count; rank < static_cast<int>(ranked_indices.size()); ++rank)
            {
                Strategy &strategy = strategies[ranked_indices[rank]];
                const Strategy &teacher = strategies[ranked_indices[randomIndex(elite_strategy_count)]];
                const double lr = randomReal(0.15, 0.45);

                blendStrategyBlock(strategy.spatial, best_strategy.spatial, teacher.spatial, lr);
                blendStrategyBlock(strategy.scale, best_strategy.scale, teacher.scale, lr * 0.8);
                blendStrategyBlock(strategy.profile, best_strategy.profile, teacher.profile, lr * 0.9);
            }
        }

        void blendStrategyBlock(StrategyBlock &block,
                                const StrategyBlock &best_block,
                                const StrategyBlock &teacher_block,
                                const double lr)
        {
            if (block.a.size() == 0)
            {
                return;
            }

            std::normal_distribution<double> normal(0.0, 0.05);
            for (int i = 0; i < block.a.size(); ++i)
            {
                block.a(i) += lr * (best_block.a(i) - block.a(i)) + normal(gen_);
                block.b(i) += 0.30 * lr * (teacher_block.b(i) - block.b(i)) + 0.5 * normal(gen_);
                block.c(i) += 0.30 * lr * (teacher_block.c(i) - block.c(i)) + 0.5 * normal(gen_);
                block.d(i) += lr * (best_block.d(i) - block.d(i)) + normal(gen_);
            }
        }

        void applyNegation(std::vector<Strategy> &strategies,
                           const std::vector<Candidate> &candidates,
                           const Candidate &best,
                           const std::vector<Candidate> &elite_archive,
                           const double step_scale)
        {
            if (strategies.empty() || candidates.empty())
            {
                return;
            }

            const Candidate &elite_ref = selectEliteReference(elite_archive, best);
            const int sample_count = std::max(1, std::min(3, static_cast<int>(candidates.size())));

            for (Strategy &strategy : strategies)
            {
                if (strategy.stagnation < options_.stagnation_threshold)
                {
                    continue;
                }

                Strategy counter = negateStrategy(strategy);
                double original_score = 0.0;
                double counter_score = 0.0;
                for (int k = 0; k < sample_count; ++k)
                {
                    const Candidate &base = candidates[randomIndex(static_cast<int>(candidates.size()))];
                    Candidate original_trial;
                    original_trial.position = guideCandidate(base.position, strategy, best.position, elite_ref.position, step_scale);
                    evaluateCandidate(original_trial);

                    Candidate counter_trial;
                    counter_trial.position = guideCandidate(base.position, counter, best.position, elite_ref.position, step_scale);
                    evaluateCandidate(counter_trial);

                    original_score += original_trial.cost + (original_trial.valid ? 0.0 : 1.0e12);
                    counter_score += counter_trial.cost + (counter_trial.valid ? 0.0 : 1.0e12);
                }

                if (counter_score < original_score)
                {
                    strategy = counter;
                    strategy.score = counter_score / sample_count;
                    strategy.stagnation = 0;
                }
            }
        }

        Strategy negateStrategy(const Strategy &strategy)
        {
            Strategy counter = strategy;
            negateStrategyBlock(counter.spatial, true);
            negateStrategyBlock(counter.scale, true);
            negateStrategyBlock(counter.profile, false);
            return counter;
        }

        void negateStrategyBlock(StrategyBlock &block, const bool flip_phase)
        {
            if (block.a.size() == 0)
            {
                return;
            }

            std::normal_distribution<double> noise(0.0, 0.08);
            for (int i = 0; i < block.a.size(); ++i)
            {
                block.a(i) = -block.a(i) + noise(gen_);
                block.d(i) = -block.d(i) + noise(gen_);
                block.b(i) = std::max(0.2, block.b(i) + noise(gen_));
                if (flip_phase)
                {
                    block.c(i) += (randomReal(0.0, 1.0) > 0.5 ? 0.5 * kDpasaPi : -0.5 * kDpasaPi);
                }
                else
                {
                    block.c(i) += noise(gen_);
                }
            }
        }

        void evaluateCandidate(Candidate &candidate)
        {
            const auto result = objective_(candidate.position);
            candidate.cost = std::isfinite(result.first) ? result.first : 1.0e20;
            candidate.valid = result.second && std::isfinite(result.first);
        }

        static bool isBetter(const Candidate &lhs, const Candidate &rhs)
        {
            if (lhs.valid != rhs.valid)
            {
                return lhs.valid;
            }
            return lhs.cost < rhs.cost;
        }

        static void updateBest(const Candidate &candidate, Candidate &best)
        {
            if (best.position.size() == 0 || isBetter(candidate, best))
            {
                best = candidate;
            }
        }

        void updateEliteArchive(const Candidate &candidate, std::vector<Candidate> &elite_archive)
        {
            if (!candidate.valid)
            {
                return;
            }

            elite_archive.push_back(candidate);
            std::sort(elite_archive.begin(), elite_archive.end(),
                      [](const Candidate &lhs, const Candidate &rhs)
                      {
                          return lhs.cost < rhs.cost;
                      });
            if (static_cast<int>(elite_archive.size()) > options_.elite_size)
            {
                elite_archive.resize(options_.elite_size);
            }
        }

        const Candidate &selectEliteReference(const std::vector<Candidate> &elite_archive, const Candidate &best)
        {
            if (elite_archive.empty())
            {
                return best;
            }
            return elite_archive[randomIndex(static_cast<int>(elite_archive.size()))];
        }

        int selectStrategyIndex(const std::vector<int> &ranked_indices, const int elite_strategy_count)
        {
            if (elite_strategy_count <= 1)
            {
                return ranked_indices.front();
            }
            return ranked_indices[randomIndex(elite_strategy_count)];
        }

        Eigen::RowVectorXd repair(const Eigen::RowVectorXd &x) const
        {
            return x.cwiseMax(lb_).cwiseMin(ub_);
        }

        Eigen::RowVectorXd randomUniform(const Eigen::RowVectorXd &min, const Eigen::RowVectorXd &max)
        {
            Eigen::RowVectorXd out(min.size());
            for (int i = 0; i < out.size(); ++i)
            {
                out(i) = randomReal(min(i), max(i));
            }
            return out;
        }

        int randomIndex(const int upper)
        {
            return std::uniform_int_distribution<int>(0, upper - 1)(gen_);
        }

        double randomReal(const double min, const double max)
        {
            return std::uniform_real_distribution<double>(min, max)(gen_);
        }
    };
}

#endif
