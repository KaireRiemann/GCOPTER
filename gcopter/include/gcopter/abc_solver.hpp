#ifndef ABC_H
#define ABC_H

#include <Eigen/Dense>
#include <vector>
#include <functional>
#include <random>
#include <algorithm>
#include <iostream>
#include <cmath>

struct Bee
{
    Eigen::RowVectorXd Position;
    double Cost;
    bool is_vaild;
    int Counter;
};

class ABC
{
public:
    ABC() = default;

    std::pair<double, Eigen::RowVectorXd> optimize(int nP, int MaxIt, const Eigen::VectorXd &ub,
                                                const Eigen::VectorXd &lb, int dim,
                                                std::function<std::pair<double, bool>(const Eigen::RowVectorXd &)> fobj)
    {

        CostFunction = fobj;
        //评估初始种群
        for (int i = 0; i < nP; ++i)
        {
            auto [Cost_local, _] = CostFunction(pop[i].Position);
            pop[i].Cost = Cost_local;
            pop[i].Counter = 0;

            if (pop[i].Cost <= BestSol.Cost)
            {
                BestSol = pop[i];
            }
        }
        const int convergence_window = 5; // 检查最后5次迭代
        std::vector<double> best_costs_history;
        const double convergence_threshold = 1e-3; // 判断收敛条件

        // 主优化循环
        for (int it = 0; it < MaxIt; ++it)
        {
            // Employed Bees Phase
            for (int j = 0; j < nPop; ++j)
            {
                int k = randi_not_eq_i(j, nPop);

                Eigen::RowVectorXd phi = a * uniformRandomVector(
                                              Eigen::RowVectorXd::Zero(nVar),
                                              Eigen::RowVectorXd::Ones(nVar));
                Eigen::RowVectorXd newPosition = pop[j].Position +
                                              phi.cwiseProduct(pop[j].Position - pop[k].Position);

                newPosition = newPosition.cwiseMin(VarMax).cwiseMax(VarMin);
                auto [newCost, is_vaild] = CostFunction(newPosition);

                if (newCost <= pop[j].Cost)
                {
                    pop[j].Position = newPosition;
                    pop[j].Cost = newCost;
                    pop[j].is_vaild = is_vaild;
                    pop[j].Counter = 1;
                    if (newCost <= BestSol.Cost)
                    {
                        BestSol.Position = newPosition;
                        BestSol.Cost = newCost;
                        BestSol.is_vaild = is_vaild;
                    }
                }
                else
                {
                    pop[j].Counter++;
                }
            }

            // Onlooker Bees Phase
            Eigen::RowVectorXd F(nPop);
            double meanCost = 0;
            for (const auto &bee : pop)
            {
                meanCost += bee.Cost;
            }
            meanCost /= nPop;

            for (int i = 0; i < nPop; ++i)
            {
                F(i) = std::exp(-pop[i].Cost / meanCost);
            }
            Eigen::RowVectorXd P = F / F.sum();

            for (int j = 0; j < nPop; ++j)
            {
                int i = rouletteWheelSelection(P);
                int k = randi_not_eq_i(i, nPop);

                Eigen::RowVectorXd phi = a * uniformRandomVector(
                                              Eigen::RowVectorXd::Zero(nVar),
                                              Eigen::RowVectorXd::Ones(nVar));
                Eigen::RowVectorXd newPosition = pop[i].Position +
                                              phi.cwiseProduct(pop[k].Position - pop[i].Position);

                newPosition = newPosition.cwiseMin(VarMax).cwiseMax(VarMin);
                auto [newCost, is_Vaild] = CostFunction(newPosition);

                if (newCost <= pop[i].Cost)
                {
                    pop[i].Position = newPosition;
                    pop[i].Cost = newCost;
                    pop[i].is_vaild = is_Vaild;
                    pop[i].Counter = 1;
                    if (newCost <= BestSol.Cost)
                    {
                        BestSol.Position = newPosition;
                        BestSol.Cost = newCost;
                        BestSol.is_vaild = is_Vaild;
                    }
                }
                else
                {
                    pop[i].Counter++;
                }
            }

            // Scout Bees Phase
            for (auto &bee : pop)
            {
                if (bee.Counter > Limit)
                {
                    bee.Position = uniformRandomVector(VarMin, VarMax);
                    auto [Cost_local, is_vaild] = CostFunction(bee.Position);
                    bee.Cost = Cost_local;
                    bee.is_vaild = is_vaild;
                    bee.Counter = 0;
                }
            }

            BestCosts[it] = BestSol.Cost;
            best_costs_history.push_back(BestSol.Cost);

            // 检查有效性和收敛性
            if (BestSol.is_vaild)
            {
                // 如果有足够的历史记录，检查收敛性
                if (best_costs_history.size() >= convergence_window)
                {
                    bool converged = false;
                    double diff = 0;
                    for (int i = 1; i < convergence_window; ++i)
                    {
                        diff += std::abs(best_costs_history[best_costs_history.size() - i] -
                                         best_costs_history[best_costs_history.size() - i - 1]);
                    }
                    diff /= (convergence_window-1);
                    if (diff < convergence_threshold)
                    {
                        converged = true;
                    }

                    if (converged)
                    {
                        // std::cout << "converged at " << it + 1 << std::endl;
                        return {BestSol.Cost, BestSol.Position};
                    }
                }
            }
        }
        // std::cout << "converged at " << MaxIt << std::endl;
        return {BestSol.Cost, BestSol.Position};
    }

    // 获取最优解
    Bee getBestSolution() const { return BestSol; }

    // 获取收敛曲线
    std::vector<double> getConvergenceCurve() const { return BestCosts; }
    // 初始化函数
    void initialize(int nP, int MaxIt, const Eigen::VectorXd &ub, const Eigen::VectorXd &lb,int dim)
    {
        nVar = dim;
        nPop = nP;
        this->MaxIt = MaxIt;
        VarMin = lb.transpose();
        VarMax = ub.transpose();

        Limit = round(0.6 * nVar * nPop);
        a = 1.0;
        gen = std::mt19937(std::random_device{}());
        dis = std::uniform_real_distribution<>(0.0, 1.0);

        // 初始化最优解
        BestSol.Cost = std::numeric_limits<double>::infinity();
        BestSol.is_vaild = false;
        BestCosts.resize(MaxIt);

        // 初始化种群
        pop.resize(nPop);
        for (int i = 0; i < nPop; ++i)
        {
            pop[i].Position = uniformRandomVector(VarMin, VarMax);
        }

        // std::cout << "ABC Initialization completed." << std::endl;
    }

    // 重载的初始化函数，接收上一轮的最优解
    void initialize(int nP, int MaxIt, const Eigen::VectorXd &ub, const Eigen::VectorXd &lb, 
                   int dim, const Eigen::RowVectorXd &previous_best)
    {
        nVar = dim;
        nPop = nP;
        this->MaxIt = MaxIt;
        VarMin = lb.transpose();
        VarMax = ub.transpose();

        Limit = round(0.6 * nVar * nPop);
        a = 1.0;
        gen = std::mt19937(std::random_device{}());
        dis = std::uniform_real_distribution<>(0.0, 1.0);

        // 初始化最优解
        BestSol.Cost = std::numeric_limits<double>::infinity();
        BestSol.is_vaild = false;
        BestCosts.resize(MaxIt);

        // 初始化种群
        pop.resize(nPop);
        // 将第一个个体设置为上一轮的最优解
        pop[0].Position = previous_best;
        // 初始化其余个体
        for (int i = 1; i < nPop; ++i)
        {
            pop[i].Position = uniformRandomVector(VarMin, VarMax);
        }

        // std::cout << "ABC Initialization completed with previous best solution." << std::endl;
    }

    // 修改初始化函数，只有部分个体基于优先级中间点生成
    void initializeWithPriority(int nP, int MaxIt, const Eigen::VectorXd &ub, const Eigen::VectorXd &lb, 
                               int dim, int priority_based_count, double priority_based_variation_radius, const Eigen::RowVectorXd &priority_waypoints)
    {
        nVar = dim;
        nPop = nP;
        this->MaxIt = MaxIt;
        VarMin = lb.transpose();
        VarMax = ub.transpose();

        Limit = round(0.6 * nVar * nPop);
        a = 1.0;
        gen = std::mt19937(std::random_device{}());
        dis = std::uniform_real_distribution<>(0.0, 1.0);

        // 初始化最优解
        BestSol.Cost = std::numeric_limits<double>::infinity();
        BestSol.is_vaild = false;
        BestCosts.resize(MaxIt);

        // 初始化种群
        pop.resize(nPop);

        // 将第一个个体设置为优先级中间点
        pop[0].Position = priority_waypoints;

        // 只有一部分个体基于优先级中间点生成，其余随机生成
        const double variation_radius = priority_based_variation_radius; // 变异半径

        // 为部分个体生成基于优先级中间点的随机变异
        for (int i = 1; i < priority_based_count; ++i)
        {
            // 复制优先级中间点
            pop[i].Position = priority_waypoints;

            // 对空间坐标进行随机变异（在球形范围内）
            for (int j = 0; j < nVar - (nVar / 4); j += 3) // 假设最后1/4的维度是时间维度
            {
                // 生成随机方向向量
                Eigen::Vector3d random_direction;
                random_direction(0) = dis(gen) * 2.0 - 1.0;
                random_direction(1) = dis(gen) * 2.0 - 1.0;
                random_direction(2) = dis(gen) * 2.0 - 1.0;
                random_direction.normalize();

                // 生成随机半径（在0到variation_radius之间）
                double random_radius = dis(gen) * variation_radius;

                // 应用随机偏移
                pop[i].Position(j) += random_direction(0) * random_radius;
                pop[i].Position(j+1) += random_direction(1) * random_radius;
                pop[i].Position(j+2) += random_direction(2) * random_radius;
            }

            // 对时间分配进行小幅度随机变异（±20%）
            for (int j = nVar - (nVar / 4); j < nVar; ++j)
            {
                double variation = (dis(gen) * 0.4 - 0.2) * pop[i].Position(j); // ±20%变异
                pop[i].Position(j) += variation;
            }

            // 确保所有值在边界内
            pop[i].Position = pop[i].Position.cwiseMin(VarMax).cwiseMax(VarMin);
        }

        // 其余个体完全随机生成
        for (int i = priority_based_count; i < nPop; ++i)
        {
            pop[i].Position = uniformRandomVector(VarMin, VarMax);
        }

        // std::cout << "ABC Initialization completed with priority waypoints." << std::endl;
    }

private:
    // 算法参数
    int nVar;               // 变量数量（维度）
    int nPop;               // 种群大小
    int MaxIt;              // 最大迭代次数
    Eigen::RowVectorXd VarMin; // 下界向量
    Eigen::RowVectorXd VarMax; // 上界向量
    int Limit;              // 限制次数
    double a;               // 加速因子

    // 种群和解
    std::vector<Bee> pop;          // 蜂群
    Bee BestSol;                   // 最优解
    std::vector<double> BestCosts; // 收敛曲线

    // 目标函数
    std::function<std::pair<double, bool>(const Eigen::RowVectorXd &)> CostFunction;

    // 随机数生成
    std::mt19937 gen;
    std::uniform_real_distribution<> dis;

    // 辅助函数
    Eigen::RowVectorXd uniformRandomVector(const Eigen::RowVectorXd &min, const Eigen::RowVectorXd &max)
    {
        Eigen::RowVectorXd v(nVar);
        for (int i = 0; i < nVar; ++i)
        {
            v(i) = min(i) + (max(i) - min(i)) * dis(gen);
        }
        return v;
    }
    int rouletteWheelSelection(const Eigen::RowVectorXd &P)
    {
        double r = dis(gen);
        double c = 0;
        for (int i = 0; i < P.size(); ++i)
        {
            c += P(i);
            if (r <= c)
            {
                return i;
            }
        }
        return P.size() - 1;
    }
    int randi_not_eq_i(int i, int n)
    {
        int j;
        do
        {
            j = std::uniform_int_distribution<>(0, n - 1)(gen);
        } while (j == i);
        return j;
    }


};

#endif // ABC_H