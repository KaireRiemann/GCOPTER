#include "SpaceTimeTrajectoryPlanner/SpaceTimeTrajectoryPlanner.hpp"
#include <thread>
#include <future>
#include <unordered_map>
#include <filesystem>
// 构造函数
SpaceTimeTrajectoryPlanner::SpaceTimeTrajectoryPlanner()
    : Node("space_time_trajectory_planner"),
      current_discretized_trajectory_index_(0),
      current_time_(0.0),
      trajectory_ready_(false),
      is_goal_moving_(false)
{
    // Declare all parameters with default values
    this->declare_parameter("space_size_x_min", -100.0);
    this->declare_parameter("space_size_x_max", 100.0);
    this->declare_parameter("space_size_y_min", -100.0);
    this->declare_parameter("space_size_y_max", 100.0);
    this->declare_parameter("space_size_z_min", -100.0);
    this->declare_parameter("space_size_z_max", 100.0);
    this->declare_parameter("obstacle_static_min_size", 10.0);
    this->declare_parameter("obstacle_static_max_size", 20.0);
    this->declare_parameter("obstacle_dynamic_min_size", 5.0);
    this->declare_parameter("obstacle_dynamic_max_size", 10.0);
    this->declare_parameter("obstacle_max_velocity", 10.0);
    this->declare_parameter("num_static_obstacles", 80);
    this->declare_parameter("num_dynamic_obstacles", 50);
    this->declare_parameter("num_agents", 10);
    this->declare_parameter("delta_t", 0.1);
    this->declare_parameter("prediction_time", 200.0);
    this->declare_parameter("prediction_interval", 0.1);
    this->declare_parameter("pre_set_time", 50.0);
    this->declare_parameter("grid_resolution", 0.5);
    this->declare_parameter("grid3d_resolution", 0.25);
    this->declare_parameter("agent_collision_radius", 2.5);
    this->declare_parameter("inflation_radius", 0.25);
    this->declare_parameter("num_Cylinders", 0);
    this->declare_parameter("num_Rings", 0);
    this->declare_parameter("ObstacleStyle", "default");
    this->declare_parameter("start_deriv_value", 0.0);
    this->declare_parameter("goal_deriv_value", 0.0);
    this->declare_parameter("min_start_goal_distance", 200.0);
    this->declare_parameter("traj_max_velocity", 10.0);
    this->declare_parameter("traj_max_acceleration", 15.0);
    this->declare_parameter("np", 50);
    this->declare_parameter("MaxIt", 500);
    this->declare_parameter("num_mid_points", 3);
    this->declare_parameter("time_lb", 0.1);
    this->declare_parameter("time_ub", 20.0);
    this->declare_parameter("optimizer_type", "ABC");
    this->declare_parameter("start_goal_generation_type", "opposite_faces");
    this->declare_parameter("is_goal_moving", false);
    this->declare_parameter("goal_max_velocity", 3.0);
    this->declare_parameter("ClusterRecycleSimulation", "NoSimulation");
    this->declare_parameter("recycling_platform_velocity", 1.0);
    this->declare_parameter("need_generate_hole", false);
    this->declare_parameter("min_cylinder_distance", 1.5);
    this->declare_parameter("cylinder_csv_path", "");
    this->declare_parameter("cubes_csv_path", "");
    this->declare_parameter("use_priority_initialization", false);
    this->declare_parameter("priority_based_count", 20);
    this->declare_parameter("priority_based_variation_radius", 1.0);

    // Get parameter values
    space_size_x_min_ = this->get_parameter("space_size_x_min").as_double();
    space_size_x_max_ = this->get_parameter("space_size_x_max").as_double();
    space_size_y_min_ = this->get_parameter("space_size_y_min").as_double();
    space_size_y_max_ = this->get_parameter("space_size_y_max").as_double();
    space_size_z_min_ = this->get_parameter("space_size_z_min").as_double();
    space_size_z_max_ = this->get_parameter("space_size_z_max").as_double();
    obstacle_static_min_size_ = this->get_parameter("obstacle_static_min_size").as_double();
    obstacle_static_max_size_ = this->get_parameter("obstacle_static_max_size").as_double();
    obstacle_dynamic_min_size_ = this->get_parameter("obstacle_dynamic_min_size").as_double();
    obstacle_dynamic_max_size_ = this->get_parameter("obstacle_dynamic_max_size").as_double();
    obstacle_max_velocity_ = this->get_parameter("obstacle_max_velocity").as_double();
    num_static_obstacles_ = this->get_parameter("num_static_obstacles").as_int();
    num_dynamic_obstacles_ = this->get_parameter("num_dynamic_obstacles").as_int();
    delta_t_ = this->get_parameter("delta_t").as_double();
    prediction_time_ = this->get_parameter("prediction_time").as_double();
    prediction_interval_ = this->get_parameter("prediction_interval").as_double();
    pre_set_time_ = this->get_parameter("pre_set_time").as_double();
    grid_resolution_ = this->get_parameter("grid_resolution").as_double();
    num_Cylinders_ = this->get_parameter("num_Cylinders").as_int();
    num_Rings_ = this->get_parameter("num_Rings").as_int();
    grid3d_resolution_ = this->get_parameter("grid3d_resolution").as_double();
    ObstacleStyle_ = this->get_parameter("ObstacleStyle").as_string();
    start_deriv_value_ = this->get_parameter("start_deriv_value").as_double();
    goal_deriv_value_ = this->get_parameter("goal_deriv_value").as_double();
    min_start_goal_distance_ = this->get_parameter("min_start_goal_distance").as_double();
    traj_max_velocity_ = this->get_parameter("traj_max_velocity").as_double();
    traj_max_acceleration_ = this->get_parameter("traj_max_acceleration").as_double();
    np_ = this->get_parameter("np").as_int();
    MaxIt_ = this->get_parameter("MaxIt").as_int();
    num_mid_points_ = this->get_parameter("num_mid_points").as_int();
    time_lb_ = this->get_parameter("time_lb").as_double();
    time_ub_ = this->get_parameter("time_ub").as_double();
    start_goal_generation_type_ = this->get_parameter("start_goal_generation_type").as_string();
    inflation_radius_ = this->get_parameter("inflation_radius").as_double();
    is_goal_moving_ = this->get_parameter("is_goal_moving").as_bool();
    goal_max_velocity_ = this->get_parameter("goal_max_velocity").as_double();
    ClusterRecycleSimulation_ = this->get_parameter("ClusterRecycleSimulation").as_string();
    recycling_platform_velocity_ = this->get_parameter("recycling_platform_velocity").as_double();
    num_agents_ = this->get_parameter("num_agents").as_int();
    agent_collision_radius_ = this->get_parameter("agent_collision_radius").as_double();
    need_generate_hole_ = this->get_parameter("need_generate_hole").as_bool();
    min_cylinder_distance_ = this->get_parameter("min_cylinder_distance").as_double();
    cylinder_csv_path_ = this->get_parameter("cylinder_csv_path").as_string();
    cubes_csv_path_ = this->get_parameter("cubes_csv_path").as_string();
    use_priority_initialization_ = this->get_parameter("use_priority_initialization").as_bool();
    priority_based_count_ = this->get_parameter("priority_based_count").as_int();
    priority_based_variation_radius_ = this->get_parameter("priority_based_variation_radius").as_double();
    // Initialize vectors for multi-agent data
    start_points_.resize(num_agents_);
    goal_points_.resize(num_agents_);
    optimized_discretized_trajectories_.resize(num_agents_);
    trajectories_.resize(num_agents_);
    current_discretized_trajectory_indices_.resize(num_agents_, 0);
    current_times_.resize(num_agents_, 0.0);
    trajectory_propagation_network_ = std::make_unique<TrajectoryPropagationNetwork>();
    // 初始化agent_contexts_
    agent_contexts_.resize(num_agents_);

    // Print all parameters
    RCLCPP_INFO(this->get_logger(), "Initialized with the following parameters:");
    RCLCPP_INFO(this->get_logger(), "============ Environment Parameters ============");
    RCLCPP_INFO(this->get_logger(), "Space size X: [%.2f, %.2f]", space_size_x_min_, space_size_x_max_);
    RCLCPP_INFO(this->get_logger(), "Space size Y: [%.2f, %.2f]", space_size_y_min_, space_size_y_max_);
    RCLCPP_INFO(this->get_logger(), "Space size Z: [%.2f, %.2f]", space_size_z_min_, space_size_z_max_);
    RCLCPP_INFO(this->get_logger(), "Obstacle Style: %s", ObstacleStyle_.c_str());
    RCLCPP_INFO(this->get_logger(), "Static Obstacle min size: %.2f", obstacle_static_min_size_);
    RCLCPP_INFO(this->get_logger(), "Static Obstacle max size: %.2f", obstacle_static_max_size_);
    RCLCPP_INFO(this->get_logger(), "Dynamic Obstacle min size: %.2f", obstacle_dynamic_min_size_);
    RCLCPP_INFO(this->get_logger(), "Dynamic Obstacle max size: %.2f", obstacle_dynamic_max_size_);
    RCLCPP_INFO(this->get_logger(), "Grid resolution: %.2f", grid_resolution_);
    RCLCPP_INFO(this->get_logger(), "Grid3D resolution: %.2f", grid3d_resolution_);
    RCLCPP_INFO(this->get_logger(), "Inflation radius: %.2f", inflation_radius_);
    RCLCPP_INFO(this->get_logger(), "Obstacle max velocity: %.2f", obstacle_max_velocity_);
    RCLCPP_INFO(this->get_logger(), "Number of static obstacles: %d", num_static_obstacles_);
    RCLCPP_INFO(this->get_logger(), "Number of dynamic obstacles: %d", num_dynamic_obstacles_);
    RCLCPP_INFO(this->get_logger(), "Minimum cylinder distance: %.2f", min_cylinder_distance_);
    RCLCPP_INFO(this->get_logger(), "Number of cylinders: %d", num_Cylinders_);
    RCLCPP_INFO(this->get_logger(), "Number of rings: %d", num_Rings_);
    RCLCPP_INFO(this->get_logger(), "Goal max velocity: %.2f", goal_max_velocity_);
    RCLCPP_INFO(this->get_logger(), "Cluster Recycle Simulation: %s", ClusterRecycleSimulation_.c_str());
    RCLCPP_INFO(this->get_logger(), "Collision radius of the agent: %.2f", agent_collision_radius_);
    RCLCPP_INFO(this->get_logger(), "Goal movement state: %s", is_goal_moving_ ? "Moving" : "Static");
    RCLCPP_INFO(this->get_logger(), "Generate hole: %s", need_generate_hole_ ? "Yes" : "No");
    RCLCPP_INFO(this->get_logger(), "Cylinder csv path: %s", cylinder_csv_path_.c_str());
    RCLCPP_INFO(this->get_logger(), "Cubes csv path: %s", cubes_csv_path_.c_str());
    RCLCPP_INFO(this->get_logger(), "============ Time Parameters ============");
    RCLCPP_INFO(this->get_logger(), "Delta t: %.2f", delta_t_);
    RCLCPP_INFO(this->get_logger(), "Prediction time: %.2f", prediction_time_);
    RCLCPP_INFO(this->get_logger(), "Prediction interval: %.2f", prediction_interval_);
    RCLCPP_INFO(this->get_logger(), "Time lower bound: %.2f", time_lb_);
    RCLCPP_INFO(this->get_logger(), "Time upper bound: %.2f", time_ub_);
    RCLCPP_INFO(this->get_logger(), "Trajectory Pre-set time: %.2f", pre_set_time_);

    RCLCPP_INFO(this->get_logger(), "============ Trajectory Parameters ============");
    RCLCPP_INFO(this->get_logger(), "Start derivative value: %.2f", start_deriv_value_);
    RCLCPP_INFO(this->get_logger(), "Goal derivative value: %.2f", goal_deriv_value_);
    RCLCPP_INFO(this->get_logger(), "Minimum start-goal distance: %.2f", min_start_goal_distance_);
    RCLCPP_INFO(this->get_logger(), "Max velocity: %.2f", traj_max_velocity_);
    RCLCPP_INFO(this->get_logger(), "Max acceleration: %.2f", traj_max_acceleration_);

    RCLCPP_INFO(this->get_logger(), "============ Optimizer Parameters ============");
    RCLCPP_INFO(this->get_logger(), "Population size (np): %d", np_);
    RCLCPP_INFO(this->get_logger(), "Max iterations: %d", MaxIt_);
    RCLCPP_INFO(this->get_logger(), "Number of midpoints: %d", num_mid_points_);
    RCLCPP_INFO(this->get_logger(), "Use priority initialization: %s", use_priority_initialization_ ? "Yes" : "No");
    RCLCPP_INFO(this->get_logger(), "Priority based count: %d", priority_based_count_);
    RCLCPP_INFO(this->get_logger(), "Priority based variation radius: %.2f", priority_based_variation_radius_);

    // Get optimizer type
    std::string optimizer_str = this->get_parameter("optimizer_type").as_string();
    RCLCPP_INFO(this->get_logger(), "Optimizer type: %s", optimizer_str.c_str());

    if (optimizer_str == "ABC")
    {
        optimizer_type_ = OptimizerType::ABC;
    }
    else if (optimizer_str == "GBO")
    {
        optimizer_type_ = OptimizerType::GBO;
    }
    else if (optimizer_str == "PSO_PAR")
    {
        optimizer_type_ = OptimizerType::PSO_PAR;
    }
    else if (optimizer_str == "PSO")
    {
        optimizer_type_ = OptimizerType::PSO;
    }
    else if (optimizer_str == "RUN")
    {
        optimizer_type_ = OptimizerType::RUN;
    }
    else if (optimizer_str == "GBO1002")
    {
        optimizer_type_ = OptimizerType::GBO1002;
    }
    else if (optimizer_str == "GBOPY")
    {
        optimizer_type_ = OptimizerType::GBOPY;
    }
    else if (optimizer_str == "GA")
    {
        optimizer_type_ = OptimizerType::GA;
    }
    else if (optimizer_str == "OABC")
    {
        optimizer_type_ = OptimizerType::OABC;
    }
    else
    {
        RCLCPP_WARN(this->get_logger(), "Unknown optimizer type: %s, using ABC as default",
                    optimizer_str.c_str());
        optimizer_type_ = OptimizerType::ABC;
    }

    // Initialize ObstacleManager
    obstacle_manager_ = std::make_shared<ObstacleManager>(
        this,
        space_size_x_min_, space_size_x_max_,
        space_size_y_min_, space_size_y_max_,
        space_size_z_min_, space_size_z_max_,
        obstacle_static_min_size_,
        obstacle_static_max_size_,
        obstacle_dynamic_min_size_,
        obstacle_dynamic_max_size_,
        obstacle_max_velocity_,
        num_static_obstacles_,
        num_dynamic_obstacles_,
        prediction_time_,
        prediction_interval_,
        num_Cylinders_,
        num_Rings_,
        ObstacleStyle_,
        grid3d_resolution_,
        inflation_radius_,
        ClusterRecycleSimulation_,
        recycling_platform_velocity_,
        num_agents_,
        min_cylinder_distance_,
        cylinder_csv_path_,
        cubes_csv_path_);

    // Create publishers
    visualizer_ = std::make_unique<SpaceTimeVisualization>(this, "world", ObstacleStyle_);
    // Initialize start_goal_manager
    start_goal_manager_ = std::make_unique<StartGoalPointsManager>(
        space_size_x_min_, space_size_x_max_,
        space_size_y_min_, space_size_y_max_,
        space_size_z_min_, space_size_z_max_,
        min_start_goal_distance_,
        obstacle_manager_,
        this->get_logger(),
        is_goal_moving_,
        goal_max_velocity_,
        ClusterRecycleSimulation_,
        recycling_platform_velocity_);

    this->declare_parameter("Numtest", 10);
    int Numtest = this->get_parameter("Numtest").as_int();
    // runNumtest(Numtest);
    // Create timers
    timer_ = this->create_wall_timer(
        std::chrono::milliseconds(100),
        std::bind(&SpaceTimeTrajectoryPlanner::timerCallback, this));

    init_timer_ = this->create_wall_timer(
        std::chrono::milliseconds(1000),
        [this]()
        {
            this->initializeOnce();
            init_timer_->cancel();
        });
}

void SpaceTimeTrajectoryPlanner::initializeOnce()
{
    try
    {
        RCLCPP_INFO(this->get_logger(), "Starting initialization...");
        obstacle_manager_->initializeObstacles();
        if (ObstacleStyle_ == "EgoPlannerStyle")
        {
            RCLCPP_INFO(this->get_logger(), "EgoPlannerStyle selected. Initializing GridMap3D...");
            obstacle_manager_->initializeEgoPlannerStyleObstacles(need_generate_hole_);
        }
        RCLCPP_INFO(this->get_logger(), "Obstacles initialized. Starting to update predicted positions...");
        obstacle_manager_->updatePredictedPositions();
        // obstacle_manager_->initializeSpatialGrid();
        RCLCPP_INFO(this->get_logger(), "Predicted positions updated. Starting to generate start and goal points...");
        start_goal_manager_->generateStartAndGoalPoints(num_agents_, start_points_, goal_points_, start_goal_generation_type_);
        start_goal_manager_->predictGoalTrajectories(prediction_time_, prediction_interval_);
        predicted_goal_trajectories_ = start_goal_manager_->getPredictedGoalTrajectories();
        RCLCPP_INFO(this->get_logger(), "Start and goal points generated. Starting to optimize trajectory...");
        // 初始化agent_contexts_
        RCLCPP_INFO(this->get_logger(), "Initializing agent contexts...");
        initializeAgentContexts();
        optimizeAgentTrajectories();
        RCLCPP_INFO(this->get_logger(), "Initialization completed successfully.");
    }
    catch (const std::exception &e)
    {
        RCLCPP_ERROR(this->get_logger(), "Initialization error: %s", e.what());
    }
}
void SpaceTimeTrajectoryPlanner::runNumtest(int Numtest)
{
    OptimizerType original_optimizer = optimizer_type_;
    RCLCPP_INFO(this->get_logger(), "Starting multi-agent benchmark test with %d environments...", Numtest);

    // Define all optimizer types to test
    std::vector<OptimizerType> optimizers = {
        OptimizerType::ABC,
        OptimizerType::GBO1002,
    };

    // Map for optimizer names
    std::map<OptimizerType, std::string> optimizer_names = {
        {OptimizerType::ABC, "ABC"},
        {OptimizerType::GBO1002, "GBO1002"},
    };

    // Create output files for each optimizer
    std::map<OptimizerType, std::ofstream> result_files;
    for (auto optimizer : optimizers)
    {
        std::string filename = optimizer_names[optimizer] + "_results_multi_agent.csv";
        result_files[optimizer].open(filename);
        // Write header with the new metrics
        result_files[optimizer] << "Environment,TotalPlanningTime,NumAgents,ClusterIterationTimes,TotalCollisionLength,";

        // Add swarm-level metrics right after TotalCollisionLength
        result_files[optimizer] << "TotalStaticCollisions,TotalDynamicCollisions,TotalAgentCollisions,TotalIterations,";

        // Add headers for each agent's metrics
        for (int i = 0; i < num_agents_; i++)
        {
            result_files[optimizer] << "Agent" << i << "_Replan,"; // Changed from Iterations to Replan
            result_files[optimizer] << "Agent" << i << "_IterationTimes,";
            result_files[optimizer] << "Agent" << i << "_TrajectoryTime,";
            result_files[optimizer] << "Agent" << i << "_TrajectoryLength,";
            result_files[optimizer] << "Agent" << i << "_ExceedVelocityMagnitude,";
            result_files[optimizer] << "Agent" << i << "_ExceedAccelerationMagnitude,";
            result_files[optimizer] << "Agent" << i << "_CollisionLength,";
            result_files[optimizer] << "Agent" << i << "_StaticCollisionCount,";
            result_files[optimizer] << "Agent" << i << "_DynamicCollisionCount,";
            result_files[optimizer] << "Agent" << i << "_OtherAgentCollisionCount,";
            result_files[optimizer] << "Agent" << i << "_Cost,";
        }
        result_files[optimizer] << "\n"; // End of header line
    }

    // For each environment
    for (int env = 0; env < Numtest; env++)
    {
        RCLCPP_INFO(this->get_logger(), "Setting up environment %d/%d", env + 1, Numtest);

        // Initialize a new environment
        obstacle_manager_->initializeObstacles();
        if (ObstacleStyle_ == "EgoPlannerStyle")
        {
            obstacle_manager_->initializeEgoPlannerStyleObstacles(need_generate_hole_);
        }
        obstacle_manager_->updatePredictedPositions();

        // Generate start and goal for all agents
        start_goal_manager_->generateStartAndGoalPoints(num_agents_, start_points_, goal_points_, start_goal_generation_type_);
        start_goal_manager_->predictGoalTrajectories(prediction_time_, prediction_interval_);
        predicted_goal_trajectories_ = start_goal_manager_->getPredictedGoalTrajectories();

        // Initialize agent contexts
        initializeAgentContexts();

        // Test each optimizer
        for (auto optimizer : optimizers)
        {
            RCLCPP_INFO(this->get_logger(), "Environment %d/%d: Testing optimizer %s",
                        env + 1, Numtest, optimizer_names[optimizer].c_str());

            // Reset agent contexts for the new optimizer
            for (int i = 0; i < num_agents_; i++)
            {
                agent_contexts_[i].discretized_trajectory.clear();
                agent_contexts_[i].optimization_costtime = 0.0;
                agent_contexts_[i].trajectory_length = 0.0;
                agent_contexts_[i].exceed_velocity_magnitude = 0.0;
                agent_contexts_[i].exceed_acceleration_magnitude = 0.0;
                agent_contexts_[i].collision_length = 0.0;
                agent_contexts_[i].collision_count_with_static_obstacles = 0;
                agent_contexts_[i].collision_count_with_dynamic_obstacles = 0;
                agent_contexts_[i].collision_count_with_other_agents = 0;
                agent_contexts_[i].fitness_value = 0.0;
                agent_contexts_[i].iteration = 0;
            }

            // Clear all trajectories from the network before starting
            trajectory_propagation_network_->clearAllTrajectories();

            // Set current optimizer
            optimizer_type_ = optimizer;

            // For tracking per-agent iteration times
            std::vector<std::map<int, std::vector<double>>> agent_iteration_times(num_agents_);

            // For tracking cluster-wide iteration times
            std::map<int, double> cluster_iteration_times;

            // Start timing the total planning process
            auto total_start_time = std::chrono::high_resolution_clock::now();

            // Multi-agent planning with the current optimizer
            // Step 1: Initial optimization for all agents
            std::unordered_map<int, std::vector<Eigen::Vector4d>> current_trajectories;
            std::vector<std::future<void>> futures;

            auto iter_start_time = std::chrono::high_resolution_clock::now();

            // Parallel optimize all agents' trajectories
            for (int i = 0; i < num_agents_; i++)
            {
                futures.push_back(std::async(std::launch::async, [this, i]()
                                             { optimizeTrajectory(agent_contexts_[i]); }));
            }

            // Wait for all optimizations to complete
            for (auto &future : futures)
            {
                future.wait();
            }

            // Record first iteration times for each agent
            auto iter_end_time = std::chrono::high_resolution_clock::now();
            double iter_time = std::chrono::duration<double>(iter_end_time - iter_start_time).count();

            // Record cluster-wide iteration time for iteration 0
            cluster_iteration_times[0] = iter_time;

            for (int i = 0; i < num_agents_; i++)
            {
                agent_iteration_times[i][0] = {agent_contexts_[i].optimization_costtime};
            }

            // Collect all trajectories and add to the propagation network
            for (int i = 0; i < num_agents_; i++)
            {
                current_trajectories[i] = agent_contexts_[i].discretized_trajectory;
                trajectory_propagation_network_->addTrajectory(
                    current_trajectories[i],
                    agent_collision_radius_,
                    size_t(prediction_time_ / prediction_interval_),
                    i);
            }

            bool has_collisions = true;
            int iteration = 0;
            const int MAX_ITERATIONS = 100; // Maximum number of iterations

            // Continue replanning until no collisions or max iterations reached
            while (has_collisions && iteration < MAX_ITERATIONS)
            {
                iteration++;

                // Generate collision matrix
                Eigen::MatrixXi collision_matrix = trajectory_propagation_network_->generateCollisionMatrix(current_trajectories);

                // Check if no collisions
                if (collision_matrix.sum() == 0)
                {
                    RCLCPP_INFO(this->get_logger(), "No collisions detected, optimization complete!");
                    has_collisions = false;
                    break;
                }

                // Find agents to replan
                std::vector<int> agents_to_replan = trajectory_propagation_network_->findMinimumVertexCover(collision_matrix);

                if (agents_to_replan.empty())
                {
                    RCLCPP_INFO(this->get_logger(), "No agents need replanning, breaking loop");
                    break;
                }

                // Clear trajectories for agents that need replanning
                for (int agent_id : agents_to_replan)
                {
                    trajectory_propagation_network_->clearTrajectory(agent_id);
                    current_trajectories.erase(agent_id);
                    agent_contexts_[agent_id].iteration = iteration;
                }

                // Replan selected agents in parallel
                futures.clear();
                iter_start_time = std::chrono::high_resolution_clock::now();

                for (int agent_id : agents_to_replan)
                {
                    futures.push_back(std::async(std::launch::async, [this, agent_id]()
                                                 { optimizeTrajectory(agent_contexts_[agent_id]); }));
                }

                // Wait for replanning to complete
                for (auto &future : futures)
                {
                    future.wait();
                }

                iter_end_time = std::chrono::high_resolution_clock::now();
                iter_time = std::chrono::duration<double>(iter_end_time - iter_start_time).count();

                // Record cluster-wide iteration time for this iteration
                cluster_iteration_times[iteration] = iter_time;

                // Record iteration times for replanned agents
                for (int agent_id : agents_to_replan)
                {
                    agent_iteration_times[agent_id][iteration] = {agent_contexts_[agent_id].optimization_costtime};
                }

                // Update trajectory collection
                for (int agent_id : agents_to_replan)
                {
                    current_trajectories[agent_id] = agent_contexts_[agent_id].discretized_trajectory;
                    trajectory_propagation_network_->addTrajectory(
                        current_trajectories[agent_id],
                        agent_collision_radius_,
                        size_t(prediction_time_ / prediction_interval_),
                        agent_id);
                }
            }

            // Calculate total planning time
            auto total_end_time = std::chrono::high_resolution_clock::now();
            double total_planning_time = std::chrono::duration<double>(total_end_time - total_start_time).count();

            // Calculate total collisions and iterations
            int total_static_collisions = 0;
            int total_dynamic_collisions = 0;
            int total_agent_collisions = 0;
            int total_iterations = 0;
            double total_collision_length = 0.0; // Total collision trajectory length across all agents

            for (int i = 0; i < num_agents_; i++)
            {
                total_static_collisions += agent_contexts_[i].collision_count_with_static_obstacles;
                total_dynamic_collisions += agent_contexts_[i].collision_count_with_dynamic_obstacles;
                total_agent_collisions += agent_contexts_[i].collision_count_with_other_agents;
                total_collision_length += agent_contexts_[i].collision_length;
            }
            total_iterations = iteration;

            // Write results to file
            result_files[optimizer] << env << "," << total_planning_time << "," << num_agents_ << ",";

            // Write cluster iteration times as a semicolon-separated string
            result_files[optimizer] << "\"";
            bool first_iter = true;
            for (const auto &iter_pair : cluster_iteration_times)
            {
                if (!first_iter)
                {
                    result_files[optimizer] << ";";
                }
                first_iter = false;
                result_files[optimizer] << iter_pair.second;
            }
            result_files[optimizer] << "\",";

            // Write total collision length
            result_files[optimizer] << total_collision_length << ",";

            // Write swarm-level metrics right after TotalCollisionLength
            result_files[optimizer] << total_static_collisions << ","
                                    << total_dynamic_collisions << ","
                                    << total_agent_collisions << ","
                                    << total_iterations << ",";

            // Write per-agent metrics
            for (int i = 0; i < num_agents_; i++)
            {
                // Calculate trajectory time for this agent
                double trajectory_time = agent_contexts_[i].time_allocation.sum();

                // First write the number of iterations (now called replan)
                result_files[optimizer] << agent_contexts_[i].iteration << ",";

                // Write iteration times as a semicolon-separated string
                result_files[optimizer] << "\"";
                bool first = true;
                for (const auto &iter_pair : agent_iteration_times[i])
                {
                    if (!first)
                    {
                        result_files[optimizer] << ";";
                    }
                    first = false;
                    result_files[optimizer] << iter_pair.second[0]; // Write optimization time for this iteration
                }
                result_files[optimizer] << "\",";

                // Write other agent metrics
                result_files[optimizer] << trajectory_time << ","
                                        << agent_contexts_[i].trajectory_length << ","
                                        << agent_contexts_[i].exceed_velocity_magnitude << ","
                                        << agent_contexts_[i].exceed_acceleration_magnitude << ","
                                        << agent_contexts_[i].collision_length << ","
                                        << agent_contexts_[i].collision_count_with_static_obstacles << ","
                                        << agent_contexts_[i].collision_count_with_dynamic_obstacles << ","
                                        << agent_contexts_[i].collision_count_with_other_agents << ","
                                        << agent_contexts_[i].fitness_value << ",";
            }

            // Remove the swarm-level metrics from the end (they're now after TotalCollisionLength)
            result_files[optimizer] << "\n";

            RCLCPP_INFO(this->get_logger(), "Completed environment %d/%d with optimizer %s",
                        env + 1, Numtest, optimizer_names[optimizer].c_str());
            RCLCPP_INFO(this->get_logger(), "Total planning time: %.3f seconds, Total iterations: %d",
                        total_planning_time, total_iterations); // Changed "iterations" to "replans"
            RCLCPP_INFO(this->get_logger(), "Total collision length: %.3f", total_collision_length);
        }
    }

    // Close all files
    for (auto &file_pair : result_files)
    {
        file_pair.second.close();
    }

    RCLCPP_INFO(this->get_logger(), "Multi-agent benchmark test completed. Results saved to CSV files.");
    // Clear all trajectories from the network before starting
    trajectory_propagation_network_->clearAllTrajectories();
    optimizer_type_ = original_optimizer;
}

void SpaceTimeTrajectoryPlanner::initializeAgentContexts()
{
    for (int i = 0; i < num_agents_; ++i)
    {
        agent_contexts_[i].agent_index = i;
        agent_contexts_[i].start_point = start_points_[i];
        agent_contexts_[i].goal_point = goal_points_[i];
        agent_contexts_[i].predicted_goal_trajectory = predicted_goal_trajectories_[i];
        agent_contexts_[i].iteration = 0;
        agent_contexts_[i].pre_set_time = pre_set_time_;
        agent_contexts_[i].optimization_costtime = 0.0;

        // Set priority initialization flag based on goal movement
        // If the goal is not moving, we can use priority initialization
        agent_contexts_[i].use_priority_initialization = use_priority_initialization_;
    }
}

void SpaceTimeTrajectoryPlanner::optimizeAgentTrajectories()
{
    std::unordered_map<int, std::vector<Eigen::Vector4d>> current_trajectories;
    std::vector<std::future<void>> futures;

    // 获取当前时间戳用于文件命名
    auto now = std::chrono::system_clock::now();
    std::time_t now_time = std::chrono::system_clock::to_time_t(now);
    std::tm *now_tm = std::localtime(&now_time);
    char timestamp_buffer[20];
    std::strftime(timestamp_buffer, sizeof(timestamp_buffer), "%Y%m%d_%H%M%S", now_tm);
    std::string timestamp_str(timestamp_buffer);

    // 获取优化器名称
    std::string optimizer_name;
    switch (optimizer_type_)
    {
    case OptimizerType::ABC:
        optimizer_name = "ABC";
        break;
    case OptimizerType::GBO:
        optimizer_name = "GBO";
        break;
    case OptimizerType::PSO_PAR:
        optimizer_name = "PSO_PAR";
        break;
    case OptimizerType::PSO:
        optimizer_name = "PSO";
        break;
    case OptimizerType::RUN:
        optimizer_name = "RUN";
        break;
    case OptimizerType::GBO1002:
        optimizer_name = "GBO1002";
        break;
    case OptimizerType::GBOPY:
        optimizer_name = "GBOPY";
        break;
    case OptimizerType::GA:
        optimizer_name = "GA";
        break;
    case OptimizerType::OABC:
        optimizer_name = "OABC";
        break;
    default:
        optimizer_name = "Unknown";
    }

    // 创建目录（如果不存在）
    std::filesystem::create_directories("data/odometry");
    std::filesystem::create_directories("data/statistics");

    // 为每个agent创建里程计数据文件
    std::vector<std::ofstream> odom_files(num_agents_);
    for (int i = 0; i < num_agents_; ++i)
    {
        std::string odom_filename = "data/odometry/odom_drone" + std::to_string(i) + "_" + optimizer_name + "_" + timestamp_str + ".csv";
        odom_files[i].open(odom_filename);
        odom_files[i] << "timestamp,px,py,pz,vx,vy,vz" << std::endl;
    }

    // 创建飞行统计数据文件
    std::ofstream flight_stats_file("data/statistics/flight_stats_" + optimizer_name + "_" + timestamp_str + ".csv");
    flight_stats_file << "Timestamp,DroneID,FlightDuration,TotalDistance,AverageSpeed,FinalPositionX,FinalPositionY,FinalPositionZ,"
                      << "CollisionWithObs,CollisionWithAgent,MinDistanceToOtherDrones,NumberOfReplans,TotalReplanTime,"
                      << "AverageReplanTime,PercentTimeReplanning" << std::endl;

    // 用于跟踪每个agent的规划时间
    std::vector<double> total_replan_times(num_agents_, 0.0);

    // 第一轮：并行优化所有Agent的轨迹
    RCLCPP_INFO(this->get_logger(), "Starting initial optimization for all agents...");
    auto start_time = std::chrono::high_resolution_clock::now();

    for (int i = 0; i < num_agents_; ++i)
    {
        futures.push_back(std::async(std::launch::async, [this, i]()
                                     { optimizeTrajectory(agent_contexts_[i]); }));
    }

    // 等待所有优化完成
    for (auto &future : futures)
    {
        future.wait();
    }

    // 收集所有轨迹并添加到传播网络
    for (int i = 0; i < num_agents_; ++i)
    {
        current_trajectories[i] = agent_contexts_[i].discretized_trajectory;
        trajectory_propagation_network_->addTrajectory(current_trajectories[i], agent_collision_radius_, size_t(prediction_time_ / prediction_interval_), i);

        // 更新总规划时间
        total_replan_times[i] += agent_contexts_[i].optimization_costtime;
    }

    bool has_collisions = true;
    int iteration = 0;
    const int MAX_ITERATIONS = 100; // 最大迭代次数

    while (has_collisions && iteration < MAX_ITERATIONS)
    {
        iteration++;
        RCLCPP_INFO(this->get_logger(), "\n=== Starting replanning iteration %d ===", iteration);

        // 生成碰撞矩阵
        Eigen::MatrixXi collision_matrix = trajectory_propagation_network_->generateCollisionMatrix(current_trajectories);

        // 打印碰撞矩阵
        std::stringstream matrix_ss;
        matrix_ss << "\nCollision Matrix:\n"
                  << collision_matrix << "\n";
        RCLCPP_INFO(this->get_logger(), "%s", matrix_ss.str().c_str());

        // 如果没有碰撞，退出循环
        if (collision_matrix.sum() == 0)
        {
            RCLCPP_INFO(this->get_logger(), "No collisions detected, optimization complete!");
            has_collisions = false;
            break;
        }

        // 找到需要重新规划的Agent
        std::vector<int> agents_to_replan = trajectory_propagation_network_->findMinimumVertexCover(collision_matrix);

        // 打印需要重规划的Agent
        std::stringstream agents_ss;
        agents_ss << "Agents selected for replanning: ";
        for (int agent_id : agents_to_replan)
        {
            agents_ss << agent_id << " ";
            agent_contexts_[agent_id].iteration = iteration;
        }
        RCLCPP_INFO(this->get_logger(), "%s", agents_ss.str().c_str());

        if (agents_to_replan.empty())
        {
            RCLCPP_INFO(this->get_logger(), "No agents need replanning, breaking loop");
            break;
        }

        // 清除需要重新规划的Agent的轨迹
        for (int agent_id : agents_to_replan)
        {
            trajectory_propagation_network_->clearTrajectory(agent_id);
            current_trajectories.erase(agent_id);
        }

        // 并行重新规划选定的Agent
        futures.clear();
        auto replan_start_time = std::chrono::high_resolution_clock::now();

        for (int agent_id : agents_to_replan)
        {
            futures.push_back(std::async(std::launch::async, [this, agent_id]()
                                         { optimizeTrajectory(agent_contexts_[agent_id]); }));
        }

        // 等待重规划完成
        for (auto &future : futures)
        {
            future.wait();
        }

        auto replan_end_time = std::chrono::high_resolution_clock::now();
        double replan_time = std::chrono::duration<double>(replan_end_time - replan_start_time).count();

        RCLCPP_INFO(this->get_logger(), "Replanning iteration %d completed in %.3f seconds",
                    iteration, replan_time);

        // 更新轨迹集合和规划时间
        for (int agent_id : agents_to_replan)
        {
            current_trajectories[agent_id] = agent_contexts_[agent_id].discretized_trajectory;
            trajectory_propagation_network_->addTrajectory(current_trajectories[agent_id],
                                                           agent_collision_radius_,
                                                           size_t(prediction_time_ / prediction_interval_),
                                                           agent_id);

            // 更新总规划时间
            total_replan_times[agent_id] += agent_contexts_[agent_id].optimization_costtime;
        }
    }

    auto end_time = std::chrono::high_resolution_clock::now();
    double total_optimization_time = std::chrono::duration<double>(end_time - start_time).count();
    RCLCPP_INFO(this->get_logger(), "All optimization completed in %.3f seconds", total_optimization_time);
    RCLCPP_INFO(this->get_logger(), "Total iterations: %d", iteration);

    // 最终更新所有轨迹到存储变量中
    for (const auto &pair : current_trajectories)
    {
        int agent_id = pair.first;
        optimized_discretized_trajectories_[agent_id] = pair.second;
        trajectories_[agent_id] = agent_contexts_[agent_id].trajectory;
    }

    // 计算并显示最终的碰撞统计
    int Swarm_collision_count_static = 0;
    int Swarm_collision_count_dynamic = 0;
    int Swarm_collision_count_agent = 0;
    double Swarm_exceed_velocity_magnitude = 0.0;
    double Swarm_exceed_acceleration_magnitude = 0.0;

    for (const auto &agent_context : agent_contexts_)
    {
        Swarm_collision_count_static += agent_context.collision_count_with_static_obstacles;
        Swarm_collision_count_dynamic += agent_context.collision_count_with_dynamic_obstacles;
        Swarm_collision_count_agent += agent_context.collision_count_with_other_agents;
        Swarm_exceed_velocity_magnitude += agent_context.exceed_velocity_magnitude;
        Swarm_exceed_acceleration_magnitude += agent_context.exceed_acceleration_magnitude;
    }

    RCLCPP_INFO(this->get_logger(), "Final Swarm planning results:");
    RCLCPP_INFO(this->get_logger(), "Swarm static obstacle collision count: %d", Swarm_collision_count_static);
    RCLCPP_INFO(this->get_logger(), "Swarm dynamic obstacle collision count: %d", Swarm_collision_count_dynamic);
    RCLCPP_INFO(this->get_logger(), "Swarm other agent collision count: %d", Swarm_collision_count_agent);
    RCLCPP_INFO(this->get_logger(), "Swarm exceed velocity magnitude: %f", Swarm_exceed_velocity_magnitude);
    RCLCPP_INFO(this->get_logger(), "Swarm exceed acceleration magnitude: %f", Swarm_exceed_acceleration_magnitude);

    // 生成里程计数据和飞行统计数据
    for (int i = 0; i < num_agents_; ++i)
    {
        // 获取轨迹总时间
        double trajectory_time = agent_contexts_[i].time_allocation.sum();
        double total_distance = 0.0;

        // 生成里程计数据
        Eigen::Vector3d prev_pos;
        bool first_point = true;

        // 以0.01秒的间隔离散化轨迹
        for (double t = 0.0; t <= trajectory_time; t += 0.01)
        {
            Eigen::Vector3d pos = agent_contexts_[i].trajectory.getPos(t);
            Eigen::Vector3d vel = agent_contexts_[i].trajectory.getVel(t);

            // 写入里程计数据
            odom_files[i] << std::fixed << std::setprecision(6)
                          << t << ","
                          << pos(0) << "," << pos(1) << "," << pos(2) << ","
                          << vel(0) << "," << vel(1) << "," << vel(2) << std::endl;

            // 计算总距离
            if (!first_point)
            {
                total_distance += (pos - prev_pos).norm();
            }
            else
            {
                first_point = false;
            }
            prev_pos = pos;
        }

        // 计算与其他无人机的最小距离
        double min_distance_to_other_drones = std::numeric_limits<double>::max();

        for (int j = 0; j < num_agents_; ++j)
        {
            if (i == j)
                continue; // 跳过自己

            double other_trajectory_time = agent_contexts_[j].time_allocation.sum();

            // 以0.1秒的间隔检查距离
            for (double t = 0.0; t <= trajectory_time; t += 0.01)
            {
                Eigen::Vector3d pos_i = agent_contexts_[i].trajectory.getPos(t);
                Eigen::Vector3d pos_j;

                // 如果当前时间超过了其他轨迹的时间，使用其他轨迹的最后位置
                if (t <= other_trajectory_time)
                {
                    pos_j = agent_contexts_[j].trajectory.getPos(t);
                }
                else
                {
                    pos_j = agent_contexts_[j].trajectory.getPos(other_trajectory_time);
                }

                double distance = (pos_i - pos_j).norm();
                min_distance_to_other_drones = std::min(min_distance_to_other_drones, distance);
            }
        }

        // 计算平均速度
        double average_speed = total_distance / trajectory_time;

        // 获取最终位置
        Eigen::Vector3d final_position = agent_contexts_[i].trajectory.getPos(trajectory_time);

        // 判断是否有碰撞
        bool collision_with_obs = (agent_contexts_[i].collision_count_with_static_obstacles > 0 ||
                                   agent_contexts_[i].collision_count_with_dynamic_obstacles > 0);
        bool collision_with_agent = (agent_contexts_[i].collision_count_with_other_agents > 0);

        // 计算规划相关统计
        int number_of_replans = agent_contexts_[i].iteration + 1; // +1 因为初始规划也算一次
        double total_replan_time = total_replan_times[i];
        double average_replan_time = total_replan_time / number_of_replans;
        double percent_time_replanning = (total_replan_time / trajectory_time) * 100.0;

        // 写入飞行统计数据
        flight_stats_file << timestamp_str << ","
                          << i << ","
                          << trajectory_time << ","
                          << total_distance << ","
                          << average_speed << ","
                          << final_position(0) << "," << final_position(1) << "," << final_position(2) << ","
                          << (collision_with_obs ? "true" : "false") << ","
                          << (collision_with_agent ? "true" : "false") << ","
                          << (min_distance_to_other_drones == std::numeric_limits<double>::max() ? "inf" : std::to_string(min_distance_to_other_drones)) << ","
                          << number_of_replans << ","
                          << total_replan_time << ","
                          << average_replan_time << ","
                          << percent_time_replanning << std::endl;
    }

    // 关闭所有文件
    for (auto &file : odom_files)
    {
        file.close();
    }
    flight_stats_file.close();

    RCLCPP_INFO(this->get_logger(), "Odometry data saved to data/odometry directory");
    RCLCPP_INFO(this->get_logger(), "Flight statistics saved to data/statistics directory");

    trajectory_ready_ = true;
    visualizer_->publishDerivatives(optimized_discretized_trajectories_, trajectories_);
    RCLCPP_INFO(this->get_logger(), "Trajectory is ready for execution");
}
// 添加新的优化器运行函数
std::pair<double, Eigen::RowVectorXd> SpaceTimeTrajectoryPlanner::runOptimizer(int nP, int MaxIt,
                                                                               const VectorXd &ub,
                                                                               const VectorXd &lb, int dim,
                                                                               AgentContext &agent_context)
{
    switch (optimizer_type_)
    {
    case OptimizerType::ABC:
    {
        auto optimizer = ABC();

        if (agent_context.iteration == 0)
        {
            if (agent_context.use_priority_initialization)
            {
                // 使用优先级初始化
                optimizer.initializeWithPriority(nP, MaxIt, ub, lb, dim, priority_based_count_, priority_based_variation_radius_, agent_context.priority_waypoints);
            }
            else
            {
                // 使用普通初始化
                optimizer.initialize(nP, MaxIt, ub, lb, dim);
            }
        }
        else
        {
            optimizer.initialize(nP, MaxIt, ub, lb, dim, agent_context.best_waypoints);
        }

        return optimizer.optimize(nP, MaxIt, ub, lb, dim,
                                  [this, &agent_context](const Eigen::RowVectorXd &x)
                                  { return this->fitness(x, agent_context); });
    }
    case OptimizerType::OABC:
    {
        auto optimizer = OABC();

        if (agent_context.iteration == 0)
        {
            if (agent_context.use_priority_initialization)
            {
                // 使用优先级初始化
                optimizer.initializeWithPriority(nP, MaxIt, ub, lb, dim, priority_based_count_, priority_based_variation_radius_, agent_context.priority_waypoints);
            }
            else
            {
                // 使用普通初始化
                optimizer.initialize(nP, MaxIt, ub, lb, dim);
            }
        }
        else
        {
            optimizer.initialize(nP, MaxIt, ub, lb, dim, agent_context.best_waypoints);
        }

        return optimizer.optimize(nP, MaxIt, ub, lb, dim,
                                  [this, &agent_context](const Eigen::RowVectorXd &x)
                                  { return this->fitness(x, agent_context); });
    }
    case OptimizerType::GBO:
    {
        auto optimizer = GBO();
        if (agent_context.iteration == 0)
        {
            optimizer.initialize(nP, MaxIt, ub, lb, dim);
        }
        else
        {
            optimizer.initialize(nP, MaxIt, ub, lb, dim, agent_context.best_waypoints);
        }
        return optimizer.optimize(nP, MaxIt, ub, lb, dim,
                                  [this, &agent_context](const Eigen::RowVectorXd &x)
                                  { return this->fitness(x, agent_context); });
    }
    case OptimizerType::PSO_PAR:
    {

        auto optimizer = PSO_PAR();
        if (agent_context.iteration == 0)
        {
            optimizer.initialize(nP, MaxIt, ub, lb, dim);
        }
        else
        {
            optimizer.initialize(nP, MaxIt, ub, lb, dim, agent_context.best_waypoints);
        }
        return optimizer.optimize(nP, MaxIt, ub, lb, dim,
                                  [this, &agent_context](const Eigen::RowVectorXd &x)
                                  { return this->fitness(x, agent_context); });
    }
    case OptimizerType::PSO:
    {
        auto optimizer = PSO();
        if (agent_context.iteration == 0)
        {
            optimizer.initialize(nP, MaxIt, ub, lb, dim);
        }
        else
        {
            optimizer.initialize(nP, MaxIt, ub, lb, dim, agent_context.best_waypoints);
        }
        return optimizer.optimize(nP, MaxIt, ub, lb, dim,
                                  [this, &agent_context](const Eigen::RowVectorXd &x)
                                  { return this->fitness(x, agent_context); });
    }
    case OptimizerType::RUN:
    {
        auto optimizer = RUN();
        if (agent_context.iteration == 0)
        {
            optimizer.initialize(nP, MaxIt, ub, lb, dim);
        }
        else
        {
            optimizer.initialize(nP, MaxIt, ub, lb, dim, agent_context.best_waypoints);
        }
        return optimizer.optimize(nP, MaxIt, ub, lb, dim,
                                  [this, &agent_context](const Eigen::RowVectorXd &x)
                                  { return this->fitness(x, agent_context); });
    }
    case OptimizerType::GBO1002:
    {
        auto optimizer = GBO1002();
        if (agent_context.iteration == 0)
        {
            if (agent_context.use_priority_initialization)
            {
                // 使用优先级初始化
                optimizer.initializeWithPriority(nP, MaxIt, ub, lb, dim, priority_based_count_, priority_based_variation_radius_, agent_context.priority_waypoints);
            }
            else
            {
                // 使用普通初始化
                optimizer.initialize(nP, MaxIt, ub, lb, dim);
            }
        }
        else
        {
            optimizer.initialize(nP, MaxIt, ub, lb, dim, agent_context.best_waypoints);
        }
        return optimizer.optimize(nP, MaxIt, ub, lb, dim,
                                  [this, &agent_context](const Eigen::RowVectorXd &x)
                                  { return this->fitness(x, agent_context); });
    }
    case OptimizerType::GBOPY:
    {
        auto optimizer = GBOPY();
        if (agent_context.iteration == 0)
        {
            optimizer.initialize(nP, MaxIt, ub, lb, dim);
        }
        else
        {
            optimizer.initialize(nP, MaxIt, ub, lb, dim, agent_context.best_waypoints);
        }
        return optimizer.optimize(nP, MaxIt, ub, lb, dim,
                                  [this, &agent_context](const Eigen::RowVectorXd &x)
                                  { return this->fitness(x, agent_context); });
    }
    case OptimizerType::GA:
    {
        auto optimizer = GA();
        if (agent_context.iteration == 0)
        {
            optimizer.initialize(nP, MaxIt, ub, lb, dim);
        }
        else
        {
            optimizer.initialize(nP, MaxIt, ub, lb, dim, agent_context.best_waypoints);
        }
        return optimizer.optimize(nP, MaxIt, ub, lb, dim,
                                  [this, &agent_context](const Eigen::RowVectorXd &x)
                                  { return this->fitness(x, agent_context); });
    }
    default:
    {
        RCLCPP_WARN(this->get_logger(), "Unknown optimizer type, using ABC");
        auto optimizer = ABC();
        if (agent_context.iteration == 0)
        {
            if (agent_context.use_priority_initialization)
            {
                // 使用优先级初始化
                optimizer.initializeWithPriority(nP, MaxIt, ub, lb, dim, priority_based_count_, priority_based_variation_radius_, agent_context.priority_waypoints);
            }
            else
            {
                // 使用普通初始化
                optimizer.initialize(nP, MaxIt, ub, lb, dim);
            }
        }
        else
        {
            optimizer.initialize(nP, MaxIt, ub, lb, dim, agent_context.best_waypoints);
        }
        return optimizer.optimize(nP, MaxIt, ub, lb, dim,
                                  [this, &agent_context](const Eigen::RowVectorXd &x)
                                  { return this->fitness(x, agent_context); });
    }
    }
}
void SpaceTimeTrajectoryPlanner::reportTrajectoryStatistics(AgentContext &agent_context)
{
    if (agent_context.discretized_trajectory.empty())
    {
        RCLCPP_WARN(this->get_logger(), "Trajectory is not ready for reporting statistics.");
        return;
    }
    // 检查速度和加速度限制
    auto [exceed_velocity_magnitude, exceed_acceleration_magnitude] = checkVelocityAndAccelerationLimits2(agent_context);
    // RCLCPP_INFO(this->get_logger(), "Exceeded velocity magnitude: %f", exceed_velocity_magnitude);
    // RCLCPP_INFO(this->get_logger(), "Exceeded acceleration magnitude: %f", exceed_acceleration_magnitude);
    agent_context.exceed_acceleration_magnitude = exceed_acceleration_magnitude;
    agent_context.exceed_velocity_magnitude = exceed_velocity_magnitude;

    // 检查碰撞并获取碰撞点数量
    agent_context.collision_count_with_static_obstacles = obstacle_manager_->checkStaticObstacleCollisions(agent_context.discretized_trajectory);
    agent_context.collision_count_with_dynamic_obstacles = obstacle_manager_->checkDynamicObstacleCollisions(agent_context.discretized_trajectory);
    agent_context.collision_count_with_other_agents = trajectory_propagation_network_->checkCollisions(agent_context.discretized_trajectory);
    if (ObstacleStyle_ == "EgoPlannerStyle")
    {
        // agent_context.collision_count_with_static_obstacles = obstacle_manager_->checkEgoPlannerStyleCollisionsReport(agent_context.discretized_trajectory);
        agent_context.collision_count_with_static_obstacles = obstacle_manager_->checkEgoPlannerStyleCollisions(agent_context.discretized_trajectory);
    }

    if (ObstacleStyle_ == "EgoPlannerStyle")
    {
        // 计算碰撞长度(动态障碍物、静态障碍物、其他智能体)
        agent_context.collision_length = obstacle_manager_->checkTrajectoryCollisionswithEgoStyle(agent_context.discretized_trajectory, &(trajectory_propagation_network_->trajectories_));
    }
    else
    {
        agent_context.collision_length = obstacle_manager_->checkTrajectoryCollisions(agent_context.discretized_trajectory, &(trajectory_propagation_network_->trajectories_));
    }
    // 计算轨迹长度
    agent_context.trajectory_length = agent_context.trajectory.getLength();
    // RCLCPP_INFO(this->get_logger(), "Static obstacle collision count: %d", agent_context.collision_count_with_static_obstacles);
    // RCLCPP_INFO(this->get_logger(), "Dynamic obstacle collision count: %d", agent_context.collision_count_with_dynamic_obstacles);
    // RCLCPP_INFO(this->get_logger(), "Other agent collision count: %d", agent_context.collision_count_with_other_agents);
}
// 优化轨迹
void SpaceTimeTrajectoryPlanner::optimizeTrajectory(AgentContext &agent_context)
{
    try
    {
        // RCLCPP_INFO(this->get_logger(), "Starting Agent %d trajectory optimization...", agent_context.agent_index);
        auto start_time = std::chrono::high_resolution_clock::now();

        int nP = np_;
        int MaxIt = MaxIt_;
        int dim = 3 * num_mid_points_ + (num_mid_points_ + 1); // 3*N个空间坐标 + (N+1)个时间分配
        VectorXd lb = VectorXd::Zero(dim);
        VectorXd ub = VectorXd::Zero(dim);

        // int spatial_dims = 3 * num_mid_points_; // 假设前3/4的维度是空间维度
        int time_dims = num_mid_points_ + 1; // 最后1/4的维度是时间维度

        // 设置空间维度的上下界
        for (int i = 0; i < num_mid_points_; i++)
        {
            // X 坐标边界
            lb(3 * i) = space_size_x_min_;
            ub(3 * i) = space_size_x_max_;
            // Y 坐标边界
            lb(3 * i + 1) = space_size_y_min_;
            ub(3 * i + 1) = space_size_y_max_;
            // Z 坐标边界
            lb(3 * i + 2) = space_size_z_min_;
            ub(3 * i + 2) = space_size_z_max_;
        }

        // 设置时间维度的上下界
        lb.tail(time_dims).setConstant(time_lb_);
        ub.tail(time_dims).setConstant(time_ub_);

        // 如果是第一次迭代且需要优先级初始化，则生成优先初始化的中间点
        if (agent_context.iteration == 0 && agent_context.use_priority_initialization)
        {
            // 生成直线上均匀分布的中间点
            generatePriorityWaypoints(agent_context);
        }

        auto [best_cost, best_waypoints] = runOptimizer(nP, MaxIt, ub, lb, dim, agent_context);

        auto end_time = std::chrono::high_resolution_clock::now();
        double total_time = std::chrono::duration<double>(end_time - start_time).count();
        agent_context.fitness_value = best_cost;
        agent_context.best_waypoints = best_waypoints;
        agent_context.optimization_costtime = total_time;
        // 根据优化器类型打印相应信息
        std::string optimizer_name;
        switch (optimizer_type_)
        {
        case OptimizerType::ABC:
            optimizer_name = "ABC";
            break;
        case OptimizerType::GBO:
            optimizer_name = "GBO";
            break;
        case OptimizerType::PSO_PAR:
            optimizer_name = "PSO_PAR";
            break;
        case OptimizerType::PSO:
            optimizer_name = "PSO";
            break;
        case OptimizerType::RUN:
            optimizer_name = "RUN";
            break;
        case OptimizerType::GBO1002:
            optimizer_name = "GBO1002";
            break;
        case OptimizerType::GBOPY:
            optimizer_name = "GBOPY";
            break;
        case OptimizerType::GA:
            optimizer_name = "GA";
            break;
        case OptimizerType::OABC:
            optimizer_name = "OABC";
            break;
        default:
            optimizer_name = "Unknown";
        }

        RCLCPP_INFO(this->get_logger(), "%s completed. Best cost: %f", optimizer_name.c_str(), best_cost);
        RCLCPP_INFO(this->get_logger(), "Trajectory optimization completed in %.3f seconds using %s optimizer", total_time, optimizer_name.c_str());

        // RCLCPP_INFO(this->get_logger(), "Fitting spline...");
        fitSpline(best_waypoints, agent_context);
        double trj_total_time = agent_context.time_allocation.sum();
        int num_samples = static_cast<int>(trj_total_time / delta_t_) + 1;
        agent_context.discretized_trajectory.clear();
        agent_context.discretized_trajectory.reserve(num_samples);

        for (double ti = 0; ti <= trj_total_time; ti += delta_t_)
        {
            Vector4d point;
            point.head<3>() = agent_context.trajectory.getPos(ti);
            point(3) = ti;
            agent_context.discretized_trajectory.push_back(point);
        }
        // 将最后的总时间加入轨迹点
        Vector4d point;
        point.head<3>() = agent_context.trajectory.getPos(trj_total_time);
        point(3) = trj_total_time;
        agent_context.discretized_trajectory.push_back(point);

        RCLCPP_INFO(this->get_logger(), "Agent %d trajectory optimization completed. Optimized discretized_trajectory size: %zu", agent_context.agent_index, agent_context.discretized_trajectory.size());
        reportTrajectoryStatistics(agent_context);
    }
    catch (const std::exception &e)
    {
        RCLCPP_ERROR(this->get_logger(), "Error optimizing trajectory: %s", e.what());
    }
}

// 适应度函数
std::pair<double, bool> SpaceTimeTrajectoryPlanner::fitness(const RowVectorXd &waypoints, AgentContext &agent_context)
{

    fitSpline(waypoints, agent_context);
    // 采样轨迹点进行评估
    std::vector<Vector4d> traj_points;
    double total_time = agent_context.time_allocation.sum();

    int num_samples = static_cast<int>(total_time / delta_t_) + 1;
    traj_points.reserve(num_samples);
    double exceed_velocity_magnitude = 0.0;
    double exceed_acceleration_magnitude = 0.0;
    double traj_length = 0.0; // 用于累加轨迹时空长度

    bool first_point = true; // 标记是否是第一个点
    int ii = 0;
    for (double ti = 0; ti <= total_time; ti += delta_t_)
    {
        Vector4d point;
        point.head<3>() = agent_context.trajectory.getPos(ti);
        point(3) = ti;
        traj_points.push_back(point);

        // 计算轨迹长度
        if (!first_point)
        {
            Vector4d diff = point - traj_points[ii - 1]; // 时空轨迹长度
            traj_length += diff.norm();
        }
        else
        {
            first_point = false;
        }
        ii++;

        // 获取速度
        Eigen::Vector3d velocity = agent_context.trajectory.getVel(ti);
        double velocity_magnitude = velocity.norm();
        // 检查速度限制
        if (velocity_magnitude > traj_max_velocity_)
        {
            // 计算超速幅度的积分
            exceed_velocity_magnitude += (velocity_magnitude - traj_max_velocity_) * delta_t_;
        }

        // 获取加速度
        Eigen::Vector3d acceleration = agent_context.trajectory.getAcc(ti);
        double acceleration_magnitude = acceleration.norm();

        // 检查加速度限制
        if (acceleration_magnitude > traj_max_acceleration_)
        {
            // 计算超加速幅度的积分
            exceed_acceleration_magnitude += (acceleration_magnitude - traj_max_acceleration_) * delta_t_;
        }
    }
    double collision_length = 0.0;
    if (ObstacleStyle_ == "EgoPlannerStyle")
    {
        collision_length = obstacle_manager_->checkTrajectoryCollisionswithEgoStyle(traj_points, &(trajectory_propagation_network_->trajectories_));
    }
    else
    {
        collision_length = obstacle_manager_->checkTrajectoryCollisions(traj_points, &(trajectory_propagation_network_->trajectories_));
    }

    double initialFitness = traj_length;

    // 计算最终的适应度值，考虑路径长度、与目标点的距离以及碰撞情况
    double final_fitness = initialFitness + 30 * (collision_length) + exceed_velocity_magnitude * 20 / delta_t_ + exceed_acceleration_magnitude * 25 / delta_t_;

    bool is_valid = false;
    if (collision_length <= 1E-12 && exceed_velocity_magnitude <= 1E-3 && exceed_acceleration_magnitude <= 1E-3)
    {
        is_valid = true;
    }
    is_valid = false;
    return {final_fitness, is_valid};
}
// 使用样条插值拟合路径
inline void SpaceTimeTrajectoryPlanner::fitSpline(const RowVectorXd &waypoints, AgentContext &agent_context)
{
    int pieceNum = num_mid_points_ + 1; // 段数 = 中间点数量 + 1

    // 设置中间点 (每三个元素表示一个点的XYZ)
    Eigen::Matrix3Xd innerPoints(3, num_mid_points_);
    for (int i = 0; i < num_mid_points_; i++)
    {
        // 每个点的XYZ坐标存储在连续的三个位置
        innerPoints(0, i) = waypoints(3 * i);     // X坐标
        innerPoints(1, i) = waypoints(3 * i + 1); // Y坐标
        innerPoints(2, i) = waypoints(3 * i + 2); // Z坐标
    }

    // 提取时间分配 (最后N+1个元素)
    agent_context.time_allocation.resize(pieceNum); // 正确的初始化方式
    for (int i = 0; i < pieceNum; i++)
    {
        agent_context.time_allocation(i) = waypoints(3 * num_mid_points_ + i);
    }
    // 获取轨迹总时长
    double total_time = agent_context.time_allocation.sum();
    // 缩放时间到预设轨迹时间
    // agent_context.time_allocation *= (agent_context.pre_set_time / total_time);
    // total_time = agent_context.time_allocation.sum();
    // 获取对应时刻的目标点位置
    int trajectory_index = static_cast<int>(round(total_time / prediction_interval_));
    Eigen::Vector3d final_goal;
    // 从StartGoalPointsManager获取预测的目标点位置
    final_goal = agent_context.predicted_goal_trajectory[trajectory_index];
    // Set start and goal for current agent
    agent_context.trajectory_state.headState.setZero();
    agent_context.trajectory_state.tailState.setZero();

    agent_context.trajectory_state.headState.col(0) = agent_context.start_point;
    agent_context.trajectory_state.tailState.col(0) = final_goal;

    // 设置优化器条件
    agent_context.curve_fitter.setConditions(agent_context.trajectory_state.headState, agent_context.trajectory_state.tailState, pieceNum);
    agent_context.curve_fitter.setParameters(innerPoints, agent_context.time_allocation);

    // 获取优化后的轨迹
    agent_context.curve_fitter.getTrajectory(agent_context.trajectory);
}

std::pair<double, double> SpaceTimeTrajectoryPlanner::checkVelocityAndAccelerationLimits2(const AgentContext &agent_context)
{

    double exceed_velocity_magnitude = 0.0;
    double exceed_acceleration_magnitude = 0.0;

    for (size_t i = 0; i < agent_context.discretized_trajectory.size(); ++i)
    {
        double t = agent_context.discretized_trajectory[i][3]; // 获取时间，Vector4d的第四个元素

        // 获取速度
        Eigen::Vector3d velocity = agent_context.trajectory.getVel(t);
        double velocity_magnitude = velocity.norm();

        // 检查速度限制
        if (velocity_magnitude > traj_max_velocity_)
        {
            exceed_velocity_magnitude += velocity_magnitude - traj_max_velocity_;
        }

        // 获取加速度
        Eigen::Vector3d acceleration = agent_context.trajectory.getAcc(t);
        double acceleration_magnitude = acceleration.norm();

        // 检查加速度限制
        if (acceleration_magnitude > traj_max_acceleration_)
        {
            exceed_acceleration_magnitude += acceleration_magnitude - traj_max_acceleration_;
        }
    }

    return {exceed_velocity_magnitude, exceed_acceleration_magnitude};
}
void SpaceTimeTrajectoryPlanner::updateAgentPositions()
{
    for (int i = 0; i < num_agents_; ++i)
    {
        current_times_[i] += delta_t_;

        // 使用索引判断是否到达轨迹末端
        if (current_discretized_trajectory_indices_[i] >= optimized_discretized_trajectories_[i].size() - 1)
        {
            // 到达轨迹末端，添加新的目标点
            Eigen::Vector4d new_goal_point;
            new_goal_point.head<3>() = goal_points_[i];
            new_goal_point[3] = optimized_discretized_trajectories_[i].back()[3] + delta_t_; // 在最后一个时间点基础上增加时间步长

            optimized_discretized_trajectories_[i].push_back(new_goal_point);
            current_discretized_trajectory_indices_[i]++; // 更新索引指向新添加的目标点
        }
        else
        {
            size_t new_index = std::min(static_cast<size_t>(current_times_[i] / delta_t_),
                                        optimized_discretized_trajectories_[i].size() - 1);
            current_discretized_trajectory_indices_[i] = new_index;
        }
    }
}

// 定时器回调函数
void SpaceTimeTrajectoryPlanner::timerCallback()
{
    if (!trajectory_ready_)
    {
        RCLCPP_DEBUG(this->get_logger(), "Waiting for trajectory optimization to complete...");
        return;
    }

    try
    {

        obstacle_manager_->updateDynamicObstacles(delta_t_);
        obstacle_manager_->publishObstacles();
        // 更新目标点位置
        goal_points_ = start_goal_manager_->updateGoalPositions(delta_t_);
        updateAgentPositions();

        // 使用新的可视化器发布轨迹
        visualizer_->publishOptimizedDiscretizedtrajectory(optimized_discretized_trajectories_, start_points_, goal_points_, current_discretized_trajectory_indices_, num_agents_);
    }
    catch (const std::exception &e)
    {
        RCLCPP_ERROR(this->get_logger(), "Error in timerCallback: %s", e.what());
        trajectory_ready_ = false;
    }
}

void SpaceTimeTrajectoryPlanner::generatePriorityWaypoints(AgentContext &agent_context)
{
    // 初始化优先级中间点
    Eigen::RowVectorXd priority_waypoints(3 * num_mid_points_ + (num_mid_points_ + 1));

    // 获取起点和终点
    Eigen::Vector3d start = agent_context.start_point;
    Eigen::Vector3d goal = agent_context.goal_point;

    // 计算起点到终点的向量
    Eigen::Vector3d direction = goal - start;
    double distance = direction.norm();

    // 归一化方向向量
    if (distance > 1e-6)
    {
        direction /= distance;
    }

    // 生成直线上均匀分布的中间点
    for (int i = 0; i < num_mid_points_; i++)
    {
        // 计算中间点在直线上的位置 (均匀分布)
        double ratio = (i + 1.0) / (num_mid_points_ + 1.0);
        Eigen::Vector3d midpoint = start + ratio * (goal - start);

        // 存储中间点坐标
        priority_waypoints(3 * i) = midpoint(0);     // X坐标
        priority_waypoints(3 * i + 1) = midpoint(1); // Y坐标
        priority_waypoints(3 * i + 2) = midpoint(2); // Z坐标
    }

    // 计算基于最大速度的时间分配
    double estimated_time = distance / (traj_max_velocity_ * 0.8); // 使用最大速度的80%
    double segment_time = estimated_time / (num_mid_points_ + 1);

    // 设置时间分配
    for (int i = 0; i < num_mid_points_ + 1; i++)
    {
        priority_waypoints(3 * num_mid_points_ + i) = segment_time;
    }

    // 存储优先级中间点
    agent_context.priority_waypoints = priority_waypoints;
}

int main(int argc, char *argv[])
{
    rclcpp::init(argc, argv);
    rclcpp::spin(std::make_shared<SpaceTimeTrajectoryPlanner>());
    rclcpp::shutdown();
    return 0;
}