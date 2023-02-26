# default experiment settings
default_config = dict(
    algo_class="temp",          # Name of the algorithm to run
    algo_config=dict(),         # Algorithm-specific settings, like network size or learning rate
    
    # Environment and Tasks
    observation_size=8,
    action_size=2,
    max_episode_length=400,
    discount_rate=0.99,
    action_offset_magnitude=0.2,
    num_train_tasks=100,
    num_test_tasks=20,
    
    # Train data rollouts
    train_task_batch_size=1,    # Number of train tasks to sample for train step
    train_episodes=1,           # Number of rollouts per task for train step
    
    # Eval data rollouts
    test_period=10,             # How often to collect test metrics
    test_task_batch_size=8,     # Number of test tasks to sample for eval step
    test_episodes=1,            # Number of rollouts per task for eval step
    
    # Record rollouts
    record_period=100,          # How often to collect rollout recordings
    record_episodes=2,          # Number of rollouts per task for recording
    
    # Saving checkpoints
    save_period=10,             # How often to save network state.
)