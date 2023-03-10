import os
import json
import click
import time
from tqdm import tqdm, trange
from typing import Optional
import numpy as np
from gymnasium.utils.save_video import save_video
from torch.utils.tensorboard import SummaryWriter

from env import ActionOffsetLunarLander, collect_trajectories
from util import deep_update_dict
from config import default_config
from algorithms import Trainer




@click.command()
@click.argument("config")
@click.option("-r", "--resume", "load_dir", default=None, help="Resume an existing run with the given output folder")
@click.option("-t", "iterations", type=int, default=10000, help="Number of iterations to run.")
def main(config: str, load_dir: Optional[str], iterations: int):
    '''
    Load command-line arg config file.
    If resuming from existing run, load config from that run directly
    (in this case, the config file passed as command-line arg is ignored).
    '''
    if load_dir:
        with open(os.path.join(load_dir, "config.json")) as fp:
            config = json.load(fp)
    else:
        with open(os.path.join(config)) as fp:
            config = json.load(fp)
    
    # Fill in missing values from default config
    config = deep_update_dict(config, default_config)
    
    
    
    '''
    Init trainer for chosen class
    New algorithms should be added to this if statement
    '''
    if config["algo_class"] == "temp":
        trainer = Trainer(config, load_dir=load_dir)
    elif config["algo_class"] == "naive_ddpg":
        from algorithms import NaiveDDPG
        trainer = NaiveDDPG(config, load_dir=load_dir)
    elif config["algo_class"] == "no_offset_sac":
        from algorithms import NoOffsetSAC
        trainer = NoOffsetSAC(config, load_dir=load_dir)
    elif config["algo_class"] == "no_offset_ddpg":
        from algorithms import NoOffsetDDPG
        trainer = NoOffsetDDPG(config, load_dir=load_dir)
    elif config["algo_class"] == "maml_ddpg":
        from algorithms import MamlDDPG
        trainer = MamlDDPG(config, load_dir=load_dir)
    elif config["algo_class"] == "pearl_ddpg":
        from algorithms import PearlDDPG
        trainer = PearlDDPG(config, load_dir=load_dir)
    elif config["algo_class"] == "simplified_pearl_ddpg":
        from algorithms import SimplifiedPearlDDPG
        trainer = SimplifiedPearlDDPG(config, load_dir=load_dir)
    elif config["algo_class"] == "ddpg_with_offset_mlp":
        from algorithms import OffsetMLPDDPG
        # TODO: add commandline option to specify offset_net_path
        trainer = OffsetMLPDDPG(config, load_dir=load_dir, offset_net_path="./notebooks/output/sas_to_offset_MLP_1_bestval.pt")
    else:
        raise NotImplementedError
    
    
    '''
    If new run, create new output folder.
    If resuming, just use previous output folder.
    '''
    if load_dir is None:
        # Create a new output_dir name with the lowest available number
        num = 0
        output_dir = os.path.join("output", config["algo_class"] + "_" + str(num))
        while os.path.exists(output_dir):
            num += 1
            output_dir = os.path.join("output", config["algo_class"] + "_" + str(num))
    else:
        output_dir = load_dir
        
    # init env with newly randomized or loaded task params
    env = ActionOffsetLunarLander(min_engine_power=config["min_engine_power"])
    
    # init tensorboard logger
    logger = SummaryWriter(log_dir=os.path.join(output_dir, "logs"))
    
    
    
    # Main Loop
    for it in trange(iterations):
        log_info = {}
        
        # Select batch of tasks
        task_indices = np.random.choice(config["num_train_tasks"], size=config["train_task_batch_size"], replace=True).tolist()
        train_task_params = trainer.task_params[task_indices]
        
        # Collect rollouts
        rollouts, train_env_metrics, _ = collect_trajectories(env, trainer.current_policy(), train_task_params, config["train_episodes"], config["max_episode_length"], eval=False)
        log_info.update({f"train_env/{k}": v for k, v in train_env_metrics.items()})
        
        # Train step
        trainer_metrics = trainer.train_step(task_indices, rollouts)
        log_info.update({f"trainer/{k}": v for k, v in trainer_metrics.items()})
        
        # Periodically run on test
        if config["test_period"] and it % config["test_period"] == 0:
            test_task_indices = (config["num_train_tasks"] + np.random.choice(config["num_test_tasks"], size=config["test_task_batch_size"], replace=True)).tolist()
            test_task_params = trainer.task_params[test_task_indices]
            rollouts, test_env_metrics, _ = collect_trajectories(env, trainer.current_policy(), test_task_params, config["test_episodes"], config["max_episode_length"], eval=True)
            log_info.update({f"test_env/{k}": v for k, v in test_env_metrics.items()})
        
        # Periodically record
        if config["record_period"] and it % config["record_period"] == 0:
            # Record videos for maximal action offsets in each direction
            record_task_params = np.array([
                [0, 1],
                [0, -1],
                [1, 0],
                [-1, 0]
            ]) * config["action_offset_magnitude"]
            
            _, _, frames = collect_trajectories(env, trainer.current_policy(), record_task_params, config["record_episodes"], config["max_episode_length"], render=True, eval=True)
            for action_offset, task_frames in zip(record_task_params, frames):
                save_video(
                    task_frames,
                    os.path.join(output_dir, "videos"),
                    episode_trigger=lambda x: True,
                    fps=env.metadata["render_fps"],
                    name_prefix=f"step_{trainer.trainer_state['global_step']}.offset_{action_offset[0]:.2f}_{action_offset[1]:.2f}"
                )
                
        # Periodically save
        if config["save_period"] and it % config["save_period"] == 0:
            trainer.save(output_dir)
        
        # Log
        for k, v in log_info.items():
            logger.add_scalar(k, v, global_step=trainer.trainer_state["global_step"])

    
    





    
if __name__ == "__main__":
    main()