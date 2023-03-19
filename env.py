from typing import Optional, List, Tuple
import math
import numpy as np
from gymnasium.envs.box2d.lunar_lander import *
    
from algorithms import Policy, Trajectory

class ActionOffsetLunarLander(LunarLander):
    def __init__(
        self,
        min_engine_power: float = 0.5,
        gravity: float = -10.0,
        enable_wind: bool = False,
        wind_power: float = 15.0,
        turbulence_power: float = 1.5,
    ):
        super().__init__(render_mode="rgb_array", continuous=True, gravity=gravity, enable_wind=enable_wind, wind_power=wind_power, turbulence_power=turbulence_power)
        
        # Minimum engine power, below which the engine is manually turned off
        # By default, at 0.5, the main engine will not fire unless the action value is > 0, in the upper half of the allowable range
        self.min_engine_power = min_engine_power
        
        # State variable to store action offset for current task
        self.action_offset = np.zeros(2)
        
    def reset(self, action_offset: np.ndarray, *args, **kwargs):
        self.action_offset = action_offset
        return super().reset(*args, **kwargs)
        
    def step(self, action):
        # Add task-specific action offset
        action = action + self.action_offset
        
        # LunarLander.step()
        assert self.lander is not None

        # Update wind
        assert self.lander is not None, "You forgot to call reset()"
        if self.enable_wind and not (
            self.legs[0].ground_contact or self.legs[1].ground_contact
        ):
            # the function used for wind is tanh(sin(2 k x) + sin(pi k x)),
            # which is proven to never be periodic, k = 0.01
            wind_mag = (
                math.tanh(
                    math.sin(0.02 * self.wind_idx)
                    + (math.sin(math.pi * 0.01 * self.wind_idx))
                )
                * self.wind_power
            )
            self.wind_idx += 1
            self.lander.ApplyForceToCenter(
                (wind_mag, 0.0),
                True,
            )

            # the function used for torque is tanh(sin(2 k x) + sin(pi k x)),
            # which is proven to never be periodic, k = 0.01
            torque_mag = math.tanh(
                math.sin(0.02 * self.torque_idx)
                + (math.sin(math.pi * 0.01 * self.torque_idx))
            ) * (self.turbulence_power)
            self.torque_idx += 1
            self.lander.ApplyTorque(
                (torque_mag),
                True,
            )

        if self.continuous:
            action = np.clip(action, -1, +1).astype(np.float32)
        else:
            assert self.action_space.contains(
                action
            ), f"{action!r} ({type(action)}) invalid "

        # Engines
        tip = (math.sin(self.lander.angle), math.cos(self.lander.angle))
        side = (-tip[1], tip[0])
        dispersion = [self.np_random.uniform(-1.0, +1.0) / SCALE for _ in range(2)]

        m_power = 0.0
        if (self.continuous) or (
            not self.continuous and action == 2
        ):
            # Main engine
            if self.continuous:
                # ------ OUR CHANGE ------
                # Adjustable dead-zone/min-power
                m_power = (action[0] + 1) / 2
                m_power = np.clip(m_power, self.min_engine_power, 1.0)
                assert m_power >= self.min_engine_power and m_power <= 1.0
                
                '''
                m_power = (np.clip(action[0], 0.0, 1.0) + 1.0) * 0.5  # 0.5..1.0
                assert m_power >= 0.5 and m_power <= 1.0
                '''
            else:
                m_power = 1.0
            # 4 is move a bit downwards, +-2 for randomness
            ox = tip[0] * (4 / SCALE + 2 * dispersion[0]) + side[0] * dispersion[1]
            oy = -tip[1] * (4 / SCALE + 2 * dispersion[0]) - side[1] * dispersion[1]
            impulse_pos = (self.lander.position[0] + ox, self.lander.position[1] + oy)
            p = self._create_particle(
                3.5,  # 3.5 is here to make particle speed adequate
                impulse_pos[0],
                impulse_pos[1],
                m_power,
            )  # particles are just a decoration
            p.ApplyLinearImpulse(
                (ox * MAIN_ENGINE_POWER * m_power, oy * MAIN_ENGINE_POWER * m_power),
                impulse_pos,
                True,
            )
            self.lander.ApplyLinearImpulse(
                (-ox * MAIN_ENGINE_POWER * m_power, -oy * MAIN_ENGINE_POWER * m_power),
                impulse_pos,
                True,
            )

        s_power = 0.0
        if (self.continuous) or (
            not self.continuous and action in [1, 3]
        ):
            # Orientation engines
            if self.continuous:
                # ------ OUR CHANGE ------
                # Adjustable dead-zone/min-power
                direction = np.sign(action[1])
                s_power = np.abs(action[1])
                s_power = np.clip(s_power, self.min_engine_power, 1.0)
                assert s_power >= self.min_engine_power and s_power <= 1.0
                
                '''
                direction = np.sign(action[1])
                s_power = np.clip(np.abs(action[1]), 0.5, 1.0)
                assert s_power >= 0.5 and s_power <= 1.0
                '''
            else:
                direction = action - 2
                s_power = 1.0
            ox = tip[0] * dispersion[0] + side[0] * (
                3 * dispersion[1] + direction * SIDE_ENGINE_AWAY / SCALE
            )
            oy = -tip[1] * dispersion[0] - side[1] * (
                3 * dispersion[1] + direction * SIDE_ENGINE_AWAY / SCALE
            )
            impulse_pos = (
                self.lander.position[0] + ox - tip[0] * 17 / SCALE,
                self.lander.position[1] + oy + tip[1] * SIDE_ENGINE_HEIGHT / SCALE,
            )
            p = self._create_particle(0.7, impulse_pos[0], impulse_pos[1], s_power)
            p.ApplyLinearImpulse(
                (ox * SIDE_ENGINE_POWER * s_power, oy * SIDE_ENGINE_POWER * s_power),
                impulse_pos,
                True,
            )
            self.lander.ApplyLinearImpulse(
                (-ox * SIDE_ENGINE_POWER * s_power, -oy * SIDE_ENGINE_POWER * s_power),
                impulse_pos,
                True,
            )

        self.world.Step(1.0 / FPS, 6 * 30, 2 * 30)

        pos = self.lander.position
        vel = self.lander.linearVelocity
        state = [
            (pos.x - VIEWPORT_W / SCALE / 2) / (VIEWPORT_W / SCALE / 2),
            (pos.y - (self.helipad_y + LEG_DOWN / SCALE)) / (VIEWPORT_H / SCALE / 2),
            vel.x * (VIEWPORT_W / SCALE / 2) / FPS,
            vel.y * (VIEWPORT_H / SCALE / 2) / FPS,
            self.lander.angle,
            20.0 * self.lander.angularVelocity / FPS,
            1.0 if self.legs[0].ground_contact else 0.0,
            1.0 if self.legs[1].ground_contact else 0.0,
        ]
        assert len(state) == 8

        reward = 0
        shaping = (
            -100 * np.sqrt(state[0] * state[0] + state[1] * state[1])
            - 100 * np.sqrt(state[2] * state[2] + state[3] * state[3])
            - 100 * abs(state[4])
            + 10 * state[6]
            + 10 * state[7]
        )  # And ten points for legs contact, the idea is if you
        # lose contact again after landing, you get negative reward
        if self.prev_shaping is not None:
            reward = shaping - self.prev_shaping
        self.prev_shaping = shaping

        reward -= (
            m_power * 0.30
        )  # less fuel spent is better, about -30 for heuristic landing
        reward -= s_power * 0.03

        terminated = False
        if self.game_over or abs(state[0]) >= 1.0:
            terminated = True
            reward = -100
        if not self.lander.awake:
            terminated = True
            reward = +100

        if self.render_mode == "human":
            self.render()
        return np.array(state, dtype=np.float32), reward, terminated, False, {}
        


def collect_trajectories(env: ActionOffsetLunarLander, policy: Policy, 
                         task_params: np.ndarray, num_episodes: int,
                         max_episode_length: int = 400, render: bool = False,
                         eval: bool = True
                         ) -> Tuple[
                                List[List[Trajectory]],
                                dict,
                                Optional[List[List[np.ndarray]]]
                            ]:
    """_summary_

    Args:
        env (ActionOffsetLunarLander): _description_
        policy (Policy): _description_
        task_params (np.ndarray): Parameters for each task to collect in. Shape = (task batch size, 2)
        num_episodes (int): _description_
        render (bool): Whether to render frames for each task.
        eval (bool): Whether we're doing evaluation. If so, make starting conditions deterministic.

    Returns:
        List[List[Trajectory]]: List (over tasks) of list (over episodes) of trajectories.
        dict: Metrics to log.
        Optional[List[List[np.ndarray]]]: List of rendered frames for each task.
    """
    trajectories = []
    if render:
        frames = []
    else:
        frames = None
        
    for action_offset in task_params:
        trajectories.append([])
        if render:
            frames.append([])
        for episode_index in range(num_episodes):
            states = []
            actions = []
            rewards = []
            
            policy.reset(action_offset, eval=eval)
            s, _ = env.reset(action_offset, seed=episode_index)
            terminated = False
            truncated = False
            episode_length = 0
            if render:
                frames[-1].append(env.render())
            
            # Save initial state
            states.append(s)
            
            while not (terminated or truncated):
                a = policy.get_action(s)
                next_s, r, terminated, truncated, info = env.step(a)
                episode_length += 1
                
                # Artificially truncate if over episode max length
                if episode_length >= max_episode_length:
                    truncated = True
                
                if render:
                    frames[-1].append(env.render())
                
                # Add to saved trajectory info
                actions.append(a)
                rewards.append(r)
                states.append(next_s)
                
                # Provide feedback to policy
                policy.update_memory(s, a, r, next_s)
                
                # Prepare for next step
                s = next_s
            
            trajectories[-1].append(Trajectory(states, actions, rewards, terminated=terminated))
            
    # TODO: Compute metrics
    metrics = {
        "mean_return": np.mean([traj.rewards.sum() for task_trajs in trajectories for traj in task_trajs]),
        "mean_ep_len": np.mean([len(traj) for task_trajs in trajectories for traj in task_trajs]),
        "return_std": np.std([traj.rewards.sum() for task_trajs in trajectories for traj in task_trajs])
    }
        
    return trajectories, metrics, frames