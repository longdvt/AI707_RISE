import numpy as np

def process_dataset(dataset_path, max_n_episodes, assigned_reward_value=1):
    data = np.load(dataset_path, allow_pickle=False)
    traj_lengths = data["traj_lengths"][:max_n_episodes]
    total_num_steps = np.sum(traj_lengths)
    states = data["states"][:total_num_steps]
    actions = data["actions"][:total_num_steps]
    assert states.shape[0] == actions.shape[0]
    
    # create next_state, assign reward, done
    rewards = np.ones((states.shape[0],)) * assigned_reward_value
    dones = np.zeros((states.shape[0],))
    if assigned_reward_value == 1:
        idx = 0
        for i in range(len(traj_lengths)):
            idx += traj_lengths[i]
            dones[idx - 1] = 1
    return states, actions, rewards, dones, traj_lengths

if __name__ == "__main__":
    expert_path = "/home/long/dppo/data/robomimic/can/train.npz"
    expert_max_n_episodes = 20
    expert_states, expert_actions, expert_rewards, expert_dones, expert_traj_lengths = process_dataset(expert_path, expert_max_n_episodes, 1.0)
    

    failed_path = "/home/long/dppo/log/robomimic-eval/can_eval_diffusion_mlp_ta4_td20/2025-12-03_18-58-01_42/failed_rollouts.npz"
    failed_max_n_episodes = 20000
    failed_states, failed_actions, failed_rewards, failed_dones, failed_traj_lengths = process_dataset(failed_path, failed_max_n_episodes, 0.0)    

    # merge expert and failed to create a offline dataset
    total_states = np.concatenate([expert_states, failed_states], axis=0)
    total_actions = np.concatenate([expert_actions, failed_actions], axis=0)
    total_rewards = np.concatenate([expert_rewards, failed_rewards], axis=0)
    total_dones = np.concatenate([expert_dones, failed_dones], axis=0)
    total_traj_lengths = np.concatenate([expert_traj_lengths, failed_traj_lengths], axis=0)
    np.savez("/home/long/dppo/data/robomimic/can/offline.npz", states=total_states, actions=total_actions, rewards=total_rewards, terminals=total_dones, traj_lengths=total_traj_lengths)
    