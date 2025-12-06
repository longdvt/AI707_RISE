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
    print("Expert dataset: {} episodes, {} steps".format(len(expert_traj_lengths), expert_states.shape[0]))
    np.savez("/home/long/dppo/data/robomimic/can/offline_expert.npz", states=expert_states, actions=expert_actions, rewards=expert_rewards, terminals=expert_dones, traj_lengths=expert_traj_lengths)
    
    total_failed_states = []
    total_failed_actions = []
    total_failed_rewards = []
    total_failed_dones = []
    total_failed_traj_lengths = []
    for i in range(6):
        failed_path = "/home/long/dppo/data/robomimic/can/failed_rollouts_{}.npz".format(i)
        failed_max_n_episodes = 20000000000
        failed_states, failed_actions, failed_rewards, failed_dones, failed_traj_lengths = process_dataset(failed_path, failed_max_n_episodes, 0.0)    
        total_failed_states.append(failed_states)
        total_failed_actions.append(failed_actions)
        total_failed_rewards.append(failed_rewards)
        total_failed_dones.append(failed_dones)
        total_failed_traj_lengths.append(failed_traj_lengths)

    # merge expert and failed to create a offline dataset
    # total_states = np.concatenate([expert_states, *total_failed_states], axis=0)
    # total_actions = np.concatenate([expert_actions, *total_failed_actions], axis=0)
    # total_rewards = np.concatenate([expert_rewards, *total_failed_rewards], axis=0)
    # total_dones = np.concatenate([expert_dones, *total_failed_dones], axis=0)
    # total_traj_lengths = np.concatenate([expert_traj_lengths, *total_failed_traj_lengths], axis=0)
    print("Failed dataset: {} episodes, {} steps".format(len(np.concatenate(total_failed_traj_lengths)), np.concatenate(total_failed_states, axis=0).shape[0]))
    np.savez("/home/long/dppo/data/robomimic/can/offline_failure.npz", states=np.concatenate(total_failed_states, axis=0), actions=np.concatenate(total_failed_actions, axis=0), rewards=np.concatenate(total_failed_rewards, axis=0), terminals=np.concatenate(total_failed_dones, axis=0), traj_lengths=np.concatenate(total_failed_traj_lengths, axis=0))
    