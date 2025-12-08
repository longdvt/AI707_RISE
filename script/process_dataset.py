import numpy as np
from tqdm import tqdm

HORIZON_STEPS = 4

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

    if assigned_reward_value == 1:
        # create failure dataset from demonstrations
        other_traj_lengths = data["traj_lengths"][max_n_episodes:]
        # for each demonstrations, I take states from 0 to 0.75 * length of that trajectory
        failed_states = []
        failed_actions = []
        failed_rewards = []
        failed_dones = []
        failed_traj_lengths = []
        curr_idx = total_num_steps
        for i in range(len(other_traj_lengths)):
            sample_length = int(np.random.uniform(0.5, 0.75) * other_traj_lengths[i])
            failed_states.append(data["states"][curr_idx:curr_idx + sample_length])
            failed_actions.append(data["actions"][curr_idx:curr_idx + sample_length])
            failed_rewards.append(0.0 * np.ones((sample_length,)))
            failed_dones.append(0.0 * np.ones((sample_length,)))
            failed_traj_lengths.append(sample_length)
            curr_idx += other_traj_lengths[i]
        failed_states = np.concatenate(failed_states, axis=0)
        failed_actions = np.concatenate(failed_actions, axis=0)
        failed_rewards = np.concatenate(failed_rewards, axis=0)
        failed_dones = np.concatenate(failed_dones, axis=0)
        failed_traj_lengths = np.array(failed_traj_lengths)
        return states, actions, rewards, dones, traj_lengths, failed_states, failed_actions, failed_rewards, failed_dones, failed_traj_lengths
    return states, actions, rewards, dones, traj_lengths

def augment_dataset(
    expert_states, expert_actions, expert_traj_lengths,
    failed_states, failed_actions, failed_traj_lengths, threshold=0.005, horizon_steps=1
):

    # ----- Build the union dataset DU = DE ∪ DNE -----
    union_states = np.concatenate([expert_states, failed_states], axis=0)
    union_actions = np.concatenate([expert_actions, failed_actions], axis=0)

    # Build trajectory boundaries for DU
    union_traj_lengths = np.concatenate([expert_traj_lengths, failed_traj_lengths])
    traj_end_indices = np.cumsum(union_traj_lengths)
    traj_starts = np.concatenate([[0], traj_end_indices[:-1]])

    # ----- Augmented dataset -----
    D_aug = []

    # Precompute for speed
    N = union_states.shape[0]

    # For each (s, a) in DU
    for i in tqdm(range(N)):
        s = union_states[i]

        # Compute all distances in batch (much faster)
        distances = np.linalg.norm(union_states - s, axis=1)

        # Filter neighbors within threshold
        neighbor_idxs = np.where(distances < threshold)[0]

        # For each matched neighbor (s′, a′)
        for j in neighbor_idxs:
            # Extract (s', a') horizon slice
            # Find trajectory boundaries
            # Determine trajectory start/end for j
            traj_id = np.searchsorted(traj_end_indices, j)
            traj_start = traj_starts[traj_id]
            traj_end = traj_end_indices[traj_id]

            # Check if window fits inside trajectory
            if j + horizon_steps <= traj_end:
                a_prime = union_actions[j : j + horizon_steps]
                D_aug.append((s, a_prime))
    
    final_states = np.stack([s for (s, a) in D_aug], axis=0)
    final_actions = np.stack([a for (s, a) in D_aug], axis=0)
    print("Augmented dataset: {} episodes, {} steps".format(len(D_aug), final_states.shape[0]))
    np.savez("/home/long/dppo/data/robomimic/can/offline_augmented.npz", states=final_states, actions=final_actions)

if __name__ == "__main__":
    expert_path = "/home/long/dppo/data/robomimic/can/train.npz"
    expert_max_n_episodes = 20
    # expert_states, expert_actions, expert_rewards, expert_dones, expert_traj_lengths = process_dataset(expert_path, expert_max_n_episodes, 1.0)
    expert_states, expert_actions, expert_rewards, expert_dones, expert_traj_lengths, failed_states, failed_actions, failed_rewards, failed_dones, failed_traj_lengths  = process_dataset(expert_path, expert_max_n_episodes, 1.0)
    print("Expert dataset: {} episodes, {} steps".format(len(expert_traj_lengths), expert_states.shape[0]))
    print("Failed dataset: {} episodes, {} steps".format(len(failed_traj_lengths), failed_states.shape[0]))
    # np.savez("/home/long/dppo/data/robomimic/can/offline_expert.npz", states=expert_states, actions=expert_actions, rewards=expert_rewards, terminals=expert_dones, traj_lengths=expert_traj_lengths)
    # np.savez("/home/long/dppo/data/robomimic/can/offline_failed_from_expert.npz", states=failed_states, actions=failed_actions, rewards=failed_rewards, terminals=failed_dones, traj_lengths=failed_traj_lengths)
    
    total_failed_states = []
    total_failed_actions = []
    total_failed_rewards = []
    total_failed_dones = []
    total_failed_traj_lengths = []
    for i in range(6):
        if i < 5:
            continue
        failed_path = "/home/long/dppo/data/robomimic/can/failed_rollouts_{}.npz".format(i)
        failed_max_n_episodes = 2000
        i_failed_states, i_failed_actions, i_failed_rewards, i_failed_dones, i_failed_traj_lengths = process_dataset(failed_path, failed_max_n_episodes, 0.0)    
        total_failed_states.append(i_failed_states)
        total_failed_actions.append(i_failed_actions)
        total_failed_rewards.append(i_failed_rewards)
        total_failed_dones.append(i_failed_dones)
        total_failed_traj_lengths.append(i_failed_traj_lengths)

    # merge failed and expert to create a offline dataset
    total_failed_states = np.concatenate([failed_states, *total_failed_states], axis=0)
    total_failed_actions = np.concatenate([failed_actions, *total_failed_actions], axis=0)
    total_failed_rewards = np.concatenate([failed_rewards, *total_failed_rewards], axis=0)
    total_failed_dones = np.concatenate([failed_dones, *total_failed_dones], axis=0)
    total_failed_traj_lengths = np.concatenate([failed_traj_lengths, *total_failed_traj_lengths], axis=0)
    print("Failed dataset: {} episodes, {} steps".format(len(total_failed_traj_lengths), total_failed_states.shape[0]))
    # np.savez("/home/long/dppo/data/robomimic/can/offline_failure.npz", states=total_failed_states, actions=total_failed_actions, rewards=total_failed_rewards, terminals=total_failed_dones, traj_lengths=total_failed_traj_lengths)
    
    # Augment expert dataset with failed dataset
    augment_dataset(expert_states, expert_actions, expert_traj_lengths, total_failed_states, total_failed_actions, total_failed_traj_lengths, horizon_steps=HORIZON_STEPS)