"""
OFC Pineapple - Training Configuration

All tunable parameters for reward shaping and training.
"""
from dataclasses import dataclass


@dataclass
class RewardConfig:
    """Reward coefficients (tunable)."""
    bust_penalty: float = -20.0       # Bust penalty (-10 to -30)
    fl_bonus: float = 15.0            # Fantasyland entry bonus
    hand_weight: float = 0.7          # Hand reward weight (α)
    session_weight: float = 0.3       # Session reward weight (β)


@dataclass
class TrainingConfig:
    """Training hyperparameters."""
    # Behavior Cloning
    bc_lr: float = 1e-3
    bc_epochs: int = 100
    bc_batch_size: int = 64
    bc_dropout: float = 0.2

    # Self-Play
    sp_lr: float = 1e-4
    sp_games_per_iter: int = 200
    sp_mcts_simulations: int = 200
    sp_iterations: int = 100
    sp_c_puct: float = 1.5
    sp_temperature: float = 1.0

    # Evaluation
    eval_interval: int = 10
    checkpoint_interval: int = 20

    # Network
    hidden1: int = 512
    hidden2: int = 256
    max_actions: int = 250


# Global defaults
REWARD_CONFIG = RewardConfig()
TRAINING_CONFIG = TrainingConfig()


def compute_hand_reward(hand_result: dict, player: int,
                        config: RewardConfig = REWARD_CONFIG) -> float:
    """Compute reward for one completed hand."""
    chip_change = hand_result["raw_score"][player]

    fl_bonus = config.fl_bonus if hand_result["fl_entry"][player] else 0.0
    bust_penalty = config.bust_penalty if hand_result["busted"][player] else 0.0

    return chip_change + fl_bonus + bust_penalty


def compute_session_reward(final_chips: list, player: int) -> float:
    """Compute normalized session reward (-1.0 to 1.0)."""
    return (final_chips[player] - 200) / 200.0


def compute_combined_reward(hand_reward: float, session_reward: float,
                            config: RewardConfig = REWARD_CONFIG) -> float:
    """Combine hand and session rewards."""
    return config.hand_weight * hand_reward + config.session_weight * session_reward
