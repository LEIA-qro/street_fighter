import torch
import torch.nn as nn
from stable_baselines3 import PPO
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3.common.env_checker import check_env
from SF2_RL_Framework.bizhawk_env import BizHawkSF2Env

class FighterFeatureExtractor(BaseFeaturesExtractor):
    def __init__(self, observation_space, features_dim=256):
        # We define a final output dimension of 256 for the policy network
        super(FighterFeatureExtractor, self).__init__(observation_space, features_dim)
        
        # 1. Continuous Branch (6 variables: P1_HP, P2_HP, P1_X, P2_X, P1_Y, P2_Y)
        n_continuous = observation_space.spaces['continuous'].shape[0]
        self.continuous_extractor = nn.Sequential(
            nn.Linear(n_continuous, 64),
            nn.ReLU()
        )
        
        # 2. Categorical Branches (Embeddings)
        # Vocabulary size is 256 (for an 8-bit integer). We map to a 16-dimensional continuous space.
        embedding_dim = 16
        self.p1_embedding = nn.Embedding(num_embeddings=256, embedding_dim=embedding_dim)
        self.p2_embedding = nn.Embedding(num_embeddings=256, embedding_dim=embedding_dim)
        
        # 3. Fusion Block
        # We concatenate along the feature dimension: 64 + 16 + 16 = 96
        concat_dim = 64 + embedding_dim + embedding_dim
        
        self.fusion_net = nn.Sequential(
            nn.Linear(concat_dim, 128),
            nn.ReLU(),
            nn.Linear(128, features_dim),
            nn.ReLU()
        )

    def forward(self, observations) -> torch.Tensor:
        # observations is a Dict of batched tensors
        
        # Process continuous data
        cont_out = self.continuous_extractor(observations['continuous'])
        
        # Process categorical data. 
        # Note: Discrete spaces are parsed as floats by SB3, we must cast to long/int64 for embeddings.
        # Squeeze removes any dummy dimensions to ensure shape is [batch_size] before embedding.
        p1_idx = observations['p1_state'].long().squeeze(-1) if observations['p1_state'].dim() > 1 else observations['p1_state'].long()
        p2_idx = observations['p2_state'].long().squeeze(-1) if observations['p2_state'].dim() > 1 else observations['p2_state'].long()
        
        p1_emb = self.p1_embedding(p1_idx)
        p2_emb = self.p2_embedding(p2_idx)
        
        # Mathematically fuse the vectors: $$ x_{fused} = x_{cont} \oplus e_{p1} \oplus e_{p2} $$
        # Concatenate along the feature dimension (dim=1)
        fused = torch.cat([cont_out, p1_emb, p2_emb], dim=1)
        
        return self.fusion_net(fused)

if __name__ == "__main__":
    # 1. Initialize our custom socket environment
    env = BizHawkSF2Env()
    
    # Optional but highly recommended: Validate the custom environment complies with Gym API
    print("Validating environment architecture...")
    check_env(env, warn=True)
    print("Environment validation passed.")
    
    # 2. Inject our custom PyTorch architecture into the PPO algorithm
    policy_kwargs = dict(
        features_extractor_class=FighterFeatureExtractor,
        features_extractor_kwargs=dict(features_dim=256),
        # We explicitly define the MLP dimensions for the Policy (Actor) and Value (Critic) networks
        net_arch=[dict(pi=[256, 128], vf=[256, 128])]
    )
    
    # 3. Instantiate the Model
    # We use a slightly lower learning rate to allow the embeddings to stabilize gracefully
    model = PPO(
        "MultiInputPolicy", # Required when using a Dict observation space
        env,
        policy_kwargs=policy_kwargs,
        learning_rate=1e-4, 
        n_steps=2048,
        batch_size=64,
        verbose=1,
        tensorboard_log="./ppo_sf2_tensorboard/"
    )
    
    print("Beginning Training Loop. Ensure BizHawk Lua Script is running...")
    
    # 4. Execute the Training Loop
    try:
        model.learn(total_timesteps=500000, progress_bar=True)
    except KeyboardInterrupt:
        print("Training interrupted manually.")
    finally:
        # Save the weights unconditionally
        model.save("ppo_sf2_genesis_model")
        env.close()
        print("Model saved and sockets closed.")