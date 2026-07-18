import torch

from models.comtkd_marl import CoMTKDMARL, MAPPOTrainer, RolloutBuffer


def test_policy_shapes_and_ppo_update():
    torch.manual_seed(4)
    batch, teachers, observation_dim = 12, 4, 12
    policy = CoMTKDMARL(
        teacher_count=teachers,
        observation_dim=observation_dim,
        hidden_dim=32,
        attention_heads=4,
        sync_rounds=1,
    )
    observations = torch.randn(batch, teachers, observation_dim)
    output = policy.act(observations)
    assert output.raw_actions.shape[:2] == (batch, teachers)
    assert torch.allclose(output.teacher_weights.sum(dim=1), torch.ones(batch), atol=1e-6)
    assert torch.all(output.active_mask.sum(dim=1) == output.cardinality.count)

    buffer = RolloutBuffer()
    rewards = torch.randn(batch)
    agent_rewards = torch.randn(batch, teachers)
    buffer.add(
        observations,
        output.global_observation,
        output.raw_actions,
        output.cardinality.index,
        output.actor_log_probs,
        output.cardinality.log_prob,
        rewards,
        agent_rewards,
        output.critic.team_value,
        output.critic.agent_values,
    )
    actor_optimizer = torch.optim.Adam(policy.actor_parameters(), lr=1e-3)
    critic_optimizer = torch.optim.Adam(policy.critic.parameters(), lr=1e-3)
    trainer = MAPPOTrainer(
        policy,
        actor_optimizer,
        critic_optimizer,
        epochs=1,
        minibatch_size=6,
    )
    stats = trainer.update(buffer, torch.device("cpu"))
    assert len(buffer) == 0
    assert all(torch.isfinite(torch.tensor(value)) for value in stats.__dict__.values())
