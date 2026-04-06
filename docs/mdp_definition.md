# MDP Definition

## State

The environment tracks session context and preference belief:
- current step
- remaining question budget
- engagement
- estimated genre preference scores (Action, Comedy, Drama, Sci-Fi, Documentary)
- uncertainty from belief magnitude
- last action type
- recent recommendation outcome
- repetition level

Two forms are provided:
- tabular discrete state id for Q-Learning
- vector observation for DQN/PPO

## Action

Discrete action space of size 8:
- `0..2`: ask question actions
- `3..7`: recommend genre actions

## Transition

At each step:
1. Agent picks action.
2. User simulator returns response:
   - answer for question
   - accept/skip/leave for recommendation
3. Environment updates belief, engagement-driven uncertainty, and repetition tracking.
4. Reward is computed.
5. Episode terminates at max steps or abandonment.

## Reward

Reward terms are configurable:
- accepted recommendation reward
- continuation bonus
- skipped recommendation penalty
- abandonment penalty
- question cost
- invalid-question penalty
- repetition penalty

## Termination

Episode ends when:
- `max_steps` reached, or
- user leaves session.
