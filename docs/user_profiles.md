# User Profiles

The simulator includes four required interpretable user profiles.

## action_focused
- high preference for Action
- lower preference for calm genres
- moderate tolerance for direct recommendations

## balanced_viewer
- relatively even genre preferences
- stable engagement
- low sensitivity to questioning and repetition

## novelty_seeking
- prefers variety
- strongest repetition sensitivity
- responds better when recommendations are not repetitive

## question_sensitive
- strongest question sensitivity
- engagement drops quickly with too many questions
- higher abandonment risk under interaction friction

## Shared simulator behavior

All profiles include:
- hidden preference vector over 5 genres
- engagement dynamics
- leave probability dynamics based on engagement and friction
- stochastic recommendation outcomes (`accept`, `skip`, `leave`)

Question responses are binary (`-1`/`1`) and derived from profile preferences with small noise to avoid full determinism.
