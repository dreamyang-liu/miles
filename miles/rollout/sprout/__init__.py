"""Sprout rollouts: Sprout runs and grades the agent, Miles trains on the messages.

Miles submits one prompt group per ``POST /rollout-groups`` to Sprout's RL
driver; Sprout executes ``n_samples_per_prompt`` independent mini-swe-agent
rollouts of the task in its own sandboxes, grades each final snapshot and
returns every trajectory as hint-free Chat Completions messages with a 0/1
reward. Miles retokenizes those messages, masks everything but the assistant
turns and trains with its ordinary GRPO path. See ``rollout_fn.SproutRolloutFn``.

With ``--sprout-rollout-rafo`` each group is a RAFO search instead
(``protocol.RafoSearchSpec``): where every root failed, Sprout's reviewer writes
a turn that exposes the problem, continuations run after it, and deeper reviews
follow in failed continuations. ``forest.read_forest`` reads the forest Sprout
returns, ``rafo.assemble_rafo_group`` gives each sample its RAFO role
(``rafo_role.RafoRole``), and the RAFO loss trains them.
"""
