# DenoiseRL v3: First Correct Box reward

`denoise_v3` copies the v2 recipe and adds the First Correct Answer (FCA)
external-redundancy reward from
[Reconsidering Overthinking](https://github.com/HenryZhen97/Reconsidering-Overthinking).
It remains a sequence-level scalar reward; GRPO advantage calculation is
unchanged.

For a rollout whose final answer is already correct:

1. Parse at most the first 10 complete, balanced `\boxed{...}` expressions from
   the policy-generated continuation.
2. Verify those boxes in order with the active reward manager.
3. End the First Correct Solution (FCS) at the first correct box's closing
   brace.
4. Compute `ERD = round(post_fcs_tokens / total_generated_tokens, 2)`.
5. Scale the original accuracy reward by `1 - ERD`.

Incorrect final rollouts, responses without a verifiably correct box among the
retained candidates, and injected DenoiseRL prefix tokens are not shaped by
this reward.

The main v3 scripts enable this reward and disable the older dynamic length
reward by default:

```bash
first_correct_box_reward_enabled=True
first_correct_box_max_boxes=10
first_correct_box_erd_round_digits=2
correct_length_reward_enabled=False
```

Every value can be overridden as an environment variable. Set
`first_correct_box_reward_enabled=False` for the unshaped ablation.

This implements the paper's FCA/external-redundancy component that targets
post-answer continuation and repeated answers. The paper's separate
embedding-service-based internal-redundancy component is not enabled here.
