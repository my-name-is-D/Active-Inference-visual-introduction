## Section 4 beats

**0. The debt.**
The agent can hold a belief, carry it through a step, and score an approximate posterior. It has been executing actions from a list settled in advance. Nothing built so far scores an action: $F$ says how well a belief accounts for an observation already received, and says nothing about what to do next.

---

**1. Preferences.**
$C$: a distribution over observations, written $p(o \mid C)$. One entry per observation, summing to one.

The type mismatch with reward is the content of this beat. Reward is a number attached to a state and unbounded above. A preference is a probability attached to an observation, and its entries compete for a fixed total, so the agent cannot strengthen one preference without weakening the others.

One variant to name, since the reader will meet it: some papers place preferences over **states** rather than observations, writing a prior over future states and calling it the goal. That form appears widely, including in the robotics literature, and it is not merely a rewriting. Where the difference matters is section 6.

//FIGURE 1: C shown as a distribution over the observation set, beside the grid. Static, nothing moves. Alongside it, a reward vector over cells, so the reader sees the two objects are different shapes: one indexed by observation, one by state. Label which is which. This figure exists to make the type mismatch visible rather than asserted. If cheap, add a third panel showing preferences over states, the variant named in the prose, so the reader can see all three object shapes at once.//

---

**2. The model becomes something to act on.**
$p(o,s)$ from the previous section described. With $C$, the model states which observations the agent expects to receive, and the agent's task becomes closing the gap between that and the world it is in. A model that only describes is inert. A model carrying preferences is something to act on.

---

**3. What it takes to score an action.**
$F$ used an observation already received. An action not yet taken has no observation attached. Unknown quantities are averaged over, so the observation moves inside the expectation. That is the move, stated before the equation arrives.

---

**4. Expected free energy.**
$G$ defined. Each piece identified: the belief predicted under the policy, the observations expected from it, and $p(o \mid C)$.

//FIGURE 2: score one policy by hand, every term shown separately, on the same grid and starting belief as Figure 1. Nothing moves. Each number traceable to its source: the belief, B, A, and C.//

---

**5. Policies.**
A single action cannot show a detour: one step towards a lamp scores worse than one step towards the goal. Define a policy as a sequence over a horizon. Note that the imagined belief spreads at every step, so epistemic value far ahead is worth less than epistemic value now.

//FIGURE 3: three or four policies scored side by side, including one straight to the goal and one detouring via a lamp. Show G for each. The detour should win before the agent has moved. Expose horizon length: at horizon one the detour loses, and the reader should watch it start winning as the horizon grows. If it never wins in the world as built, report that rather than adjusting the world.//

---

**6. The decomposition.**
$G$ splits into epistemic value, how much an observation would tell the agent about where it is, and pragmatic value, how well the expected observations match $p(o \mid C)$. The epistemic term was not added. It appears when the expression is separated, and that is the section's claim.

//FIGURE 4: the table from Figure 3 with G split into its two terms per policy. Same policies, same numbers, one more column. The reader should see which term the detour wins on and which it loses on. No new run.//

---

**7. Softmax.**
From a score per policy to a distribution over policies: negative $G$, exponentiated, normalised. Temperature as the parameter controlling how sharply the best policy is favoured.

//FIGURE 5: the scores from Figure 4 converted to probabilities, temperature exposed. Low temperature gives near-deterministic choice, high temperature approaches uniform. Same table, one more column, nothing re-run.//

---

**8. The demonstration.**
The agent acts: detour to a lamp, localise, then to the goal.

//FIGURE 6: the only figure with motion. Same world, same starting belief. Grid and belief side by side at every step. The reader selects one comparison agent so only two are on screen at once:
— the reward maximiser: same A, same B, same belief, differing only in what it scores. Should walk confidently in the wrong direction, because its belief is broad and it acts as though it were sharp.
— tabular Q-learning: no model, learns from experience. What most readers reach for first.
— nothing: a clean run of the active inference agent alone.
Horizon and temperature remain available, since the reader has met each separately.//

//HONESTY NOTE for whoever builds this, and for the prose: tabular Q-learning assumes the state is observed, so this is not a fair comparison and the text must say so. The point is that the assumption is what breaks under partial observability, not that reinforcement learning cannot solve such problems. Methods carrying belief states or recurrent policies do considerably better. One sentence in the prose with a pointer out; this tutorial does not teach reinforcement learning.//

---

**Payoff.**
The detour was not added to the objective. It was implied by it.

---

**Not in this section:** $C$ in log space, how preference strength shapes behaviour, the failure mode where $C$ is a smuggled reward, the risk and ambiguity grouping, the identity between groupings, and the relation between the outcome and state forms of risk. All section 6. Parameter uncertainty and novelty are section 5.

**Settled for section 6, arising from this:** risk must be written over outcomes, $D_{KL}[q(o \mid \pi) \parallel p(o \mid C)]$, or the two groupings will not give the same number. The state form differs by an expected evidence bound, which is non-negative, so the identity holds exactly only in the outcome form.

**Open before drafting:** the world must be fixed, since all six figures use it. Grid size, lamp positions, goal cell, starting belief.