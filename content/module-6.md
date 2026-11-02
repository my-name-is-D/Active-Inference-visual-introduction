# What the agent was handed

## Four things it never questioned

Module 5 opened with two tables the agent had been handed, *A* and *B*, and ended with the agent learning both: counting what it saw and where its steps led, and choosing where to go partly by how much it expected to learn there.

Those were not the only things handed to it. Four more are still fixed from outside.

It was given:

- a rule for turning scores into a choice: take the lowest. 

- a flat prior over policies, written $E$: every policy started equal, so only its score could set it apart. 

- a fixed **precision** $\gamma$, the number controlling how sharply it favours a better score over a worse one, although under take-the-lowest the precision changes nothing. 

- a starting belief about where it was: flat in modules 1 to 3, three possible cells in module 4, and in module 5 the exact cell it started from.

None of those is wrong. Each is a choice that was made for the agent rather than by it. This module is a refinement: an agent without any of what follows still works, and some implementations leave parts of it out.

## Taking the lowest, or sampling

Module 4 turned the policy scores $G(\pi)$ into a distribution over policies:

$$q(\pi) = \sigma\!\left(-\gamma G(\pi)\right)$$

An interactive widhet in module 4 shows how $\gamma$ reshapes this distribution, but no agent used it to choose a move. The agents in modules 4 and 5 computed their scores and took the lowest; they never sampled from $q(\pi)$.

Taking the lowest score and sampling a policy from $q(\pi)$ are two different selection method, and they are often described as though they were the same thing.

Those selection methods differ most when the scores are close, meaning their gaps are small compared with $1/\gamma$. 

- The greedy rule reads only the ordering, so the gap between the best policy and the next counts for nothing. 

- Sampling reads the gaps: a policy scoring a little worse is selected a little less often, and one scoring much worse is selected almost never. 

Scores are close when the agent's options look nearly equally good. That case, a nearly flat $G$, will present itself later in this module.

<widget id="greedy-vs-sampling"></widget>

Sampling means the agent sometimes picks a policy that does not have the lowest score. The scores come from the agent's beliefs, and those beliefs may be wrong, so a small gap between two scores may not reflect a real difference between the policies.

> **From policies to a move**
>
> $q(\pi)$ gives a probability to each whole policy, but the agents in this tutorial carry out only the first action of a policy and then plan again (a design choice; an agent could carry out the whole policy). Implementations therefore add up the probability of every policy that starts with the same action, which gives a probability for each action, written $P(u)$. A second precision, $\alpha$, can then sharpen these action probabilities: a larger $\alpha$ favours the most probable action more strongly.
>
> $\gamma$ acts earlier, on the scores: it decides how much a difference in $G$ matters. $\alpha$ acts on the action probabilities: it decides how firmly the agent commits to the most probable action. 
>
>In module 4's grid with $\gamma = 0.2$, east gets probability 24% and the other four actions 17% to 20%. With $\alpha = 16$ instead the agent goes east 88% of the time. A large $\alpha$ makes the agent as decisive as taking the most probable action, even when a small $\gamma$ kept the action probabilities close.
>

## Habits

People often do something because they usually do it, not because they have weighed it against the alternatives. An agent can have the same kind of leaning. So far, in our modules, only the score $G$ decided which policies it favoured.

That leaning can be written as a probability for each policy the agent is considering, with the probabilities summing to one. Call this list $E$, the **prior over policies**. A policy with a large entry is one the agent leans towards before any score is computed. A large entry says nothing about whether the policy scores well; it only records the leaning. $E$ joins the score inside the softmax:

$$q(\pi) = \sigma\!\left(\ln E - \gamma G(\pi)\right)$$

Reading it in pieces: $\ln E$ contributes a fixed amount to each policy whatever the situation, $-\gamma G(\pi)$ contributes the scored part, and the softmax turns the total into a probability for each policy. A flat $E$ adds the same amount to every policy and therefore changes nothing.

It is reasonable to ask why an agent that can score its options would want a bias that ignores the score. The scores are computed fresh at each step from a belief that may be wrong, and a habit is a record of what the agent tend to do, which $G$ does not see. The formula also sets how the two are balanced: $\gamma$ scales $G$ but not $\ln E$, so the less the agent trusts its scores, the more its habits decide.

## Habits by counting

$E$ can be given, or learned by the same counting that module 5 applied to *A* and *B*.

To learn it, the agent keeps a positive count for each policy, written $e$ in lower case, as $a$ held the counts behind *A*. The counts are updated once per **trial**: one run of the task, with a fixed number of steps, after which the agent starts again. For that, each policy has to be a sequence of moves covering the whole trial, from the first step to the last, not the short plan the earlier agents dropped after its first step. At the end of the trial, the agent's belief about which of these policies it followed, $q(\pi)$, is added to the counts:

$$e \leftarrow e + q(\pi)$$

So $e$ is a running total. After each trial, every policy's count goes up by the probability $q(\pi)$ gave it: a policy the agent was 70% sure it had followed gets 0.7, one it was 10% sure of gets 0.1. Over many trials, the policies the agent has often followed build up large counts, and $E$ leans towards them.

The estimate is the counts normalised:

$$E_\pi = \frac{e_\pi}{\sum_{\pi'} e_{\pi'}}$$

These are the concentration parameters of a **Dirichlet distribution** over the policy prior, the same object the previous module used for the columns of *A* and *B*. Normalising gives the estimate and discards the size of the counts, and the size is the confidence: once a policy's count is large, adding to it barely changes its share of $E$, while a small count swings with every addition. A habit, in this description, is not a special mechanism. It is a count that has grown large.

$E$ learns posterior policy occupancy, not which policies worked. The update contains no success signal beyond the quantities already used to infer the policy, so repeatedly inferred policies become easier to infer again. It need not equal the frequency of sampled policies.

## Where habits decide

<widget id="habit-vs-score"></widget>

When the agent roughly knows where it starts, as in module 4, the scores differ and a flat $E$ leaves east ahead; raising north's habit only overturns that past the threshold below. When it has no idea where it starts, every score is equal, and any habit decides immediately.

A habit and a score compete on the same scale. Comparing two policies, the habit favours $i$ by $\ln(E_i/E_j)$ and the score favours $j$ by $\gamma\,(G_i - G_j)$, so the habit wins when

$$\ln\frac{E_i}{E_j} > \gamma\,(G_i - G_j).$$

With module 4's one-step scores and $\gamma = 1$, a habit only 2.07 times stronger for north than for east is enough to make north more probable than east. Where the scores are flat, any habit decides. Where they are far apart, or $\gamma$ is large, it takes a strong habit to overturn them.

## Learning the precision

$\gamma$ has been a number set in advance. It is an **expected-free-energy precision**: it scales the scores inside the
softmax, so it decides how much a difference in $G$ matters, and under
take-the-lowest it changed nothing at all. It can be inferred instead.

Treat $\gamma$ as a quantity the agent holds beliefs about, distributed according to a **Gamma distribution** with shape 1. This belief is set by a single number, its rate $\beta>0$, and the expected precision is $\gamma = 1/\beta$. $\beta$ can be read as doubt: the larger $\beta$, the lower the precision the agent expects, and the less a difference between its scores counts.

<widget id="precision-belief"></widget>

The curve shows how strongly the agent believes in each possible value of $\gamma$. With shape 1 it is not a bell: it is highest at zero and decays. The agent uses only its average, $1/\beta$, marked by the vertical line; a larger $\beta$ pulls the belief towards zero, so the average precision falls. The dashed grey curve is the belief before any observation; the update below always starts from it.

The agent has a further thing to update after each observation, alongside its state belief and its counts.

The update is driven by a disagreement between two distributions over policies. The first, written $q_0$, is the one used so far, before the observation: it combines the habits and the scores only.

$$q_0 = \sigma\!\left(\ln E - \gamma G\right)$$

The second, written $q$, is the distribution after the observation: it also takes into account how well each policy explains the observation that has just arrived. That is the variational free energy $F(\pi)$: the surprise of the observation if the agent had taken that policy's first action from its previous belief.

$$q = \sigma\!\left(\ln E - \gamma G - F\right)$$

If the observation favours policies that $G$ already rated well, the two distributions agree. If it favours policies $G$ rated poorly, they disagree. The quantity $(q - q_0)\cdot G$ measures that, and the update is

$$\beta = \beta_0 + (q - q_0)\cdot G$$

with $\gamma = 1/\beta$ and $\beta_0$ the prior rate, the same at every update. This is a fixed-point equation, because both policy distributions depend on $\gamma$; it is iterated to convergence and is valid only while $\beta$ remains positive. When the evidence confirms the ranking, $\beta$ falls below $\beta_0$ and precision rises. When it contradicts the ranking, $\beta$ rises and precision falls. Because each update is anchored to $\beta_0$, $\gamma$ describes the current agreement rather than accumulating without bound.

This requires something the earlier modules did not compute. $F$ has been one number per step, for the agent's belief. Here it is needed per policy. Here, as with habits, a policy spans the whole trial, so each policy's $F$ accumulates the evidence from observations so far. That is the model used here for precision learning.

This also exposes a limit. The earlier agents discard a policy after taking its first action and plan afresh. Newly created future-only policies all share the same past $F$, so $F$ cannot distinguish them: $q = q_0$ and $\gamma$ stays at its prior. Precision learning therefore cannot simply be attached to that shallow replanning rule. The widget contrasts it with a small set of trial-spanning policies.

## Precision does not scale habits

Look again at where the two quantities sit:

$$q(\pi) = \sigma\!\left(\ln E - \gamma G(\pi)\right)$$

$\gamma$ multiplies $G$. It does not multiply $\ln E$.

So raising precision does not make the agent more habitual. It makes the scored part of the total larger while the habitual part stays where it is, and the habit therefore counts for relatively less. An agent that becomes confident in its own planning becomes less governed by what it usually does, and an agent that loses confidence falls back on its habits by default rather than by
decision.

That is why the two belong in the same section. Learned $\gamma$ and counted $E$ are not two independent refinements: one sets how much the other matters.

## What moves the precision

<widget id="precision-update"></widget>

With shallow replanning, the line is exactly flat: the observation cannot distinguish policies that did not exist when it arrived. With trial-spanning policies, informative evidence moves precision up when it supports the ranking by $G$ and down when it opposes it; uninformative evidence leaves precision unchanged.

Precision responds to agreement between two rankings the agent computes itself. It is not a measure of whether the agent's model is correct: a consistently wrong model can still generate internally agreeing $F$ and $G$. Module 4 made the same distinction for the state belief, where confidence and accuracy came apart.

## Where it went to sleep

The agent wakes in the dark with a rough idea of where it went to sleep. That rough idea is $D$, a distribution over states describing where it expects to start, and it is the belief every run in this tutorial has begun from before any observation arrived.

$D$ has always been given: flat in modules 1 to 3, three possible cells in module 4, the exact starting cell in module 5. None of these was worked out by the agent.

Now suppose the agent does not sleep just anywhere. It has a few cells it tends to settle in, so that across many nights the cell it wakes in is not drawn uniformly from the grid. Nothing about the agent changes: this is a fact about the world, and the agent has no access to it beyond what it can infer from waking up repeatedly.

The question this raises is whether an agent that tends to start in the same place should keep waking equally unsure.

## Counting where you started

$D$ is a distribution over states, so it can be counted like any other. Keep one positive number per cell, add to them at the end of each run, and normalise:

$$D_s = \frac{d_s}{\sum_{s'} d_{s'}}$$

What to add is the question. The obvious choice is the belief the agent held at the start:

$$d \leftarrow d + q(s_0)$$

but that belief is the prior it walked in with, unchanged by any observation, so adding it teaches the agent only what it already assumed. The counts would reinforce the prior indefinitely.


The useful estimate comes from hindsight. Module 5 introduced **smoothing** ("Letting a later observation clarify an earlier step"): a backward pass revises the belief about an earlier step using observations that arrived after it. Module 5 limited the pass to a five-step window so that counts could be updated while the agent moved. *D* is updated only once per run, after the run ends, so the pass can cover the whole run. The starting state is the extreme case: any observation in the run can bear on where the agent began, and none of them had arrived when it was there. How much they reveal depends on the readings. A start that nothing later tells apart stays as uncertain in hindsight as it was at the time. So the contribution is the smoothed belief:

$$d \leftarrow d + q(s_0 \mid o_{1:T},u_{1:T})$$

In words: **count where the run says you started, not where you assumed you started.**

This update is an approximation. The agent does not know where it started, so instead of adding one count to the true starting cell, it shares that count across cells in proportion to its smoothed belief. Module 5 called this an **expected-count approximation** and used it for *A* and *B*: it is exact only when the state is known.

## Fixed, given, and learned

<widget id="starting-prior"></widget>

In the verification runs, the given $D$ raises belief on the true cell at step 1 from 0.049 with a flat prior to 0.340. A learned $D$ lies between them initially, but can later fall below the flat prior because the observations identify some starting cells better than others. After the sleeping cells move, the old learned prior remains worse than the flat prior throughout the tested 60 nights.

The given $D$ is the control. It is what a correct prior is worth, and therefore what learning is aiming at rather than what learning achieves.

This comparison has two limits. $D$ is used once, as the belief before the first observation, so a correct $D$ helps most at the start, and its advantage fades as informative observations accumulate. A learned $D$ can do worse than a flat one, even while the habit holds: it learns only the starts that later readings can identify, and on nights that begin elsewhere it points the agent to the wrong cells. If the agent's habits change, the learned $D$ still points to the old starting cells, so the agent starts confident about the wrong place, while an agent with a flat $D$ starts uncertain but not wrong. Correcting $D$ also takes time: adding one run's count to many earlier ones changes $D$ little. Module 5 noted the same limit for *A* and *B*: a strong incorrect prior can slow correction, and nothing in the model discounts old evidence.

## Two remarks

**Precision elsewhere.** The $\gamma$ above is a precision on expected free energy. Related constructions place precision on other parts of the model: an inverse temperature on the transitions, and another on the likelihood, each expressing how much the agent trusts that part. Module 5 learned the counts behind *A* and
*B* without ever asking how much the resulting tables should be trusted; Parr and Friston take up that question. As with the expected-free-energy precision, and unlike the state updates, the normalising constants depend on the parameters being updated and cannot be dropped. See [Parr and Friston, *Uncertainty,
epistemics and active inference*](https://royalsocietypublishing.org/doi/10.1098/rsif.2017.0376).

**Preferences that change with time.** $C$ has been the same at every step, and the policy score summed $G(\pi,\tau)$ over the horizon with that constant $C$ inside each term. That is a choice. A $C$ indexed by $\tau$ expresses a preference that applies only at some steps of the horizon, for example only at its end. Nothing in the score needs altering, only which $C$ each term reads.

## What is still handed over

The agent began this module with four things chosen for it by whoever built it, not by the agent itself. One has been replaced by a different design choice, and three are now learned from experience:

- **The rule for turning scores into a choice.** Before: take the lowest score. Now: sample a policy from $q(\pi)$. This is still chosen for the agent, not learned by it.
- **The prior over policies, $E$.** Before: flat. Now: counts built from the posterior distribution over trial-spanning policies, added once per trial.
- **The precision, $\gamma$.** Before: a fixed number. Now: a belief about $\gamma$, updated as observations agree or disagree with the ranking given by $G$.
- **The starting belief, $D$.** Before: flat in modules 1 to 3, three cells in module 4, the exact starting cell in module 5. Now: counts of where, in hindsight, each run started, added at the end of the run.

Two things did not change. The preferences $C$, introduced in module 4, are still written by hand. The agent can act without them, as it did with a uniform $C$ in modules 4 and 5, but nothing in this tutorial learns them. 

And the shape of the model is still ours. The agent learns what is in each column of *A* and *B*; it never learns how many columns there should be. That one cell is one state, that there are twenty-five of them, that there are five actions: all of that was decided before the agent ran. So an agent that has learned how a step behaves in one cell learns it again from nothing in the next, because it has no way to hold a rule that applies across cells.

That is a limitation of this tutorial rather than of the framework. Discrete active inference has two complementary approaches to it: **Bayesian model reduction**, which merges states when a model with fewer of them explains the observations more parsimoniously, and **Bayesian model expansion**, which adds a state when the larger model explains the observations better once its extra complexity is counted (Friston et al., 2023). An earlier approach reserves unused states in advance and engages them when new patterns recur (Smith et al., 2020). Neither is demonstrated here.

Those are where the next section starts.

## To go further

- Smith, Friston and Whyte, [“A step-by-step tutorial on active inference and
  its application to empirical
  data”](https://www.sciencedirect.com/science/article/pii/S0022249621000973).
  The accompanying scripts include one that simulates precision updates through
  the prior $\beta$: set the policy prior, the expected free energies and the
  variational free energies after a new observation, and watch the resulting
  changes in $\gamma$ over successive
  updates.

- Parr and Friston, [“Uncertainty, epistemics and active
  inference”](https://royalsocietypublishing.org/doi/10.1098/rsif.2017.0376),
  for precision parameters on the transitions and the likelihood rather than on
  policies.

- Sajid et al., [“Active inference: demystified and
  compared”](https://arxiv.org/abs/1909.10863), for the precision update
  written as a gradient rather than as a discrete step.

- pymdp, [source on GitHub](https://github.com/infer-actively/pymdp). In `control.py`, `update_posterior_policies` forms $q(\pi)$ from $\gamma$, $G$ and $E$; `sample_action` adds up the policies by first action and applies $\alpha$ as a power, with taking the most probable action as the default; `sample_policy` samples a whole policy instead. In `agent.py`, $\gamma$ is a fixed parameter and parameter learning covers *A* and *B*; the expected-utility function in `control.py` accepts a $C$ indexed by planning step.

- SPM, [`spm_MDP_VB_X.m`](https://github.com/spm/spm/blob/main/toolbox/DEM/spm_MDP_VB_X.m), the MATLAB implementation module 6 is compared against: the action precision `alpha` applied as `spm_softmax(alpha*log(Pu))`; the precision update through `qb`, `beta` and `dFdg`, which restarts from the prior `beta` at each step; and the habit counts, `MDP.e = MDP.e + eta*u(:,T)`, added once per trial.

- Friston et al., [“Supervised structure learning”](https://arxiv.org/abs/2311.10300), for Bayesian model expansion: the recipe for growing a model state by state, and the comparison of the augmented model's free energy with the original's.

- Smith, Schwartenbeck, Parr and Friston, [“An active inference approach to modeling structure learning: concept learning as an example case”](https://www.frontiersin.org/articles/10.3389/fncom.2020.00041/full), for reserve hidden states that are engaged when new patterns recur, and for Bayesian model reduction resetting states that turn out to be redundant.
