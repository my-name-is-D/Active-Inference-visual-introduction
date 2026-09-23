# What the agent does not know about the world

## Two tables, never questioned

*A* said what each cell looks like; *B* said where each step leads. The agent has read them, trusted them.

That was a choice, and it has been doing a lot of work. Every belief the agent has held was computed from those two tables. Every score it has assigned to a policy was computed from them too. Errors in those tables can distort inference and planning. Unexpected observations can expose a poor prediction, but the agent so far has no rule for updating the tables themselves.

This page removes that.

Learning the two tables does not pose the same problem. To learn *A*, the agent must work out what it sees at a place. To learn *B*, it must work out both where a step began and where it ended. But *B* is also what carries its position belief through that step. Learning *B* therefore depends on localisation, while localisation already depends on *B*.

We will separate those dependencies in three comparisons:

1. learn *A* while *B* is known;
2. learn *B* while *A* is known, first at night and then by day;
3. learn *A* and *B* together.

And we will introduce two new changes in the world, a part of the world in under work, thus changing the observation to expect there, and adding walls to an environment that did not have any. Moreoever, this world that has known only darkness will also know daylight now, which decreases the aliasing of the environment, the dead-reckoning of the agent and improve localisation overall.

## Two kinds of not knowing

The agent has been uncertain since the first page, but always about the same thing: which cell it occupies. A step spreads this uncertainty, an observation sharpens it, and it is different at every moment.

That is **state uncertainty**, and the whole tutorial so far has been about it. The new kind of uncertainty we will see is about **parameter uncertainty**, because *A* and *B* are the parameters of the agent's model. They will no longer be fixed and assumed known.

Those two behave differently. State uncertainty can be lost and regained: a lost agent becomes less lost by looking and more lost by walking without landmarks. Parameter uncertainty does not move that way. The agent reduces its uncertainty by moving somewhere and looking, and walking away does not make it forget what a cell looks like.

## Learning by counting

Suppose the agent does not know one column of *A*: there is a cell it has no information about.

It can find out by standing there and looking. Seeing lit from that cell is evidence the cell is lit. Seeing it again is more evidence. Seeing dark adds evidence for a higher probability of dark observations from that cell. Learning this column alone does not distinguish sensor noise from a change in the cell itself.

So the agent keeps a count. One number for each state-observation pair, and when it observes $o$ while believing it is at $s$, it adds to the count for that pairing:

$$a \leftarrow a + o \otimes q(s)$$

Here $o$ is the one-hot vector for the observation, $q(s)$ is the state belief after incorporating that observation, and $\otimes$ is the outer product

The positive counts $a_{\cdot s}$ are the concentration parameters of a **Dirichlet distribution** over the possible columns $A_{\cdot s}$. Larger counts make that distribution narrower. The Dirichlet is useful here because observing outcome $o$ simply adds one to $a_{os}$. Even before any data, every
outcome needs a positive pseudocount, so “no observations yet” means using a prior, not an all-zero column.


<widget id="outer-product"></widget>

Write $a$, in lower case, for the counts, and keep *A* for the observation model the agent plans with. They are different objects: $a$ grows without limit and carries the confidence, *A* is the columns of $a$ normalised. The literature uses the same convention, and it uses $b$ and *B* the same way. To not confuse with the action $a$ (sometime $u$), not used in the same context.

The addition is weighted by the belief, because the agent is not certain where it is. If its belief is spread across four cells, the count is shared among them considering its confidence over each cell being its current localisation. This fractional update is an expected-count approximation. When the state is known, adding 1 to its column is exact conjugate learning; with an uncertain state, the exact parameter posterior is generally a mixture of Dirichlets rather than a single Dirichlet for each column.

The estimate of *A* is the counts, normalised:
$$A_{ij} = \frac{a_{ij}}{\sum_k a_{kj}}$$

where $A_{ij} = p(o = i \mid s = j)$, as in page 1: $i$ the observation, $j$ the state.

The loop is **predict → move →see → update**. For the chosen action, the agent first predicts where it will be and what it may observe, using its current *A* and *B*. It executes the action and receives the observation. Only then does it update its state belief, use that updated belief to update the counts, and normalise them for the next prediction. For *B*, the count update uses the joint belief about the previous and current states after the new observation.

The agent learns after each observation,. The current observation is incorporated into the state belief using the model available before that observation; the resulting count update prepares the model for the next prediction.

## The counts carry two things

Normalise a column and you get the estimate. What normalising throws away is the size of the numbers.

Compare two columns. One holds the counts $[0.1, 0.4]$. The other holds $[20, 80]$. Normalise each and they are identical: $[0.2, 0.8]$, lit with probability 0.8 in both cases.

Now add one observation of dark to each. The first column becomes $[1.1, 0.4]$, which normalises to $[0.73, 0.27]$. The second becomes $[21, 80]$, which normalises to $[0.21, 0.79]$.

Same estimate, entirely different response to evidence. That difference is confidence, and the agent keeps no separate record of it. It is the size of the counts.

That is the property the rest of this page is built on. A column the agent has visited a hundred times barely moves. A column it has never visited swings on a single observation.

For a state $s$, normalise its counts to obtain the corresponding column of
the observation model:

$$A_{os}=\frac{a_{os}}{\bar a_s},
\qquad \bar a_s=\sum_o a_{os}.$$

$A_{os}=p(o\mid s)$ is the estimated probability of observation $o$ in state $s$. The total $\bar a_s$ is the confidence in that estimate: larger counts make the column harder to change. In the figure, **expected novelty** means how much the next observation is expected to teach the agent about this column, averaged over the possible next dark or lit observation.

<widget id="counts-confidence"></widget>


## Novelty

We previously measured the expected change in the agent's belief about its **state**. Novelty applies the same construction to its belief about the **model**. If $\theta$ denotes the uncertain observation array, then

$$\text{novelty} = \mathbb{E}_{q(s_\tau \mid \pi)\,p(o_\tau \mid s_\tau)}\Big[D_{\mathrm{KL}}\big[\,q(\theta \mid o_\tau, s_\tau) \;\|\; q(\theta)\,\big]\Big]$$

For a hypothetical state $s_\tau$, observing $o_\tau$ adds one to the corresponding Dirichlet count. The KL divergence measures that update, and the expectation averages it over the states and observations predicted under the policy. This is **expected parameter information gain**. Equivalently, it is conditional mutual information between the model parameters and the next observation; the next section writes the same quantity as a difference of entropies.

This form conditions on knowing which state generated the observation. The acting agent does not know that state and therefore spreads its count update according to its state belief. Novelty is consequently a planning score under the factorisation $q(s_\tau\mid\pi)q(\theta)$, not the exact KL change caused by the agent's later fractional update.

## From one column to a policy

The calculation follows the idea directly. For each possible observation $o$:

1. make a temporary copy of the column counts;
2. add one to the count for $o$;
3. measure the KL divergence from the updated Dirichlet belief to the current one;
4. weight that change by the predicted probability $p_o=a_o/\bar a$.

Adding those weighted changes gives the novelty $N(a)$ of one column. A policy may lead to several states, so its novelty is the corresponding weighted average:

$$N_A(\pi,\tau)=\sum_s q(s_\tau=s\mid\pi)N(a_{\cdot s}).$$

This is the quantity used in code: first score every column the policy might visit, then weight those scores by the predicted state belief. Low count totals generally make a column easier to change and therefore more novel. Large counts make it less novel, whether those counts describe the world correctly or not. Novelty measures opportunity to learn, not model error.

<widget id="novelty-branches"></widget>

In words: **novelty = probability of each possible observation × how much that observation would change the model**.

The low-count and high-count examples predict the same observations: 20% dark and 80% lit. Only their count totals differ. The high-count model changes less after either observation, so its novelty is lower. **A more established model offers less expected learning.**

> **Optional: exact novelty for a Dirichlet column**
>
> Adding one to entry $o$ gives the following KL divergence:
>
> $$D_{\mathrm{KL}}\!\left[\mathrm{Dir}(a+e_o)\,\|\,\mathrm{Dir}(a)\right]
> =\ln\frac{\bar a}{a_o}+\psi(a_o+1)-\psi(\bar a+1),$$
>
> where $\bar a=\sum_o a_o$, $e_o$ selects the observed entry, and $\psi$ is the digamma function. Averaging over the possible observations gives
>
> $$N(a)=H(p)+\sum_o p_o\left[\psi(a_o+1)-\psi(\bar a+1)\right],
> \qquad p_o=\frac{a_o}{\bar a}.$$
>
> The same quantity can be written as
>
> $$N(a)=H(p)-\mathbb{E}_{\vartheta\sim\mathrm{Dir}(a)}[H(\vartheta)].$$
>
> The first term is uncertainty about the next observation. The second is the observation noise expected if the true column $\vartheta$ were known. Novelty is the part of predictive uncertainty that learning the column can resolve.
>
> This also shows that $0\leq N(a)\leq\ln K$ for a column with $K$ possible observations. At fixed proportions, novelty approaches zero as the count total grows. The commonly used large-count shortcut
>
> $$\frac{1}{2}\left(\frac{1}{a_o}-\frac{1}{\bar a}\right)$$
>
> can be inaccurate when any individual count is small, so the calculations on this page use the exact expression.

## The third term

Novelty is a number attached to a policy, so it goes into the score.

$G$ had two terms. It now has three:

$$G(\pi, \tau) = \underbrace{-\,\mathbb{E}_{q(s_\tau\mid\pi)p(o_\tau\mid s_\tau)}\big[D_{\mathrm{KL}}[\,q(\theta \mid o_\tau, s_\tau) \,\|\, q(\theta)\,]\big]}_{\text{novelty}} \;  \underbrace{-\,\mathbb{E}_{q(o_\tau\mid\pi)}\big[D_{\mathrm{KL}}[\,q(s_\tau \mid o_\tau, \pi) \,\|\, q(s_\tau \mid \pi)\,]\big]}_{\text{epistemic value}} \;\underbrace{-\,\mathbb{E}_{q(o_\tau \mid \pi)}\big[\log p(o_\tau \mid C)\big]}_{\text{pragmatic value}}$$

Nothing about the other two has changed. The agent still scores what a policy would deliver and what it would reveal about its position, and now also what it would reveal about its model.

The signs work the same way throughout. A divergence is never negative, so the novelty term is never positive: zero for a policy that would teach the agent nothing about its model, negative for one that would teach it something. Like the epistemic term, it can only lower $G$.

Three terms, three reasons to prefer one policy over another, and the agent adds them up and takes the lowest. This score uses a coefficient of one for each term. All are measured in nats, but sharing units does not require equal coefficients or guarantee comparable magnitudes.

As evidence accumulates, familiar columns often offer less to learn. This can favour exploration early and pursuit of preferences later, but it is not a guaranteed sequence. Contradictory evidence can reopen a question the agent considered settled:

| | Before | After one dark observation |
|---|---:|---:|
| Counts `[dark, lit]` | `[0.01, 10]` | `[1.01, 10]` |
| Predicted dark | 0.1% | 9.2% |
| Novelty | 0.004 | 0.038 nats |

Before the observation, the agent is almost certain that this state produces lit. One dark observation weakens that certainty, making the next observation more informative. Novelty therefore rises rather than falls.

Policy novelty also depends on where the policy is expected to take the agent. It can change because the counts change, because the predicted route changes, or both.

## Learning *A* while *B* is fixed

We first isolate observation learning. The movement model is known and fixed at reliability 0.8. It still leaves some uncertainty about whether a command succeeded, so the agent does not always know which cell produced an observation.
The agent knows about the world and expect transition between cells, however it does not know what to learn from the top left area of the world were works (and walls) have been built on.

Each unknown observation over state has the same total prior concentration, 0.5, divided equally among its $K$ possible observations:

$$a_{os}=\frac{0.5}{K},\qquad s\in\text{works}.$$

Thus, at night, the uncertain column starts with pseudocounts \([0.25,0.25]\), whose total concentration is \(0.5\). By day, the same total concentration is divided equally among six possible observation categories, so each pseudocount is \(0.5/6=1/12\). Daylight provides more distinguishable observations without giving the agent a more confident prior.

At night the sensor distinguishes lamps from dark cells. By day it also distinguishes Home, the grey path, green cells and the shared W works marker.

We compare three ways of choosing actions. The random-walk baseline does not use expected free energy $G$ to choose. At every step it selects each of the five actions (up, down, left, right or stay) with equal probability:

$$q(u_t=u)=\frac{1}{5}$$

To isolate the change in action selection, **AIF without *A* novelty** uses the expected-free-energy score from the previous page. It values information about position, but not information about *A*:

$$G_4(u)=\text{pragmatic cost}(u)-\mathrm{IG}_{s}(u).$$

**AIF with *A* novelty** adds the new term:

$$u_t=\operatorname*{arg\,min}_{u}G(u),$$

$$G_5(u)=\underbrace{\text{pragmatic cost}(u)}_{\text{preferred observations}}
-\underbrace{\mathrm{IG}_{s}(u)}_{\text{learn position}}
-\underbrace{\mathrm{IG}_{A}(u)}_{\text{learn }A}.$$

Preferences are uniform here, so pragmatic cost is the same for every equal-length policy. The choice is governed by information about position and information about *A*. After the chosen action, all three agents use the same
**predict → move → see → update** rule.

<widget id="smoothing-comparison"></widget>

In the figure above, at each step the agent acts, receives one sensor observation, and updates its beliefs and model. The coloured number beside each curve gives its cumulative physical visits to the works cells at that point in time.


Read the curves from 100% = starting error towards 0% = model learned. Lower is better. The coloured number is how many times the agent visited the works cells.

The middle curve is a control: the original agent used in the previous page held *A* fixed, whereas this control updates *A* after every observation.
Both AIF conditions use the same update equation. The only difference is that one includes $-\mathrm{IG}_A$ in its action score which directs action towards uncertain observation columns.

**At night: learn dark vs lit**. Each works column starts uncertain between these two observations. Because the works cells actually appear dark, a falling curve means that the agent is learning their dark-observation probability.

**By day: learn what kind of place this is**. Daylight adds the W observation category, so the day condition tests whether the agent learns that these cells belong to the works area. Active inference visits the works more often, but its error does not finish lower. The path maps show why: its visits concentrate on some works cells, while all three produce the same W observation. Seeking information does not guarantee balanced evidence.


## Letting a later observation clarify an earlier step

In the figure above, we presented the impact of **smoothing** over 5steps. Here is the explanation.

At step 1, the agent observed dark but was equally uncertain between two cells. By step 5, a lit observation at a recognisable landmark makes the earlier route through cell 1 more likely:

```text
When dark arrived                 After the later observation
Cell 1: 50%   Cell 2: 50%   →    Cell 1: 90%   Cell 2: 10%
```

The dark observation has not changed. What changes is the agent's belief about where it occurred.

<widget id="smoothing-counts"></widget>

Smoothing replaces the earlier count contribution; it does not add the observation again. The total dark evidence remains one count, but more of it is now assigned to cell 1.

The mathematical change is that the state at step 1 is reconsidered using the complete five-step observation history:

$$\underbrace{q(s_1\mid o_1)}_{\text{belief after the dark observation}}
\quad\longrightarrow\quad
\underbrace{q(s_1\mid o_{1:5})}_{\text{same state, revised after step 5}}$$

The count rule keeps the same form:

$$\text{ordinary contribution}=o_1\otimes q(s_1\mid o_1)$$

$$\text{smoothed contribution}=o_t\otimes q(s_t\mid o_{1:t+n})$$

In words: **count contribution = observation × belief about where it occurred**.

For a smoothing window of length $L$, the implementation is:

1. store the action, observation and model used at each recent step;
2. run a backward pass through the window to revise the earlier state beliefs;
3. replace the recent count contributions using those revised beliefs;
4. when a contribution leaves the window, add it once to $a_{\mathrm{frozen}}$.

The agent still acts after every observation. A five-step window means that the latest five contributions remain revisable; it does not mean waiting five steps before acting.

> **Optional: the backward pass and frozen counts**
>
> Start at the newest state with a backward message of ones. For each saved step, moving from newest to oldest:
>
> 1. multiply its observation likelihood by the backward message;
> 2. combine that result with its saved transition matrix and earlier filtered belief;
> 3. normalise to obtain the revised joint transition belief;
> 4. sum over origins to obtain the revised state belief;
> 5. carry the message one step backwards through the transition matrix.
>
> After the pass, rebuild the counts as
>
> **current counts = frozen older counts + revised contributions inside the window**
>
> The calculation is exact for the saved *A* and *B* inside that window. Because those models are themselves learned through expected counts, it is not exact joint inference over states and parameters.

<!--No oracle on this A-learning chart. It is plotted on the B-learning
comparison instead, where the ceiling it sets is the point. It assigns each
observation to the true physical cell, whereas both agents must use their
inferred state belief: a diagnostic ceiling, not an action-selection strategy
available to the agent. -->


## Learning where steps lead

The agent has learned what observations to expect in each state. It can use the same counting method to learn where its actions lead. One column of *A* asks, “What will I observe in this state?” One column of *B* asks, “If I start in state $j$ and take action $u$, where will I arrive?”. (Earlier lessons used \(a\) for an action. From this page onward, \(a\) denotes the counts underlying A, so we write \(u\) for an action.)

As in lesson 2, *B* allows a commanded step to end somewhere other than its intended destination. These probabilities describe what the agent believes may happen. In the world itself, a step goes exactly where it was aimed.

Except in the upper corner, where the works have put up walls. There, some steps that used to go through now leave the agent where it was.

So the agent's *B* is wrong in two ways at once. Everywhere, it is too pessimistic about its own movement. At the walls, it is too optimistic, expecting to succeed where there is now a wall blocking its step.

For *B*, each column describes a destination state $i$, given an origin $j$ and an action $u$. Keep positive counts $b_{iju}$ and normalise over destinations:

$$B_{iju}=\frac{b_{iju}}{\sum_k b_{kju}}.$$

If both states are known, add one to the entry for the observed transition. When they are uncertain, use their joint posterior after the new observation:

$$b_{iju}\leftarrow b_{iju}+q(s_t=i,s_{t-1}=j\mid o_{1:t},u_{1:t-1}),\qquad u=u_{t-1}.$$

The agent takes action $u$ and receives one dark observation. Its belief about where it started spreads over pairs of origin and destination, $q(s_t=i,s_{t-1}=j\mid o_{1:t},u)$:

<widget id="joint-transition"></widget>

With a smoothing window, the same replacement applies to recent transitions. The ordinary contribution to $b_{iju}$ at step $t$ is the joint belief given the observations available up to that point:

$$\text{ordinary contribution}=q(s_t=i,s_{t-1}=j\mid o_{1:t},u_{1:t-1})$$

A window of $n$ further steps widens the conditioning the same way it did for *A*, from $o_{1:t}$ to $o_{1:t+n}$:

$$\text{smoothed contribution}=q(s_t=i,s_{t-1}=j\mid o_{1:t+n},u_{1:t-1})$$

Each contribution goes into the slice for the action $u_{t-1}$ taken on that transition. As with *A*, the earlier contribution is replaced rather than counted again.

This is again an expected-count approximation. Multiplying separate state marginals is a further approximation: uncertainty about where the agent started and where it ended is generally correlated. Knowing only the destination does not identify which transition column to update.

In implementation terms:

**transition counts = old transition counts + joint belief about the transition**

$$b_{\cdot ju}\leftarrow b_{\cdot ju}+q(s_t=\cdot,s_{t-1}=j\mid o_{1:t},u_{1:t-1})$$

**transition model *B* = transition counts normalised over destinations**

$$B_{iju}=\frac{b_{iju}}{\sum_k b_{kju}}$$

Transition novelty then repeats the calculation already used for *A*: the same four steps, applied to a column $b_{\cdot ju}$ instead of $a_{\cdot s}$, give its novelty $N(b_{\cdot ju})$. A policy weights that novelty by the probability of leaving from each origin under each action it takes:

$$N_B(\pi,\tau)=\sum_j q(s_{\tau-1}=j\mid\pi)N(b_{\cdot ju}),\qquad u=u_{\tau-1}.$$

**transition novelty = probability of each possible transition × how much that transition would change *B***

> **Optional: exact transition novelty**
>
> Conditional transition novelty treats the origin and destination as known for each hypothetical transition, then averages over the transitions predicted under the policy. With $\theta_B$ denoting the uncertain transition array,
>
> $$N_B(\pi,\tau)=\mathbb{E}_{q(s_{\tau-1}\mid\pi)B_{s_\tau,s_{\tau-1},u}}\left[D_{\mathrm{KL}}\left[q(\theta_B\mid s_\tau,s_{\tau-1},u)\,\|\,q(\theta_B)\right]\right],\qquad u=u_{\tau-1}.$$
>
> For independent Dirichlet columns and the same factorised planning approximation, this is $\sum_j q(s_{\tau-1}=j\mid\pi)N(b_{\cdot ju})$. The column formula is unchanged, but its outcomes are destination states. As with *A*, this conditional planning score assumes access to hypothetical states; it need not equal the parameter information recoverable later from noisy observations.

## What a wall needs before it can be learned

We now reverse the experiment: the observation model is known, and the transition model must learn the two boundary walls.

A wall is an obstacle that prevents a step from reaching the next cell. The agent never sees the wall itself. It only observes the cell beneath it, so a wall appears as a step that left the agent where it started. Detecting one requires a sensor that can distinguish the starting cell from the intended destination, and a transition model that admits staying put as a possible outcome.

The sensor condition is the difference between night and day:

```text
Night:  dark ── wall ── dark     blocked and successful steps look alike
Day:       W ── wall ── green    blocked → W, successful → green
```

For each of the four movement commands—up, down, left and right—the prior assigns probability 0.8 to reaching the neighbouring cell and probability 0.2 to remaining in the current cell:

$$B^0_{i s u}=\begin{cases}
0.8,&i=\operatorname{move}(s,u),\\
0.2,&i=s,\\
0,&\text{otherwise.}
\end{cases}$$

There is one exception: the stay command, which leaves the agent in its current cell with probability 1.

## Three action rules, one learning rule

Learning a wall is not like learning what a cell looks like. Every step produces an observation, so *A* gains evidence constantly. A wall, in this simple world, only reveals itself when the agent walks into it, and there are just four blocked directions among a 125 state-action columns. Wandering at random, the agent tests a wall on roughly one step in thirty. In real situations, the obstacles could be detected as observations, through specific sensors, but we will not cover that here.

The comparison changes how actions are chosen, not how *B* is updated. All three agents use the same state belief, transition counts and update equation.

The **random walk** gives every action equal probability:

$$q(u_t=u)=\frac{1}{5}.$$

The second agent is Active Inference without transition novelty. It scores two-step policies using only expected information about its state:

$$G_{\text{without }N_B}(\pi)=\sum_{\tau=1}^{2}-\mathrm{IG}_s(\pi,\tau).$$

The third adds expected information about *B*:

$$G_{\text{with }N_B}(\pi)=\sum_{\tau=1}^{2}\left[
-\mathrm{IG}_s(\pi,\tau)-N_B(\pi,\tau)\right].$$

Preferences are uniform, so their pragmatic cost is identical for every two-step policy and is omitted. The controlled comparison is therefore:

```text
random walk              no planning score
AIF without B novelty    state information only
AIF with B novelty       state information + transition information
```

As in the *A*-learning figure, the curve measures mean total-variation error over the columns being learned. Here those are the *B* columns whose origin lies in the works area. **Lower = a more accurate transition model.** Wall probability and blocked attempts remain in the readout as diagnostics.

<widget id="b-learning-comparison"></widget>

### What the comparisons teach

**Night vs day tests whether the evidence exists.** At night, none of the strategies reduces error in the works-area transition columns despite encountering blocked moves. Exploration cannot recover a distinction absent from the observations.

**Random walk vs AIF without $N_B$ tests what state-information planning changes.** By day, the state-information planner produces a more accurate works-area transition model than the random walk. Seeking observations that localise the agent also brings it repeatedly to the distinctive works boundaries.

**AIF without vs with $N_B$ isolates transition novelty.** Both model learn rather well because state information gain already leads both agents towards the informative works boundaries. Adding $N_B$ often leaves the chosen action unchanged; when it does change the action, it may favour another uncertain transition outside the works area measured here. Transition novelty therefore changes where evidence is gathered, but does not guarantee lower error in one chosen region of *B*.

## Why the wall cannot be learned at night

The model predicts that the step probably succeeds. Because both sides look dark, the observation does not contradict that prediction, and the agent concludes that it probably moved.

The small chart presented above isolates what happens after a blocked command. In both conditions, the agent tries 5 times to move through the same wall, while its true position remains in the works cell. At night, the works cell and the intended destination both look dark, so the observation cannot correct the agent's prediction that it moved. Its belief in its true position therefore falls. By day, W means that it stayed and green would mean that it crossed the wall, so each observation confirms its true position.

This is the circularity from the start of the page. The agent needs a useful *B* to localise, but it also needs localisation to assign evidence to the correct column of *B*. Smoothing can reassign evidence that later observations clarify; it cannot create a distinction absent from the complete observation history.

## Learning *A* and *B* together

The previous experiments isolated the two learning problems. We now remove that simplification: the daylight agent learns observations and transitions at the same time. Home remains known as an anchor, while progressively more of the rest of the map starts uncertain:

```text
Works only              3 uncertain states
Larger region           the upper-left 3 × 3 region
Whole map               every state except Home
```

At each uncertain state, the agent must learn one observation column of *A* and the four movement columns of *B*. Every uncertain column starts with the same prior strength. The amount of uncertainty changes because there are more columns to learn, not because each column is made less certain. The agent plans with all three information terms:

$$G(\pi,\tau)=-\mathrm{IG}_s(\pi,\tau)-N_A(\pi,\tau)-N_B(\pi,\tau).$$

The two learned models support one another:

```text
learning A improves localisation
            ↓
better localisation improves learning B
            ↓
better B improves predictions of future states
            ↓
better predictions help the agent choose where to learn A
```

<widget id="joint-learning"></widget>

Both errors fall in all three conditions. Active Inference can therefore learn *A* and *B* together rather than requiring one of them to be known first. When more of the map is uncertain, evidence must be distributed across more columns, so a larger fraction of the initial error remains after the same number of steps.

This result depends on usable observations, repeated access to the unknown parts of the world, and priors that allow the true outcomes. The night experiment showed what happens when the observations cannot distinguish the alternatives.

<!--The B comparison data is generated by
.scratch/lesson-5-verification/b_fixed_a.py. All conditions use move-or-stay
support, the same count update and seeds. They vary the sensor and whether
action selection is random, uses state information gain alone, or also uses
transition novelty.-->

## Two unknowns, one machinery

The agent began this page trusting two tables. It ends it holding beliefs about both.

Nothing was added to make it curious about its model. The novelty term was not invented and bolted on; it is the same divergence the previous page produced, pointed at a different unknown. Where the epistemic term asks what an observation would tell the agent about where it is, novelty asks what it would tell the agent about what the world is like, and the two are the same question with a different subject.

That is the claim this page was built to make. There are two things an agent can fail to know, and the framework does not treat them as different problems. It writes down what it does not know, scores actions by how much of that they would resolve, and lets the arithmetic decide what to do first.

**Active inference can direct limited observations towards what the agent expects to learn from, while also considering localisation and preferences**. This is
especially useful when:

- the world is large and random coverage is expensive;
- informative locations are sparse;
- observations are costly;
- the agent must learn while pursuing a goal;
- uncertainty about its current state matters;
- different actions reveal different parts of the model; or
- sampling every state-action pair uniformly is impractical.

Its advantage depends on whether the model, sensor, planning horizon and learning metric make the relevant information identifiable and useful.

Novelty does not directly measure whether the model is wrong. Identical counts give identical novelty whether or not they describe the world accurately. Large counts can therefore make a mistaken column look unpromising to revisit. Contradictory observations can still correct finite counts through the updates already given, provided the agent gathers evidence and assigns it to the relevant states. What this page does not provide is a dedicated mechanism for detecting a changed world or discounting old evidence.
