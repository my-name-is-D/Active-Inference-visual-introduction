# Module 2. Moving

## The agent has been standing still

Everything so far has happened to an agent that did not move. It looked, it revised what it held, it looked again, and each reading sharpened its sense of where it was. 

That is of limited use. If several cells produce the same reading, every one of them explains it equally well, and no amount of further looking will separate them. The agent has to move.

The moment it moves, something happens that nothing so far accounts for. Its belief describes where it was, and it is now somewhere else.

Anyone who has walked across a dark room knows the effect: uncertain steps can leave you less sure of where you are. An informative observation may help you locate yourself again. What follows is how the agent accounts for both.

## Two ways a belief can change

Keep the two causes apart, because they are different operations and the agent applies them at different moments.

The first is the one already covered. Something arrives. An observation reaches the agent from outside, and the agent revises what it holds to account for it. The agent is passive here: it receives, and it responds.

The second is new. The agent takes a step, and its belief has to move with it: if it was probably on cell 3 and stepped right, it is now probably on cell 4. Nothing has been observed yet. This change comes from the agent's prediction of where the step leads, not from anything the world told it.

Both operations act on the same belief, and that belief is over states: a probability for each cell the agent might be on. Starting now, the agent decides on an action and forms a belief about where that action would put it, then takes the step, observes from wherever it has arrived, and revises that belief with what it saw. The first belief is **the prior** for this turn, the revised one, after observation, **the posterior**.

For now, the agent keeps a single belief about its current cell, and each cycle overwrites the last. That is the simplest choice, not the only one. An agent can also hold beliefs about several past steps and revise them together (for instance by marginal message passing), so that a reading taken now sharpens where it thinks it was three steps ago. Module 5 uses this, under the name **smoothing**.

## The second table

To account for its own movement, the agent needs something that says where a step takes it. As with observations, the simplest thing that does the job is a table, and it is the second of the two the agent will carry.

Call it *B*, it is the **transition model**. For each cell the agent might be standing on, *B* says how likely it is to end up on each of the cells after taking a step.

Read it one origin at a time, exactly as you read *A*. Fix a starting cell. Look down the column belonging to it. What you find is a set of probabilities saying how likely each destination is, given that you started there and moved. It's giving you how likely you are to transition between two states. 

$$B_{ij} = p(s' = i \mid s = j)$$


The column index is where you were, the row index is where you end up. The prime on $s'$ marks the next state, as distinct from the current one.

For a fixed origin $s$, the corresponding column of *B* is a probability distribution over destinations:

$$
\sum_{s'} p(s' \mid s,a)=1.
$$

In other words, *B* redistributes probability without creating or losing it. We will call this **probability conservation**.

<widget id="b-columns"></widget>


The transition model *B* is the second table the agent carries, after the likelihood *A*. Together they make up the agent's model of its world, and later parts add to it: Module 4 adds C, which says what the agent wants to see.


## Why the table is not a permutation

If actions were exact, each column of *B* would contain one one and otherwise zeros. Moving would shift the belief across the grid without changing its shape.

Real actions are uncertain: a step may stop short or drift beside its target, and the agent does not know where it actually landed. Each column of *B* therefore spreads probability over the intended cell and nearby alternatives. On this wrapping grid, repeated uncertain moves make the belief less sharp.

The figure below controls this spread. At one, motion is exact: *B* is a permutation and moving adds no uncertainty.

Motion is not the only source of uncertainty. Several cells may produce the same observation, so looking cannot distinguish them. This is **perceptual aliasing**, as in a corridor of identical doors.

The two sources require different remedies:

- Uncertainty added by motion can be reduced by looking, if the surroundings are informative.

- Uncertainty caused by aliasing can be reduced only by moving somewhere that produces a different observation.

Without an informative observation, motion errors accumulate from one step to the next.

Real motion is imprecise, and the agent cannot directly observe where each action takes it. B must therefore represent what may actually happen, not just what the agent intends. Treating motion as perfect would give the agent unwarranted confidence, a dangerous flaw in robotics.

> **If you want the number behind the figure**
>
> The figure below measures the sharpness of the belief using **entropy**:
>
> $$H = -\sum_s p(s) \log p(s)$$
>
> Entropy is zero when the belief is concentrated on one cell and largest when it is spread evenly across all 25. The figure presents this as a scale from sharp to flat; nothing else in this module requires the formula.

<widget id="permutation-vs-spread"></widget>

In the figure above, take several steps without looking, then look. On this wrapping grid, each step spreads the belief; an informative observation can narrow it again.

This spreading relies on more than probability conservation. Here *B* is **doubly stochastic**: its rows as well as its columns sum to one, so prediction cannot make the belief more concentrated. With walls, probability may accumulate at a boundary, and prediction can sometimes sharpen the belief.

## Predicting

With *B* defined, take the agent at the point where it is about to move. It holds a belief about where it is now, and it wants a belief about where it will be after taking an action.

Say the agent has narrowed its position to two cells (read as row, column): it believes it is either at (2,2) or at (2,3), with no strong preference between them. It commands a step east. How plausible is it that the agent ends up at (2,3)?

Both origins can put it there, in different ways. From (2,2), a step that lands as intended. From (2,3), a step that fails to carry the agent out of the cell it was already in.

Each contributes in proportion to two things: how plausible that origin was, and how likely a step east from there is to end at (2,3). Multiply those two for each origin, add the results, and you have the answer.

$$p(s') = \sum_{s} p(s' \mid s, a) \, p(s)$$

With $a$ being the desired motion. Then do that for every destination and you get the prediction step. Codes write it is as

    belief = B @ belief

The matrix product is doing exactly the sum above, once for each destination. If that identity is not yet clear, the box below writes it out as a loop.

> **If matrix products do not yet read as sums to you**
>
> The product `B @ belief` is the following loop, written more compactly:
>
>     new_belief = np.zeros(n_states)
>     for i in range(n_states):          # destination
>         for j in range(n_states):      # origin
>             new_belief[i] += B[i, j] * belief[j]
>
> Row `i` of the result is row `i` of `B`, multiplied elementwise against the belief and then summed. Nothing else is happening. Every matrix product in this tutorial can be read this way, and if you are ever unsure what one is doing, expanding it into this loop will tell you.

In the figure below, hover over a cell in the prediction row to see its weighted contributions from every origin. The update rows will matter when we return to observation.

<widget id="conserves-vs-not"></widget>


## One table is not enough

There is something wrong with *B* as written. It describes what happens when the agent takes one kind of action, as though it could only move one way.
 
The agent can step east, south, west or north, or stay where it is. Those are five different actions with five different consequences, and one table cannot hold them all. A step east puts most of the probability on the cell to the east of wherever the agent started; a step north puts it on the cell to the north; staying put leaves it where it was. Each of the five needs its own table, saying how likely the agent is to end up in each cell under that action.
 
So *B* gains an index:

$$B_{ijk} = p(s' = i \mid s = j, a = k)$$

<widget id="b-stack"></widget>

or in code, `B[s_next, s, a]`. It is a stack of matrices, one per action, and each matrix conserves probability.

The first two indices are ordered, because states are: the belief is a list of cells in a fixed order, and B is square in that order. The third is not. Which table holds "north" and which holds "stay put" is arbitrary; the action index only selects a table, and all that matters is that the choice is fixed once and used consistently everywhere.
 
Fixing the action recovers what you already have. `B[:, :, a]` is a single matrix, and

    belief = B[:, :, a] @ belief

is the same operation as before, a matrix multiplied by the belief. The stack does not complicate the prediction step; it only says which table to use.
 
For now the agent executes a sequence of actions decided in advance, and how it might select them comes in module 4.
 

## Expectation and marginalisation

The prediction step has two names because the same calculation can be viewed in two ways:

$$
p(s' \mid a)
= \sum_s p(s' \mid s,a)\,p(s)
$$

For a fixed destination $s'$, the transition probability $p(s' \mid s,a)$ has a different value for each possible origin $s$. The current belief $p(s)$ weights these values according to how likely each origin is. Their weighted sum is therefore an **expectation**:

$$
p(s' \mid a)
= \mathbb{E}_{p(s)}\!\left[p(s' \mid s,a)\right].
$$

Unlike an ordinary average, which gives every value equal weight, an expectation gives more weight to values associated with more probable states.

The same calculation is also a **marginalisation**. Before summing, the prediction depends on both the origin $s$ and the destination $s'$. Summing over every possible origin removes $s$, leaving a distribution over the destination alone. We say that $s$ has been **marginalised out**, or **summed out**.

Thus, the two terms emphasise different aspects of the same operation:

- **Expectation:** take a probability-weighted average over $s$.
- **Marginalisation:** sum over $s$ so that it no longer appears in the result.

The matrix product `B @ belief` performs this calculation for every destination $s'$ at once.

## Why updating requires normalisation

Prediction and observation updating use different kinds of multiplication.

Prediction uses a matrix–vector product:

$$
p(s' \mid a)
=
\sum_s p(s' \mid s,a)\,p(s).
$$

Because *B* conserves probability, the result is already a belief.

Observation updating instead multiplies the belief elementwise by the likelihoods associated with the observed outcome:

$$
\tilde p(s \mid o)=p(o \mid s)\,p(s).
$$

These likelihoods come from one row of *A*. They compare how well different states explain the observation; they do not form a probability distribution over states. The resulting values are therefore relative weights and must be normalised:

$$
p(s \mid o)
=
\frac{p(o \mid s)\,p(s)}
{\sum_{\bar s}p(o \mid \bar s)\,p(\bar s)}.
$$

So *B* **redistributes** probability, whereas *A* **reweights** it.

The update rows in the earlier figure show the reweighting before and after normalisation.

> **A useful implementation check**
>
> A prediction should conserve the total probability. If it does not, check that each column of the selected *B* matrix sums to one and that origins and destinations have not been transposed. An observation update, by contrast, should be normalised explicitly.

## Order

Each cycle begins with the posterior produced by the previous observation. The agent then follows this sequence:

$$
\text{posterior}_{t-1}
\xrightarrow{\text{predict}}
\underbrace{\text{predicted belief}_{t}}_{\text{prior}_{t}}
\rightarrow
\text{move}_{t}
\rightarrow
\text{observe}_{t}
\xrightarrow{\text{update}}
\text{posterior}_{t}.
$$

In short, the loop is:

$$
\text{predict} \rightarrow \text{move} \rightarrow \text{see} \rightarrow \text{update}.
$$

Some texts begin the same cycle at the observation rather than the action, so they present updating before prediction. The underlying sequence is unchanged; only the point chosen as the start of a cycle differs. What matters is that time indices are used consistently, because prediction and updating do not commute.

## Imagining

The agent can use *B* to predict the result of an action without taking it. It can then predict another step from that result, building a possible future one action at a time.

On this wrapping grid, predictions without observations spread the belief until it becomes flat. The farther ahead the agent looks, the less certain it is about where it will be.

The agent can now predict how actions change its belief and use observations to revise it. The next question is how it should choose what to do.



> **Optional: why the literature is written in logarithms**
>
> Nothing in this section requires logarithms, and you can implement everything above without them. This box is here so the notation is familiar when it starts to matter, which is very very soon.
>
> The observation update multiplies a belief by a likelihood. Prediction also contains products, but it sums them over possible origins. Logarithms simplify individual products; they do not remove that sum.
>
> Logarithms turn products into sums:
>
> $$\log(xy) = \log x + \log y$$
>
> This makes the factors in the observation update easier to examine. In log space, that update becomes
>
> $$\log p(s \mid o) = \log p(o \mid s) + \log p(s) - \log p(o)$$
>
> Here $o$ is the observation the agent received. On the right, the log likelihood and log prior are added, while $\log p(o)$ is subtracted. The probability $p(o)$ is the total of $p(o \mid s)p(s)$ across all states; it normalises the update. Dividing by it becomes subtracting its logarithm.
>
> This is why the active inference literature writes almost everything in log space. Expected free energy, which the agent will eventually use to choose its next actions, is a sum of terms that are each logarithms of probabilities, and it is only readable as a sum. If you meet $\log p(o \mid s)$ in a paper, it is the same entry of *A* you have been reading all along, with a logarithm applied.
>
> One practical note. Any single probability lies between zero and one, and the logarithm of such a number is never positive: $\log 1 = 0$, and anything smaller gives a negative result. So log probabilities are negative almost everywhere, and the less plausible something is, the more negative its logarithm. Nothing is wrong when you see negative numbers throughout a calculation in log space; that is what the space looks like.

## To go further

- Namjoshi, [*Fundamentals of Active Inference*](https://mitpress.mit.edu/9780262050951/fundamentals-of-active-inference/), Chapter 9, especially §9.2 and §9.4, develops prediction and dynamic filtering with transition matrices, then indexes those matrices by action in discrete POMDPs.

- Da Costa et al., [“Active inference on discrete state-spaces: a synthesis”](https://pmc.ncbi.nlm.nih.gov/articles/PMC7732703/), writes the action-indexed transition probability as $s_\tau\cdot B_{\pi_{\tau-1}}s_{\tau-1}$ (Table 2) and embeds it in the complete discrete active-inference model.
