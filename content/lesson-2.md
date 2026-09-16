# Lesson 2. Moving

## The agent has been standing still

Everything so far has happened to an agent that did not move. It looked, it revised what it held, it looked again, and each reading sharpened its sense of where it was. 

That is of limited use. If several cells produce the same reading, every one of them explains it equally well, and no amount of further looking will separate them. The agent has to move.

The moment it moves, something happens that nothing so far accounts for. Its belief describes where it was, and it is now somewhere else.

Anyone who has walked across a dark room knows the effect: the further you go without touching anything, the less sure you are of where you are standing. Looking sharpens a belief. Moving blunts it. What follows is why that happens, and what the agent has to carry in order to account for it.

## Two ways a belief can change

Keep the two causes apart, because they are different operations and the agent applies them at different moments.

The first is the one already covered. Something arrives. An observation reaches the agent from outside, and the agent revises what it holds to account for it. The agent is passive here: it receives, and it responds.

The second is new. The agent commits to a step, and the cells it might be on afterwards are not the cells it might be on now, so the belief has to shift with it. Nothing has been observed yet; this change is worked out from what the agent predict, not from anything the world told it.

Both operations act on the same belief, and that belief is over states: a probability for each cell the agent might be on. By the end of this section, each cycle will run like this: the agent decides on an action and forms a belief about where that action would put it, then takes the step, observes from wherever it has arrived, and revises that belief with what it saw. The first belief is **the prior** for this turn, the revised one **the posterior**.

This tutorial keeps a single belief about where the agent is now, and each cycle overwrites the last. That is the simplest choice, not the only one. Some approaches hold beliefs about several past time points at once and revise them together (for instance by marginal message passing), so that a reading taken now can sharpen the agent's account of where it was three steps ago. They use the same two operations but they apply them to more than one moment.

## The second table

To account for its own movement, the agent needs something that says where a step takes it. As with observations, the simplest thing that does the job is a table, and it is the second of the two the agent will carry.

Call it *B*, it is the **transition model**. For each cell the agent might be standing on, *B* says how likely it is to end up on each of the cells after taking a step.

Read it one origin at a time, exactly as you read *A*. Fix a starting cell. Look down the column belonging to it. What you find is a set of probabilities saying how likely each destination is, given that you started there and moved. It's giving you how likely you are to transition between two states. 

$$B_{ij} = p(s' = i \mid s = j)$$


The column index is where you were, the row index is where you end up. The prime on $s'$ marks the next state, as distinct from the current one, this is standard notation.

$$\sum_i B_{ij} = 1 \quad \text{for every } j$$

Each column sums to one, because wherever the agent starts, it ends up somewhere.

<widget id="b-columns"></widget>


This is the second table the agent carries, and there will be more. Its model of the world is not a single object but a collection of them, and everything added later is added to this collection.

## Why the table is not a permutation

If an action always landed the agent exactly on the cell it aimed for, *B* would be trivial. Each column would hold a single one and the rest zeros, and the belief would move across the grid with the agent, unchanged in shape.

Actions do not behave like that. A step sometimes fails to carry the agent out of the cell it started in, and sometimes carries it past or beside the cell it aimed for. What matters is not that this happens often, but that the agent cannot tell whether it happened: it commanded a step east, and afterwards it holds no independent record of where it landed. It is what wheel odometry does on a real floor, and what your own step does across a dark room.

So the columns of *B* are spread rather than sharp. Most of the probability sits on the cell the agent aimed for. The rest is shared between the cell it started in, for the step that did not carry, and the cells adjacent to the intended one, for the step that drifted.

That spread is what makes moving costly, every motion leaves it less sure of where it is than it was before.

How much of the probability stays on the intended cell is a number the agent carries, and the widget at the end of this section lets you set it. At one, *B* is the permutation above and moving costs nothing at all.

Take the spread out and the agent still does not know where it started, and still has to carry its belief through every action. What disappears is the cost: a belief that was sharp before the action is exactly as sharp after it, so the agent could move indefinitely without becoming less certain of where it was, and looking would be useful but never urgent.

That is only half the story, because the spread in *B* is not the only thing standing between the agent and knowing where it is. The other comes from the world rather than from the agent. If several cells produce the same observation, then looking from any of them gives the same reading, and no amount of looking will say which one the agent is on. This is called **perceptual aliasing**, and it is why a corridor of identical doors is harder to navigate than a corridor of numbered ones, and a dark room or an open desert harder still, since everywhere in them reports the same thing.

The spread and the aliasing, both source of uncertainty, are undone by different means, and it helps to keep them apart. 

- Uncertainty the agent picks up by acting is undone by looking, provided there is something worth seeing. 

- Uncertainty that comes from aliasing is not undone by looking, since looking again returns the same reading. It is undone only by going somewhere that reports something different.

This is also why the spread in *B* is the realistic modelling rather than a pessimistic one. The step itself is imprecise whether or not the agent can see; what looking buys is the chance to correct for it afterwards. Where that correction is unavailable, the error from one step is still in the belief when the next step adds to it, and the two accumulate rather than being cleared.

> **If you want the number behind the bar**
>
> The bar is the belief's **entropy**, the standard measure of how spread out a distribution is:
>
> $$H = -\sum_s p(s) \log p(s)$$
>
> It is zero when the belief sits entirely on one cell, and largest when the belief is even across all twenty-five. Nothing else in this lesson needs it; the bar is labelled sharp and flat and can be read without it.

<widget id="permutation-vs-spread"></widget>


## Predicting

With *B* defined, take the agent at the point where it is about to move. It holds a belief about where it is now, and it wants a belief about where it will be after taking an action.

Say the agent has narrowed its position to two cells (read as row, column): it believes it is either at (2,2) or at (2,3), with no strong preference between them. It commands a step east. How plausible is it that the agent ends up at (2,3)?

Both origins can put it there, in different ways. From (2,2), a step that lands as intended. From (2,3), a step that fails to carry the agent out of the cell it was already in.

Each contributes in proportion to two things: how plausible that origin was, and how likely a step east from there is to end at (2,3). Multiply those two for each origin, add the results, and you have the answer.

$$p(s') = \sum_{s} p(s' \mid s, a) \, p(s)$$

Then do that for every destination. That is the prediction step, and it is what the code writes as

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

Notice what the result does not need. When the agent multiplied its belief by the likelihood, the numbers that came out did not sum to one, and had to be divided by their total before they were a belief again. Here they already sum to one, and no correction is required.

## One table is not enough

There is something wrong with *B* as written. It describes what happens when the agent takes one kind of action, as though moving were the only thing it could do.
 
The agent can step east, south, west or north, or stay where it is. Those are five different actions with five different consequences, and one table cannot hold them all. A step east puts most of the probability on the cell to the east of wherever the agent started; a step north puts it on the cell to the north; staying put leaves it where it was. Each of the five needs its own table, saying how likely the agent is to end up in each cell under that action.
 
So *B* gains an index:

$$B_{ijk} = p(s' = i \mid s = j, a = k)$$

<widget id="b-stack"></widget>

or in code, `B[s_next, s, a]`. It is a stack of matrices, one per action. Every one of them has columns summing to one, for the same reason as before: whatever the agent does, it ends up somewhere.
The first two indices are ordered, because states are: the belief is a list of cells in a fixed order, and B is square in that order. The third is not. Which table holds "north" and which holds "stay put" is arbitrary; the action index only selects a table, and all that matters is that the choice is fixed once and used consistently everywhere.
 
Fixing the action recovers what you already have. `B[:, :, a]` is a single matrix, and

    belief = B[:, :, a] @ belief

is the same operation as before, a matrix multiplied by the belief. The stack does not complicate the prediction step; it only says which table to use.
 
For now the agent executes a sequence of actions decided in advance, and how it might select them comes later.
 

## What that sum is called 

Two names are attached to the prediction step you just computed, and you will meet both in any paper on this subject.
 
Note first when the prediction step is computed. The agent has not moved. It has chosen an action and is working out where that action would put it, which is why the belief providing the weights is still its belief about where it is now.
 
**Expectation.** Look again at the structure of the prediction step. You had a quantity whose value depends on which cell the agent is currently in: the probability that a step east from that cell lands on (2,3). From (2,2) that quantity is high. From (2,3) it is small. From a cell on the far side of the grid it is zero.

Call that quantity $f(s)$: for each possible origin $s$, the probability that a step east from that origin lands on (2,3). It is one entry of *B*, read for a fixed action and a fixed destination, and there is one such entry per origin.

The agent does not know with full certainty which origin is the real one. What it has instead is a probability of being at each of its known cells, which is its current belief $p(s)$. If it knew its origin, it could take the one value of $f$ belonging to that origin and stop there. It does not, so instead it takes the value of $f$ at every origin, weights each one by how plausible that origin is under the belief, and adds the results. That is an **expectation** of $f$ under $p$.

Textbooks write it like this:

$$\mathbb{E}_{p(s)}\!\left[f(s)\right] = \sum_s p(s) \, f(s)$$

The letter $f$ is a placeholder. It stands for any quantity whose value changes with the state, and textbooks use it when stating a rule that holds whatever that quantity happens to be. The subscript names the distribution supplying the weights.

The active inference literature rarely uses $f$. It writes the quantity out instead, so for this step you would see the sum in full:

$$p(s') = \sum_s p(s' \mid s, a) \, p(s)$$

The two equations say the same thing. In the second, $p(s' \mid s, a)$ has taken the place of $f(s)$, and $p(s)$ is still the belief supplying the weights.

The quantity inside the brackets will not always be a transition probability. Later in this tutorial the agent will take expectations of preferences, of logarithms of probabilities, and of scores attached to outcomes it has not yet seen. The arithmetic is identical in each case: a value for every state, a weight for every state, multiply and add.

The only thing that distinguishes an expectation from an ordinary average is where the weights come from. When you average a set of numbers in the ordinary way, you add them all up and divide by how many there are, which treats every number as counting for exactly as much as every other. An expectation does not. Each value is multiplied by how plausible its state is before being added in, so a value belonging to a state the agent considers unlikely contributes proportionally little to the result, and one belonging to a state it has nearly ruled out contributes almost nothing.
 
Every remaining part of this tutorial uses expectations, and by the point where the agent is choosing its own actions it will be taking them over quantities considerably less concrete than which cell it is standing on. Get comfortable with the notation now, while the quantity inside the brackets is something you can point at.
 
**Marginalisation.** describes what the sum achieves rather than how it is computed.

Before the sum, everything the agent could say about its next position was tied to its current one. If I am at (2,2), then a step east puts me at (2,3) with such and such probability; if I am at (2,3), then east puts me at (2,3) with some other probability (probably lower). Every statement had an "if I am at" in front of it. Written out, each of those statements is one entry of the transition matrix *B*:

$$p(s' \mid s, a)$$

and the bar is the "if I am at": the quantity is not usable until an origin $s$ is supplied.

The agent cannot use a quantity in that form, because it does not know which origin applies. What it needs is a statement about the destination alone, with nothing to the right of a bar:

$$p(s')$$

Summing over origins is what removes the "if I am at". Each origin supplies one transition probability, weighted by the belief that the agent is at that origin:

$$p(s') = \sum_s p(s' \mid s, a) \, p(s)$$

Read the right-hand side and then the left. On the right, $s$ appears twice. On the left it does not appear at all. Every origin was involved in producing the number, and none of them survives into it, because the sum runs over all of them and leaves nothing to specify.

That removal has a name. The origin has been **marginalised out**, or equivalently **summed out**. Both phrases appear in the literature and mean the same thing: the first names the operation, the second names how it is carried out, and since it is carried out by summing, they refer to the same step.

So the two names describe the same arithmetic from different angles. It is an expectation, because each transition probability is multiplied by the belief in its origin before being added in. It is a marginalisation, because once the additions are done, the origin no longer appears in the answer. And whenever you see a matrix product in this literature, this is usually what it is: a marginalisation, written in one line instead of a loop.

## Why one multiplication needs fixing and the other does not

The agent has two operations, and each of them multiplies. The update, which takes in an observation, multiplies the belief by the likelihood:

$$p(s \mid o) \propto p(o \mid s) \, p(s)$$

The prediction step, which accounts for an action, multiplies the transition matrix *B* by the belief, which is the sum derived above:

$$p(s') = \sum_s p(s' \mid s, a) \, p(s)$$

Both are multiplications, but they are not the same kind. The update produces a result that does not sum to one and has to be divided by its total before it is a belief again.

$$p(s \mid o) = \frac{p(o \mid s) \, p(s)}{\sum_{s'} p(o \mid s') \, p(s')}$$

Note that the first equation carries a proportional sign and the second an equals sign. The proportional sign is there precisely because the result needs dividing; the equals sign appears once the division has been done, and the prediction step needs no such step to earn it.

The prediction step is a matrix product. Every cell the agent might currently be on contributes to every cell it might end up on. The sum runs over the whole grid, and a cell the agent considers implausible contributes proportionally little. In the example only (2,2) and (2,3) carried any appreciable belief, so those were the two contributions that mattered. The result sums to one on its own, and needs no correction.

<widget id="conserves-vs-not"></widget>

The difference follows from the shape of *B*. Each column of *B* belongs to one starting cell *s*, and it holds the probability of arriving at each cell *s'* on the grid from that start, under one action. Those probabilities sum to one because the agent ends up somewhere: wherever it starts, some destination receives it. So whatever belief the agent had in a starting cell is passed on in full, spread across the destinations that cell can reach, and nothing is lost or created on the way. Add up all the destinations and you have the same total you began with.

The likelihood has no such constraint. Its entries are read across a row of *A*, and they answer a separate question for each cell: how probable was this observation, had the agent been standing there? Those are independent questions about independent cells, and nothing requires their answers to add to anything in particular. Multiplying the belief by them therefore produces a total that is whatever it happens to be, and normalisation is what puts it back to one.

In your code, if a prediction step returns something that does not sum to one, look at *B* first: the usual cause is a column that does not sum to one, either because it was built by hand or because the indices were written the wrong way and you are reading rows where you mean columns. If an update returns something that already sums to one before you normalise it, look at the likelihood: the usual cause is that every cell received the same score, which means the observation told the agent nothing and the belief has come out of the update unchanged.

## Order

The agent now predicts and updates, in a fixed order each turn. It settles on an action and predicts where that action would put it, which gives the prior for this turn. It then takes the action, observes from wherever it has arrived, and updates that prior with what it saw, which gives the posterior. The prior carries the action the agent intended; the posterior carries the action it took and the observation it received.

That order is a convention. Updating first and predicting afterwards describes the same cycle entered at a different point, and the literature contains both. What matters is that the ordering is consistent, because the two do not commute and a mixed convention produces a belief that is off by one step in a way that is very difficult to see.

This tutorial predicts, then updates.

## Moving without looking

Now the demonstration.

Start the agent with a belief about where it is, sharply peaked on one cell, and let it act repeatedly without observing anything in between. Each prediction step spreads the belief a little, since probability that sat on one cell is shared out across the cells that cell can reach, and the spreads overlap and accumulate.

Watch the shape. It starts peaked, becomes broad, and ends flat, which is the belief of an agent that has no idea where it is. Nothing went wrong on the way: each step was handled correctly, and the agent is uncertain because it has taken a series of actions whose outcomes it could not check. Flat is where it stops, because once every cell is equally plausible there is nothing left for a step to redistribute.

Then let one observation arrive, and update. The belief collapses again. Predicting can only ever widen a belief; observing is the only thing that narrows one, which is why the agent needs both.

That first half is worth stating exactly, because it is a property of *B* rather than a tendency. Every column of *B* sums to one, because the agent ends up somewhere. On a wrapping grid every *row* sums to one as well, because every cell receives exactly as much probability as it sends out. A matrix with both properties cannot concentrate a distribution, only spread it or leave it alone. Put a wall in and the second property fails: probability piles up against the wall instead of passing through, and a prediction step can then sharpen a belief rather than blunt it.

## Imagining

One last thing follows from the prediction step.

Nothing about it requires the agent to actually move. It is arithmetic on a belief and a table, so the agent can apply it to an action it has not taken and get a belief about where it would be, then apply it again and get a belief about where it would be after two.

Call that imagining. It is prediction with no observation to come back on: nothing arrives to check it, so nothing corrects it, and it spreads at every step for the reason just demonstrated. The agent's picture of its own future therefore gets vaguer the further ahead it looks. A few steps are informative. Twenty are not.

## Where this leaves the agent

An agent that only acts is on a clock. Every action costs it certainty, the cost accumulates, and the only thing that resets it is looking.

So looking has a value, and that value can be weighed against the cost of standing still to do it. The agent does not yet know how to make that comparison, but it now holds both halves of what is needed to work it out.



> **Optional: why the literature is written in logarithms**
>
> Nothing in this section requires logarithms, and you can implement everything above without them. This box is here so the notation is familiar when it starts to matter, which is very very soon.
>
> Look back at what the agent has done. The update multiplied a belief by a likelihood. The prediction step multiplied entries of *B* by entries of the belief. Every operation so far has been a product, and products are awkward to reason about: there is no easy way to see what a product of four terms is doing by looking at it, and there is no easy way to separate the contribution of one factor from the rest.
>
> Logarithms turn products into sums:
>
> $$\log(xy) = \log x + \log y$$
>
> so a chain of multiplications becomes a chain of additions, in which each factor contributes its own term and can be examined on its own. The update, written in logarithms, becomes
>
> $$\log p(s \mid o) = \log p(o \mid s) + \log p(s) - \log p(o)$$
>
> Here $o$ is the observation the agent received. The first two terms on the left part of the equation are the likelihood and the prior, now added rather than multiplied. The third, $p(o)$, is the probability of receiving that observation at all. It is worked out by taking, for each cell, the probability of seeing $o$ from that cell, multiplying it by the probability the agent assigns to being in that cell, and adding the results across the grid. It is the total that the update divides by, and dividing becomes subtracting in log space.
>
> This is why the active inference literature writes almost everything in log space. Expected free energy, which the agent will eventually use to choose its next actions, is a sum of terms that are each logarithms of probabilities, and it is only readable as a sum. If you meet $\log p(o \mid s)$ in a paper, it is the same entry of *A* you have been reading all along, with a logarithm applied.
>
> One practical note. Any single probability lies between zero and one, and the logarithm of such a number is never positive: $\log 1 = 0$, and anything smaller gives a negative result. So log probabilities are negative almost everywhere, and the less plausible something is, the more negative its logarithm. Nothing is wrong when you see negative numbers throughout a calculation in log space; that is what the space looks like.