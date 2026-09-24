# Module 3. Free energy
## The step that gets expensive

Go back to the update the agent performs when an observation arrives. It multiplies the belief it held beforehand by the likelihood read from *A*, and then divides by the total:

$$p(s \mid o) = \frac{p(o \mid s) \, p(s)}{p(o)} \tag{1}$$

Throughout this section $s$ is the cell the agent is in now, the one it is trying to infer given the observation that has just arrived. No transition is involved and no next state appears: the prediction step has already happened, and $p(s)$ is what it produced.

The numerator is cheap. For any cell, $p(o \mid s)$ is one entry of *A* and $p(s)$ is that predicted belief read at the same cell, so the product for that cell can be worked out on its own, without reference to any other cell.

The denominator is different. $p(o)$ is that same product, $p(o \mid s) \, p(s)$, worked out for every state in the world and added up. It is there because the products do not sum to one on their own, and dividing by their total is what makes them a distribution again. Which means the total has to be complete: every state contributes to it, including the ones the agent has nearly ruled out, because a total missing some of its terms would not be the total and dividing by it would leave the result summing to something other than one.

That completeness is what makes it expensive, and the expense is structural. The numerator for a given cell needs only that cell's entries, so the agent can work out one cell and stop. The denominator cannot be approached that way: no single cell of the posterior can be reported until every state has been visited, so there is no partial answer and no way to spend less than the whole cost.

In the grid, that is a few dozen numbers and no difficulty at all. But a state is a distinct situation the agent can be in, not simply a cell, and the number of them multiplies with everything the agent has to keep track of. A grid of 100 cells, with 4 orientations, and 8 lamps that may each be on or off, gives $100 \times 4 \times 2^8 = 102{,}400$ states, from a world that is still only a room with some lamps in it. Add one more thing to keep track of and the count multiplies again. The denominator sums over all of them, and it has to be recomputed at every step, for every observation.


## The posterior as a target

Up to now the posterior has been a result. The agent had a prior, an observation arrived, and the posterior was what came out of the arithmetic. There was no question of aiming at it, because it was produced directly.

Turn that around. Treat the posterior as a target: the distribution the agent would like to hold.

This is a change in how the problem is posed, and nothing about the maths has changed yet. If the posterior is a result, the agent either computes it or it does not, and there is nothing in between. If it is a target, the agent can get close to it, judge how close it has got, and choose to stop there because the remaining improvement is not worth what it would cost.

## The approximate posterior

If the agent is going to approach the posterior rather than compute it, it needs something to approach with.

So let it put forward a distribution of its own: a probability for each cell, summing to one, exactly the same kind of object as every belief so far. Call it $q(s)$, and call it the **approximate posterior**, since approximating the posterior is what it is for. It is also called the **variational distribution**, and both names appear throughout the literature.

Every distribution in this tutorial has so far been produced by a rule. The prior came from the previous step. The posterior came from multiplying and normalising. The predicted belief came from *B*. Each was the output of an operation, and there was never any choice about its value.

$q$ is not like that. The agent proposes it, and it can propose anything that qualifies as a distribution: flat, peaked on one cell, peaked on the wrong cell, close to the posterior or far from it. Nothing constrains the choice, and nothing yet says whether a given choice is any good. A $q$ that approximates the posterior badly is still an approximate posterior; the name says what the object is for, not that it succeeds.

That last point is what makes the idea safe. Letting the agent propose a distribution sounds like letting it invent its own beliefs, which would be an odd thing to permit, and it is not what is happening. What the agent proposes is an answer offered to be scored, and the scoring is what the rest of this section provides. A $q$ that fits the evidence badly will score badly, and the agent has no way of protecting it. Freedom to propose is not freedom to be right.

Two consequences follow from $q$ being a free object rather than a derived one.

The first is that the agent can hold a belief it knows to be imperfect. Until now, whatever the agent held was by construction the correct consequence of everything it had seen. There was no notion of a belief being approximately right, because there was nothing to be approximately right relative to. With $q$ and a score, there is: $q$ can be near the posterior or far from it, and the agent can know which.

The second is that the agent's task changes shape. Computing the posterior was arithmetic, with one answer and one route to it. Finding a good $q$ is a search, over the set of all distributions the agent could propose, looking for the one that scores best. That set has a name: it is the **probability simplex**, the set of all lists of probabilities that sum to one. Every valid belief is somewhere in it. The posterior is one point in that set, and the agent is looking for it without being able to see where it is.

<widget id="q-lives-in-the-simplex"></widget>

So the object this section is built on is a belief the agent adopts provisionally, which it can change at will, and which is judged rather than derived. What remains is to say how it is judged.

## How far apart are two distributions?

An approximate posterior is only useful if the agent can tell a good one from a bad one, and a good $q$ means one that is close to the posterior.

Consider what such a measure has to do. Take one cell. The approximate posterior assigns it probability $q(s)$; the true posterior assigns it $p(s \mid o)$. If the two agree, $q$ is right about that cell, and a measure of how far apart the two distributions are should return zero. The greater the disagreement between the two beliefs, the higher the value.

The ratio is the natural thing to look at:

$$\frac{q(s)}{p(s \mid o)} \tag{2}$$


The ratio ranks disagreements correctly: the further it sits from one, the more the two distributions differ about that cell. What it gets wrong is the case of no disagreement at all. These per-cell values are going to be added up across the grid, so a cell $q$ is right about has to contribute nothing to the total. The ratio contributes one instead, and a $q$ that was perfect everywhere would total the number of cells rather than zero.

The logarithm fixes exactly that, since $\log 1 = 0$.

$$\log \frac{q(s)}{p(s \mid o)} \tag{3}$$


Now agreement returns zero, a $q$ that is too confident returns a positive number, and one that is not confident enough returns a negative number.

That is one cell. To get a single number for the whole distribution, the contributions have to be combined, and they cannot simply be added, because a cell $q$ considers irrelevant should not count as much as one it is staking everything on. So weight each cell's contribution by the probability $q$ assigns to it, which is an expectation under $q$:

$$D_{\mathrm{KL}}\!\left[q(s) \,\|\, p(s \mid o)\right] = \mathbb{E}_{q(s)}\!\left[\log \frac{q(s)}{p(s \mid o)}\right] = \sum_s q(s) \log \frac{q(s)}{p(s \mid o)} \tag{4}$$


This is the **Kullback-Leibler divergence** from $q$ to $p(s \mid o)$, written $D_{\mathrm{KL}}$ and usually just called the KL divergence. It is the same operation as the prediction step: a value for every state, a weight for every state, multiply and add. The value being averaged is now a log ratio rather than a transition probability.

Three properties, and the third is the one that catches people out.

This divergence measure has three essential properties. 
- It is never negative, though this is not obvious: a cell contributes a negative term whenever q is less confident than the posterior, but the weights systematically underweight these terms, with the floor at zero reached only when both distributions match everywhere (Gibbs' inequality, proved via Jensen's inequality on the logarithm's concavity). 
- It equals zero only when the two distributions are identical, and approaches zero as they converge. 
- It is not symmetric: D $D_{\mathrm{KL}}[q \| p]$ and $D_{\mathrm{KL}}[p \| q]$ are different, since the weights 
depend on which distribution is written first. This makes it a measure of 
discrepancy rather than a distance; the order in the brackets cannot be 
reversed without changing the value.


<widget id="kl-never-negative"></widget>


## Why that gap cannot be measured directly

The agent now has what it wanted: a way of saying how far its approximate posterior $q$ sits from the true posterior $p(s \mid o)$. Look at what equation 4 asks for. For every cell, it needs the value of $q$ at that cell, which the agent has, since $q$ is its own proposal. And it needs the value of $p(s \mid o)$ at that cell, which the agent does not have. That value is the posterior, the very quantity the agent set out to find.

So the expression cannot be evaluated as it stands. The agent would have to know the answer in order to measure how far it is from the answer.

But it is not the whole posterior that is missing. Equation 1 shows where the gap is, since $p(o \mid s)$ is available. What it cannot compute is $p(o)$, for the reason set out at the start of this section.

What follows is a way of separating the divergence into the part that depends on that sum and the part that does not.

## Free energy

We have the divergence, and we cannot evaluate it presented in equation 4.

The posterior $p(s \mid o)$ inside the logarithm is the problem, we do not have the information yet. Everything else on the line is available to the agent at any point in time.

The posterior is built from three things:

$$\underbrace{p(s \mid o)}_{\text{posterior}} = \frac{\overbrace{p(o \mid s)}^{\text{likelihood}}\ \overbrace{p(s)}^{\text{prior}}}{\underbrace{p(o)}_{\text{evidence}}}, \qquad p(o) = \sum_s p(o \mid s)\,p(s)$$

Taken together, the likelihood and the prior form a single object:

$$p(o \mid s)\,p(s) = p(o, s)$$

This is the **generative model**: the joint probability that the agent is in state $s$ and receives observation $o$. One object rather than two, and the agent holds all of it, since it is its own model. *A* supplies one half and the prior supplies the other.

Written in terms of the joint, the posterior is

$$p(s \mid o) = \frac{p(o, s)}{p(o)}$$

The posterior can be taken out of the divergence and replaced by a quantity the agent has, divided by one it does not. Putting it in:

$$D_{\mathrm{KL}}\!\left[q(s) \,\|\, p(s \mid o)\right] = \sum_s q(s) \log \frac{q(s)}{\dfrac{p(o,s)}{p(o)}}$$

Dividing by a fraction is multiplying by its inverse, which lifts $p(o)$ into the numerator:

$$= \sum_s q(s) \log \frac{q(s) \, p(o)}{p(o,s)}$$

A product inside a logarithm splits into a sum of logarithms, which takes $p(o)$ out of the fraction:

$$= \sum_s q(s) \left[ \log \frac{q(s)}{p(o,s)} + \log p(o) \right]$$

And $\log p(o)$ does not depend on $s$, so it comes out of the sum. Since $\sum_s q(s) = 1$, it survives unchanged:

$$= \underbrace{\sum_s q(s) \log \frac{q(s)}{p(o,s)}}_{\text{free energy}} + \ \log p(o)$$

The posterior has been removed and $p(o)$ has been driven out of the expression and into a single term of its own.

The first term is the **variational free energy** of the approximate posterior $q$, written $F$:

$$F = \sum_s q(s) \log \frac{q(s)}{p(o,s)}$$

Everything in it is available. $q(s)$ is the distribution the agent put forward. $p(o,s)$ is its own generative model, read at the observation that arrived. There is nothing else in it.

Substituting gives the relation in its final form:

$$D_{\mathrm{KL}}\!\left[q(s) \,\|\, p(s \mid o)\right] = F + \log p(o)$$

Then moving $\log p(o)$ across gives

$$F = D_{\mathrm{KL}}\!\left[q(s) \,\|\, p(s \mid o)\right] - \log p(o)$$

### What the leftover term is

$p(o)$ is the probability of receiving this observation at all, under the agent's model and without reference to where the agent might be.

It depends on the model and on the observation that arrived. It does not depend on where the agent thinks it is. A belief about position, however confident, does not reach back to change how likely the reading was.

So in the relation above:

$$\underbrace{F}_{\text{varies with } q} = \underbrace{D_{\mathrm{KL}}\!\left[q(s) \,\|\, p(s \mid o)\right]}_{\text{varies with } q} - \underbrace{\log p(o)}_{\text{fixed once the observation has arrived}}$$

Change the belief about the states and the KL divergence varies, and so does $F$, by the same amount, because what separates them does not move. Lowering $F$ by some quantity lowers the divergence by exactly that quantity. The agent cannot evaluate $\log p(o)$, and for the purpose of choosing between beliefs it does not have to, because the choice does not depend on it.

What that leaves is a clean division of what the agent knows. It cannot say how far its approximate posterior sits from the true one in absolute terms, since that number contains $\log p(o)$. It can say which of two approximate posteriors is closer. Its own improvement is fully visible to it; its remaining distance from the target is not.

Two consequences.

The $q$ minimising $F$ minimises the divergence, and the divergence is 0 only when $q$ is the posterior. So the approximate posterior with the lowest $F$ is the posterior. An agent that starts anywhere and keeps lowering $F$ ends up holding the posterior, without ever computing it.

And since the divergence is never negative, the relation above gives $F \ge -\log p(o)$. The free energy of any $q$ is bounded from below by a quantity depending on the model and the observation alone, which is what is meant when free energy is called a **bound** on the evidence. Some of the literature writes the same relation negated, as $-F = \log p(o) - D_{\mathrm{KL}}[q(s) \,\|\, p(s\mid o)]$, and calls $-F$ the **evidence lower bound**, or ELBO; maximising it and minimising $F$ are the same thing.

## Seeing it happen

Two ways to watch this, and they show different things.

### Compare situations over one step
Fix the starting point first, since $F$ depends on it. The agent arrives at this moment holding a prior: the belief produced by its last prediction step, spread over a few neighbouring cells rather than concentrated on one. That prior is the same for all five approximate posteriors below, as is the observation. Only $q$ changes from one to the next.

Choose a handful of possible situations the agent might be in, and compute $F$ for each of them, using the same observation throughout.

**The agent has no idea where it is.** Every cell equally plausible. This is what it holds before any evidence arrives.

**The agent is confident and correct.** Most of its probability on the cell the evidence best supports, once the prior is taken into account. Two cells are lit, so the reading alone does not separate them; what breaks the tie is that the agent already thought one of them more likely.

**The agent is confident and wrong.** Just as concentrated, but on a dark cell, one the reading argues against. The kind of state reached after a misread sensor, holding on to a cell the evidence does not support.


**The agent is leaning the right way without committing.** The correct cell favoured, the others still in contention.


And a fifth for comparison: the exact posterior for this observation, obtained by multiplying the prior by the likelihood cell by cell and dividing by the total across every cell, which in a grid this size is still possible.


$F$ is computed the same way in every case: take the approximate posterior, take the prior the agent held, read the column of *A* for the observation received, and evaluate

$$F = \mathbb{E}_{q(s)}\!\left[\log \frac{q(s)}{p(o \mid s)\,p(s)}\right]$$

Nothing is being optimised. Five approximate posteriors are being scored, and the five numbers are then compared. Hovering a row shows the sum it came from, term by term.

<widget id="score-five-q"></widget>


One thing is guaranteed: the exact posterior has the lowest $F$ of the set, since it is the only one at which the divergence is zero. Compare the four other scores against the distributions that produced them and see what the ordering actually is.

### Step by step correction of one situation

The second shows the claim being used rather than verified. Start from a $q$ that is plainly wrong and let an optimiser adjust it step by step, each adjustment making $F$ smaller. That is **descending** on $F$, or gradient descent: the optimiser cannot see where the lowest point is, only which direction is downhill from where it stands.

<widget id="descend-on-f"></widget>

Watch two things together: the shape of $q$ changing, and $F$ falling. $q$ moves towards the posterior, drawn as a dashed outline, and stops on it. The optimiser was never told what that outline is; it was only ever given $F$ to reduce.

The fall is steep and then slow, which is why the panel plots how far $F$ still sits above its floor rather than $F$ itself, on a logarithmic scale. On a plain scale the whole descent happens in the first few steps and the rest is a flat line. That floor is $-\log p(o)$, the quantity the previous section drove out of the comparison: the agent cannot evaluate it, and the figure prints it only so the axis means something to the reader.


## End note

In a grid this size, free energy is not necessary. The posterior can be computed directly by multiplying and normalising, and doing so is less work than searching for it. That is exactly why the demonstration above was possible: the exact posterior was available, so it could be scored alongside the approximate ones and shown to be the one free energy picks out. What free energy buys is that everything shown here still holds when the state count reaches bigger scales, and the sum over every state can no longer be afforded. Direct computation stops being possible at that point. Scoring an approximate posterior does not.

## To go further

- Namjoshi, [*Fundamentals of Active Inference*](https://mitpress.mit.edu/9780262050951/fundamentals-of-active-inference/), Chapter 4, develops variational inference and the different forms of variational free energy used in this module.

- Da Costa et al., [“Active inference on discrete state-spaces: a synthesis”](https://pmc.ncbi.nlm.nih.gov/articles/PMC7732703/), equations (2) and (3), state the free-energy bound and its complexity–accuracy decomposition in the discrete setting.
