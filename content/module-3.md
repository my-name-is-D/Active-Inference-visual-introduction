# Module 3. Free energy
## The step that gets expensive

Go back to the update the agent performs when an observation arrives. It multiplies the belief it held beforehand by the likelihood read from *A*, and then divides by the total:

$$p(s \mid o) = \frac{p(o \mid s) \, p(s)}{p(o)} \tag{1}$$

Throughout this section, $s$ denotes the agent’s current state. The prediction step has already produced the prior $p(s)$, and an observation has just arrived.

For any particular state, the numerator is local:

$$
p(o \mid s)\,p(s). \tag{2}
$$

It requires only one likelihood from *A* and the prior probability of that state.

The denominator is global:

$$
p(o)=\sum_s p(o \mid s)\,p(s). \tag{3}
$$

This global sum is cheap in a small grid but grows rapidly with the state space. If the agent tracks 100 locations, 4 orientations, and 8 lamps that may each be on or off, it must consider $100 \times 4 \times 2^8 = 102{,}400$ possible states for every observation. Each additional variable multiplies that number.

Exact Bayesian updating therefore becomes impractical as the model grows. This motivates **variational inference**: instead of calculating the posterior directly, the agent searches for a tractable distribution that approximates it. Free energy provides the score that guides that search.


## The posterior as a target

Up to now the posterior has been a result. The agent had a prior, an observation arrived, and the posterior was what came out of the arithmetic.

Turn that around. Treat the posterior as a target: the distribution the agent would like to hold.

This changes how the problem is posed. Instead of calculating the posterior exactly in one operation, the agent searches for an approximation and improves it step by step. It can stop when further computation would improve the approximation too little to be useful.


## The approximate posterior

To search for the posterior, the agent needs a distribution it can adjust. Call this distribution $q(s)$. It assigns a probability to every state and sums to one, just like any other belief.

Unlike the beliefs introduced so far, $q$ is not obtained directly from Bayes’ rule or from *B*. It is a candidate solution. The agent can change it and compare the new candidate with the previous one. It is called the **approximate posterior**, or **variational distribution**.

The word *approximate* describes its role, not its quality. A particular $q$ may match the posterior closely or poorly. The task of variational inference is to adjust $q$ until it is the best available approximation.

In this grid, $q$ may be any point on the **probability simplex**: the set of all distributions over the cells. In larger models, the agent usually searches within a restricted family of distributions. That restriction makes the search tractable, although it may prevent $q$ from matching the exact posterior perfectly.

A three-state example makes this search space visible. Each corner of the triangle represents certainty about one state, while the centre represents the flat belief $(1/3,1/3,1/3)$. Every point inside is a valid distribution.

Drag the blue point $q$. The bars show the same belief as three probabilities, which remain non-negative and sum to one. The marked posterior is the target and moving $q$ means testing a different approximation. The optional scores connect this picture to the KL divergence and free energy introduced below.

<widget id="q-lives-in-the-simplex"></widget>

The next step is to define what makes one candidate better than another. We will first describe closeness to the posterior directly, then deal with the fact that the posterior itself is unavailable.

## How far apart are two distributions?

An approximate posterior is only useful if the agent can tell a good one from a bad one, and a good $q$ means one that is close to the posterior.

Both $q(s)$ and $p(s\mid o)$ are distributions over all possible states. To compare them, begin with one particular cell $s_i$. The approximate posterior assigns that cell probability $q(s_i)$, while the exact posterior assigns it probability $p(s_i\mid o)$.

Their ratio compares the two assignments:

$$\frac{q(s_i)}{p(s_i \mid o)} \tag{4}$$

The ratio is one when the distributions agree about the cell. It is greater than one when $q$ assigns the cell more probability than the posterior does, and less than one when it assigns less.

Taking the logarithm turns the ratio into a score that is zero when the two distributions agree about the state:

$$\log \frac{q(s_i)}{p(s_i \mid o)} \tag{5}$$

This is a signed contribution: it is positive when $q(s_i)>p(s_i\mid o)$ 
and negative when $q(s_i)\lt p(s_i\mid o)$. Only the sum over all states, introduced next, is guaranteed to be non-negative.

That is one cell. To get a single number for the whole distribution, the contributions have to be combined, and they cannot simply be added, because a cell $q$ considers irrelevant should not count as much as one it is staking everything on. So weight each cell's contribution by the probability $q$ assigns to it, which is an expectation under $q$:

$$D_{\mathrm{KL}}\!\left[q(s) \,\|\, p(s \mid o)\right] = \mathbb{E}_{q(s)}\!\left[\log \frac{q(s)}{p(s \mid o)}\right] = \sum_s q(s) \log \frac{q(s)}{p(s \mid o)} \tag{6}$$


This is the **Kullback-Leibler divergence** from $q$ to $p(s \mid o)$, written $D_{\mathrm{KL}}$ and usually just called the KL divergence. It is the same operation as the prediction step: a value for every state, a weight for every state, multiply and add. The value being averaged is now a log ratio rather than a transition probability.

This divergence measure has three essential properties.

- It is never negative, though this is not obvious: a cell contributes a negative term whenever q is less confident than the posterior, but the weights systematically underweight these terms, with the floor at zero reached only when both distributions match everywhere (Gibbs' inequality, proved via Jensen's inequality on the logarithm's concavity). 
- It equals zero only when the two distributions are identical, and approaches zero as they converge. 
- It is not symmetric: $D_{\mathrm{KL}}[q \| p]$ and $D_{\mathrm{KL}}[p \| q]$ are different, since the weights 
depend on which distribution is written first. This makes it a measure of 
discrepancy rather than a distance; the order in the brackets cannot be 
reversed without changing the value.

The figure below shows these properties term by term. Choose different values of $q$ and compare the last two rows: individual contributions may be negative, but their total never is. When $q$ matches the posterior, every contribution (and therefore the KL divergence) is zero. Hover over a state to see its calculation.

<widget id="kl-never-negative"></widget>


## Why that gap cannot be measured directly


Equation (6) defines the score we want, but it cannot be used directly. The agent knows its candidate $q(s)$, but evaluating the KL divergence also requires the exact posterior $p(s\mid o)$. Computing that posterior first would defeat the purpose of approximating it.

Bayes’ rule in Equation (1) shows where the difficulty lies. The prior $p(s)$ and likelihood $p(o\mid s)$ are available; the costly part is the evidence $p(o)$, which requires the global sum in Equation (3).

The next derivation isolates $p(o)$ as a fixed term. What remains is a quantity the agent can use to compare candidate distributions without first calculating the exact posterior.

What follows is a way of separating the divergence into the part that depends on that sum and the part that does not.

## Free energy

Substitute Bayes’ rule for the posterior that appears in Equation (6):

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

Everything needed to compute $F$ is available. The agent supplies the candidate $q(s)$, while $p(o,s)=p(o\mid s)p(s)$ comes from its likelihood and prior. The $p$ in this expression is therefore the joint model, not the unavailable posterior. Neither the evidence $p(o)$ nor the exact posterior $p(s\mid o)$ is needed to evaluate $F$; they appear below only to explain why minimising it moves $q$ towards the posterior.


Substituting gives the relation in its final form:

$$D_{\mathrm{KL}}\!\left[q(s) \,\|\, p(s \mid o)\right] = F + \log p(o)$$

Rearranging gives:

$$
\underbrace{F}_{\text{variational free energy}}
=
\underbrace{
D_{\mathrm{KL}}\!\left[q(s)\,\|\,p(s\mid o)\right]
}_{\text{posterior divergence}}
+
\underbrace{
-\log p(o)
}_{\text{surprise, or negative log evidence}}
$$
### What the leftover term is

$p(o)$ is the probability of receiving this observation at all, under the agent's model and without reference to where the agent might be.

The evidence $p(o)$ depends on the observation, the likelihood, and the prior. Once these are fixed, it does not change as the candidate distribution $q$ changes. A belief about position, however confident, does not reach back to change how likely the reading was.

So in the relation above:

$$\underbrace{F}_{\text{varies with } q} = \underbrace{D_{\mathrm{KL}}\!\left[q(s) \,\|\, p(s \mid o)\right]}_{\text{varies with } q} - \underbrace{\log p(o)}_{\text{fixed once the observation has arrived}}$$

For a fixed prior and observation, $-\log p(o)$ is constant. Any decrease in $F$ therefore produces an equal decrease in the KL divergence. The agent can use $F$ to compare candidate distributions without computing the evidence, although it cannot determine its absolute distance from the exact posterior.
Two consequences.

Lower $F$ means that $q$ is closer to the exact posterior. If $q$ may be any distribution, $F$ reaches its minimum when $q=p(s\mid o)$. If the search is restricted to a simpler family of distributions, the minimum instead gives the member of that family closest to the posterior.

Because KL divergence is non-negative,

$$F \geq -\log p(o).$$

Variational free energy is therefore an upper bound on **surprise**, $-\log p(o)$. Negating the relation gives

$$-F=\log p(o)-D_{\mathrm{KL}}\!\left[q(s)\,\|\,p(s\mid o)\right] \leq \log p(o).$$

Thus, $-F$ is the **evidence lower bound**, or ELBO, familiar from variational inference in machine learning. Minimising $F$ and maximising the ELBO are the same optimization written with opposite signs; both move $q$ towards the posterior without first computing the evidence.

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

The second demonstration puts this result to work. It starts from a plainly wrong $q$ and adjusts it by **gradient descent**, taking successive steps that lower $F$. The optimiser is given $F$, not the exact posterior. The figure shows that following this downhill signal nevertheless moves $q$ towards the posterior.

The figure demonstrates the central result of this module: the agent can approach the exact posterior by minimising $F$, without computing that posterior first.

<widget id="descend-on-f"></widget>

Move the pointer across the lower plot to follow the optimization. At each update, the bars show the current $q$; the dashed outline shows the exact posterior for comparison. The optimizer never sees that outline. It only adjusts $q$ to reduce $F$, and $q$ gradually converges towards the posterior.
The curve plots the remaining gap above the minimum,
$$
F-[-\log p(o)]
=F+\log p(o)
=D_{\mathrm{KL}}\!\left[q(s)\,\|\,p(s\mid o)\right],
$$ 
on a logarithmic scale. This makes small improvements near convergence visible. Calculating the gap requires the evidence or exact posterior, so it is shown only as a diagnostic for the reader; the optimizer itself uses $F$.


## What this module established

In this small grid, computing the exact posterior directly is still the simplest method. The purpose of the example was to show the principle: variational free energy turns an unavailable distance from the posterior into an objective the agent can evaluate and minimise.

If the chosen family of $q$ contains the exact posterior, minimising $F$ recovers it. If the family is restricted, it returns the closest approximation available within that family. This becomes computationally useful when the structure of $q$ or the model makes optimisation cheaper than exact inference; free energy does not provide that saving automatically.

So far, the agent has used free energy to revise its belief after receiving an observation. The next module asks how it can evaluate actions before their observations have arrived.

## To go further

- Namjoshi, [*Fundamentals of Active Inference*](https://mitpress.mit.edu/9780262050951/fundamentals-of-active-inference/), Chapter 4, develops variational inference and the different forms of variational free energy used in this module.

- Da Costa et al., [“Active inference on discrete state-spaces: a synthesis”](https://pmc.ncbi.nlm.nih.gov/articles/PMC7732703/), equations (2) and (3), state the free-energy bound and its complexity–accuracy decomposition in the discrete setting.
