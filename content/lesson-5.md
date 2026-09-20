# What the agent does not know about the world

## Two tables, never questioned

*A* said what each cell looks like; *B* said where each step leads. The agent has read them, trusted them.

That was a choice, and it has been doing a lot of work. Every belief the agent has held was computed from those two tables. Every score it has assigned to a policy was computed from them too. If they are wrong, nothing downstream is worth anything, and the agent has no way of finding out.

This page removes that.

## Two kinds of not knowing

The agent has been uncertain since the first page, but always about the same thing: which cell it occupies. That uncertainty moves. A step spreads it, an observation sharpens it, and it is different at every moment.

That is **state uncertainty**, and the whole tutorial so far has been about it. The new kind of uncertainty we will see is about **parameter uncertainty**, because *A* and *B* are the parameters of the agent's model. They will not longer be fixed and assumed known.

Those two behave differently. State uncertainty can be lost and regained: a lost agent becomes less lost by looking and more lost by walking without landmarks. Parameter uncertainty does not move that way. The agent reduces its uncertainty by moving somewhere and looking, and walking away does not make it forget what a cell looks like.

## Learning by counting

Suppose the agent does not know one column of *A*: there is a cell it has no information about.

It can find out by standing there and looking. Seeing lit from that cell is evidence the cell is lit. Seeing it again is more evidence. Seeing dark once in a while is evidence the sensor is unreliable rather than evidence the cell is dark, still it will be considered.

So the agent keeps a count. One number for each state-observation pair, and when it observes $o$ while believing it is at $s$, it adds to the count for that pairing:

$$a \leftarrow a + o \otimes q(s)$$

Write $a$, in lower case, for the counts, and keep *A* for the observation model the agent plans with. They are different objects: $a$ grows without limit and carries the confidence, *A* is the columns of $a$ normalised. The literature uses the same convention, and it uses $b$ and *B* the same way.

The addition is weighted by the belief, because the agent is not certain where it is. If it is confident it stands at one cell, nearly all of the count lands there. If its belief is spread across four cells, the count is shared among them.

The estimate of *A* is the counts, normalised:
$$A_{ij} = \frac{a_{ij}}{\sum_i a_{ij}}$$

where $A_{ij} = p(o = i \mid s = j)$, as in section 1: $i$ the observation, $j$ the state.

Nothing is deferred. The counts change at every step, and so does the estimate the agent plans with. It is learning while it works, not between attempts.

## The counts carry two things

Normalise a column and you get the estimate. What normalising throws away is the size of the numbers.

Compare two columns. One holds the counts $[0.1, 0.4]$. The other holds $[20, 80]$. Normalise each and they are identical: $[0.2, 0.8]$, lit with probability 0.8 in both cases.

Now add one observation of dark to each. The first column becomes $[1.1, 0.4]$, which normalises to $[0.73, 0.27]$. The second becomes $[21, 80]$, which normalises to $[0.21, 0.79]$.

Same estimate, entirely different response to evidence. That difference is confidence, and the agent keeps no separate record of it. It is the size of the counts.

That is the property the rest of this page is built on. A column the agent has visited a hundred times barely moves. A column it has never visited swings on a single reading.

//FIGURE 1: one column of A, three panels updating together as the reader adds observations one at a time — the raw counts, the normalised estimate, and the novelty value for that column.
Two starting conditions the reader switches between, with identical normalised estimates and different count magnitudes: a0 = 0.5 and a0 = 100, both at shape [0.2, 0.8].
Observation stream: lit, lit, dark, lit, lit, lit, dark, lit.
Verified numbers: at step 0 the small column has novelty 0.330125 and the large 0.004956, a ratio of 66.6. One observation swings the small estimate by 0.1333 and the large by 0.00198, a factor of 67.3. Both start at exactly [0.2, 0.8].
Use the exact Dirichlet KL, never the shortcut — see the note below.//

## What the counts are, formally

A set of counts specifies a distribution over what the column might be, and that distribution has a name: it is a **Dirichlet**, and the counts are its **concentration parameters**, often called pseudocounts. Other methods can be used as well, but in this lesson, we will use this.

 The counts are not the agent's belief about the column. They are the parameters of it. The belief is a distribution over distributions: the agent does not hold one candidate for what the cell looks like, it holds a spread of candidates, and the counts say how wide that spread is.

Why a Dirichlet and not something else: it is the conjugate prior of the categorical distribution, which means that multiplying it by an observation and renormalising gives another Dirichlet. That is what makes the update addition rather than anything harder.

The agent could count without any of this, simply tallying what it has seen and normalising, which is close to what tabular reinforcement learning does with state-action pairs. What it would lose is the spread: it would hold an estimate with no measure of how firmly it holds it, and everything later on this page depends on having that measure.

## Novelty

The agent can now ask a new question. Not "what would this observation tell me about where I am", which is the epistemic term from the previous page, but "what would this observation tell me about my model".

The shape is identical. Take the belief the agent holds about a column now. Take the belief it would hold after standing there and seeing $o$, which is the same Dirichlet with one added to the count at that entry. The divergence between them is how much that observation would change the agent's mind about the world. Averaged over the observations it might receive:

$$\text{novelty} = \mathbb{E}_{q(s_\tau \mid \pi)\,p(o_\tau \mid s_\tau)}\Big[D_{\mathrm{KL}}\big[\,q(\theta \mid o_\tau, s_\tau) \;\|\; q(\theta)\,\big]\Big]$$

where $\theta$ stands for the array being learned.

Same operation as the epistemic term, different belief. There, the belief about position. Here, the belief about the model.

That definition says how to work novelty out. It does not say what the number means: you can evaluate it and still have no picture of what you are holding. But the expectation and the divergence can both be carried out by hand, and what comes out the other side is an expression in quantities the reader already has.
## What novelty actually equals


Adding 1 to entry $o$ changes only that entry, so the divergence collapses to:

$$\ln\frac{\bar{a}}{a_o} + \psi(a_o + 1) - \psi(\bar{a} + 1)$$

where $\bar{a}$ is the column total and $\psi$ is the digamma function, a standard function available in any numerical library. For present purposes the only property that matters about $\psi(x)$ is that it becomes close to $\ln x$ once $x$ is above about 5, and the gap between them keeps narrowing as $x$ grows. which is what makes the correction term shrink as the counts grow.

$$\text{novelty} = H(p) + \sum_o p_o\big[\psi(a_o+1) - \psi(\bar{a}+1)\big]$$


The first term is the entropy of the agent's current estimate of that column. The second is a correction that shrinks as the counts grow.

So novelty is, near enough, the entropy of what the agent currently thinks, discounted by how much it already knows. With no counts the correction vanishes and novelty is exactly the entropy: the agent stands to learn as much as the column is uncertain. With large counts the two terms nearly cancel and novelty goes to almost nothing: there is little left to learn.

Two consequences follow:

**Novelty is bounded.** Write $K$ for the number of observations the agent can receive, which here is three: lit, dark, marked. The entropy of a distribution over $K$ outcomes is largest when all of them are equally likely, and it cannot exceed $\ln K$. Since novelty is that entropy less a correction that is never negative, novelty cannot exceed it either.

So knowing that you do not know is worth at most $\ln K$ nats, and no amount of ignorance is worth more. That is a sensible property, and it is not one you would guess from the definition.

**Novelty depends on the counts alone.** Not on whether the estimate is correct. A column with large counts has low novelty whether those counts are right or wrong. What that implies comes up shortly, and it is not what you would expect.

> **A note on the shortcut**
>
> Implementations usually compute novelty with
>
> $> $$W \approx \tfrac{1}{2}\left(\frac{1}{a} - \frac{1}{\bar{a}}\right)$$
>
> and this appears in the literature without much comment, usually written with $a_0$ in place of $\bar{a}$, which there means the column total and not the entry at index zero. It is a large-count expansion of the expression above, and it is accurate to within 5% only above $\bar{a} \approx 10$.
>
> Below that it fails, and it fails in the region novelty matters most. At $\bar{a} = 0.5$ it overstates by 133%. As counts go to zero it diverges without limit, while the exact form saturates at $\ln K$ as it should. Used carelessly, it turns "the agent knows nothing here" into an unbounded reward for ignorance.
>
> One frequently quoted consequence: a hundredfold novelty ratio between small and large counts. For a uniform column the shortcut reduces to $(K-1)/2\bar{a}$, so its ratio is the count ratio by construction, and the hundredfold is arithmetic rather than a property of novelty. The exact ratio is 66.6 for the column above, 86.3 for a uniform one, and 16.7 for a sharply peaked one. It depends on the shape of the column, which the shortcut cannot see.

## The third term

Novelty is a number attached to a policy, so it goes into the score.

$G$ had two terms. It now has three:

$$G(\pi, \tau) = \underbrace{-\,\mathbb{E}\big[D_{\mathrm{KL}}[\,q(s_\tau \mid o_\tau, \pi) \,\|\, q(s_\tau \mid \pi)\,]\big]}_{\text{epistemic value}} \; \underbrace{-\,\mathbb{E}\big[D_{\mathrm{KL}}[\,q(\theta \mid o_\tau, s_\tau) \,\|\, q(\theta)\,]\big]}_{\text{novelty}} \; \underbrace{-\,\mathbb{E}_{q(o_\tau \mid \pi)}\big[\log p(o_\tau \mid C)\big]}_{\text{pragmatic value}}$$

Nothing about the other two has changed. The agent still scores what a policy would deliver and what it would reveal about its position, and now also what it would reveal about its model.

The signs work the same way throughout. A divergence is never negative, so the novelty term is never positive: zero for a policy that would teach the agent nothing about its model, negative for one that would teach it something. Like the epistemic term, it can only lower $G$.

Three terms, three reasons to prefer one policy over another, and the agent adds them up and takes the lowest. Nothing weighs them against each other; they are in the same units because they are all log probabilities, which is what the previous page's note about log space was for.

One property is worth noting now because it shapes everything the agent does. The epistemic and pragmatic terms depend on where the agent is and where it wants to be, so they change as it moves. Novelty depends on the counts, and the counts only grow. So novelty is largest at the start and declines from there, and an agent carrying all three terms explores early and pursues its preference later, without anyone ordering that sequence.


//BEAT 9 — SECOND DEMONSTRATION: LEARNING A AND B TOGETHER

WHAT THIS HAS TO ESTABLISH:

1. The same machinery handles a second kind of unknown. Four terms, one score,
   no new mechanism.

2. It handles it less well, and the reason is the asymmetry in the prose above.
   The comparison with the first demonstration is the content of this one.

3. The wall is learned, or it is not, and either outcome is worth showing
   honestly. A previous run found the agent never localised well enough to
   learn it at all.

4. The earned leak: once the wall is in the model, being blocked is
   informative about position. Established in closed form at 0.3197 nats
   against exactly 0 without it, with belief on the wall cell rising from 0.04
   to 0.18 from a uniform start. If the live agent never gets there, show the
   closed-form comparison and label it as potential rather than achieved.

HOW:

World: as the first demonstration, plus walls in the works corner. Several,
not one, or the effect is anecdotal.
CONSTRAINTS: the walls must not enclose any cell, or novelty there never
resolves and the demonstration has no ending. They must sit away from both
lamps, since a wall next to one lamp and not the other would let the agent
separate the two candidates by which steps succeed, which is a resolution the
previous page never had.

Both A and B learned, on the same run, at every step.

READER CONTROL: how much of B is unknown, always starting from the same
corner, spanning the same two conditions as before — no counts, and confident
counts from the old layout.

Runs to show:
— A and B error against time, side by side. Claim 2 is whether B lags A, and
  by how much.
— all four terms of G through the run.
— a blocked step, shown as what the agent actually receives, which is the same
  observation again.
— the earned leak, as a before-and-after comparison of what a blocked step
  tells the agent with and without the wall in its model.

TO VERIFY RATHER THAN ASSUME:
— whether B converges at all in a run of reasonable length. A previous pass
  found it diverging monotonically, with the cause isolated to state-inference
  error rather than the update rule: under an oracle that supplied the true
  position, both arrays converged cleanly. If that holds with the pragmatic
  term restored, the honest figure is the oracle comparison, labelled as such,
  and the prose says B does not converge here and why.
— whether novelty over B dominates the score. If it does, check whether the
  cause is the prior admitting implausible destinations before reaching for a
  precision term.
— whether the confident-and-wrong condition is inert for B as it is for A.
  Expect yes, same cause.
— whether the agent ever learns the wall.

Report what happens. Do not tune the world to produce a clean run. A
demonstration that shows B failing, with the reason identified, teaches more
than one arranged to succeed.//


## Two unknowns, one machinery

The agent began this page trusting two tables. It ends it holding beliefs about both.

Nothing was added to make it curious about its model. The novelty term was not invented and bolted on; it is the same divergence the previous page produced, pointed at a different unknown. Where the epistemic term asks what an observation would tell the agent about where it is, novelty asks what it would tell the agent about what the world is like, and the two are the same question with a different subject.

That is the claim this page was built to make. There are two things an agent can fail to know, and the framework does not treat them as different problems. It writes down what it does not know, scores actions by how much of that they would resolve, and lets the arithmetic decide what to do first.

What it does not give the agent is any sense of being wrong. Novelty is a function of the counts, and the counts say how much the agent has seen, not whether what it concluded was correct. An agent that has looked a hundred times and drawn the wrong conclusion has no more reason to look again than one that was right. Correcting that requires something this page does not contain.