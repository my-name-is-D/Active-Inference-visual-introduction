# What the agent does not know about the world

## Two tables, never questioned

*A* said what each cell looks like; *B* said where each step leads. The agent has read them, trusted them.

That was a choice, and it has been doing a lot of work. Every belief the agent has held was computed from those two tables. Every score it has assigned to a policy was computed from them too. Errors in those tables can distort inference and planning. Unexpected readings can expose a poor prediction, but the agent so far has no rule for updating the tables themselves.

This page removes that.

## Two kinds of not knowing

The agent has been uncertain since the first page, but always about the same thing: which cell it occupies. That uncertainty moves. A step spreads it, an observation sharpens it, and it is different at every moment.

That is **state uncertainty**, and the whole tutorial so far has been about it. The new kind of uncertainty we will see is about **parameter uncertainty**, because *A* and *B* are the parameters of the agent's model. They will no longer be fixed and assumed known.

Those two behave differently. State uncertainty can be lost and regained: a lost agent becomes less lost by looking and more lost by walking without landmarks. Parameter uncertainty does not move that way. The agent reduces its uncertainty by moving somewhere and looking, and walking away does not make it forget what a cell looks like.

## Learning by counting

Suppose the agent does not know one column of *A*: there is a cell it has no information about.

It can find out by standing there and looking. Seeing lit from that cell is evidence the cell is lit. Seeing it again is more evidence. Seeing dark adds evidence for a higher probability of dark readings from that cell. Learning this column alone does not distinguish sensor noise from a change in the cell itself.

So the agent keeps a count. One number for each state-observation pair, and when it observes $o$ while believing it is at $s$, it adds to the count for that pairing:

$$a \leftarrow a + o \otimes q(s)$$

Here $o$ is the one-hot vector for the reading, $q(s)$ is the state belief after incorporating that reading, and $\otimes$ is the outer product.

Write $a$, in lower case, for the counts, and keep *A* for the observation model the agent plans with. They are different objects: $a$ grows without limit and carries the confidence, *A* is the columns of $a$ normalised. The literature uses the same convention, and it uses $b$ and *B* the same way.

The addition is weighted by the belief, because the agent is not certain where it is. If it is confident it stands at one cell, nearly all of the count lands there. If its belief is spread across four cells, the count is shared among them. This fractional update is an expected-count approximation. When the state is known, adding one to its column is exact conjugate learning; with an uncertain state, the exact parameter posterior is generally a mixture of Dirichlets rather than a single Dirichlet for each column.

The estimate of *A* is the counts, normalised:
$$A_{ij} = \frac{a_{ij}}{\sum_k a_{kj}}$$

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
Verified numbers: at step 0 the small column has novelty 0.330054 and the large 0.004956, a ratio of 66.6. One lit observation swings the small estimate by 0.1333 and the large by 0.00198, a factor of 67.3. Both start at exactly [0.2, 0.8].
Use the exact Dirichlet KL, never the shortcut — see the note below.//

## What the counts are, formally

A set of strictly positive counts specifies a distribution over what the column might be, and that distribution has a name: it is a **Dirichlet**, and the counts are its **concentration parameters**, often called pseudocounts. Even before any data, each possible outcome needs a positive pseudocount. “No observations yet” therefore means a prior, not an all-zero column.

 The counts are not the agent's belief about the column. They are the parameters of it. The belief is a distribution over distributions: the agent does not hold one candidate for what the cell looks like, it holds a spread of candidates, and the counts say how wide that spread is.

Why a Dirichlet and not something else: it is the conjugate prior of the categorical distribution, which means that multiplying its density by the categorical likelihood of an observation from a known state and renormalising gives another Dirichlet. That is what makes the update addition rather than anything harder.

The agent could count without any of this, simply tallying what it has seen and normalising, which is close to what tabular reinforcement learning does with state-action pairs. What it would lose is the spread: it would hold an estimate with no measure of how firmly it holds it, and everything later on this page depends on having that measure.

## Novelty

The agent can now ask a new question. Not "what would this observation tell me about where I am", which is the epistemic term from the previous page, but "what would this observation tell me about my model".

The shape is identical. Take the belief the agent holds about a column now. Take the belief it would hold after standing there and seeing $o$, which is the same Dirichlet with one added to the count at that entry. The divergence between them is how much that observation would change the agent's mind about the world. Averaged over the observations it might receive:

$$\text{novelty} = \mathbb{E}_{q(s_\tau \mid \pi)\,p(o_\tau \mid s_\tau)}\Big[D_{\mathrm{KL}}\big[\,q(\theta \mid o_\tau, s_\tau) \;\|\; q(\theta)\,\big]\Big]$$

Here $\theta$ denotes the uncertain observation array, with independent Dirichlet beliefs over its columns, and $p(o_\tau\mid s_\tau)$ is the posterior predictive probability obtained from the normalised counts. The calculation assumes the planning belief factors as $q(s_\tau\mid\pi)q(\theta)$.

This is parameter information gain conditional on the hypothetical state. It asks how much the observation would teach us if we knew which column generated it. The actual agent receives only the observation and uses fractional counts when uncertain about its position. The conditional novelty is a planning quantity, not an exact prediction of the KL change produced by that fractional update.

Same operation as the epistemic term, different belief. There, the belief about position. Here, the belief about the model.

That definition says how to work novelty out. It does not say what the number means: you can evaluate it and still have no picture of what you are holding. But the expectation and the divergence can both be carried out by hand, and what comes out the other side is an expression in quantities the reader already has.
## What novelty actually equals


Adding 1 to entry $o$ changes only that entry, so the divergence collapses to:

$$\ln\frac{\bar{a}}{a_o} + \psi(a_o + 1) - \psi(\bar{a} + 1)$$

where $\bar{a}=\sum_o a_o$ is the column total and $\psi$ is the digamma function. For a single column, put $p_o=a_o/\bar{a}$ and average over its possible observations:

$$N(a) = H(p) + \sum_o p_o\big[\psi(a_o+1) - \psi(\bar{a}+1)\big]$$

Equivalently, if $\vartheta$ is the uncertain probability vector for this column,

$$N(a)=H(p)-\mathbb{E}_{\vartheta\sim\mathrm{Dir}(a)}[H(\vartheta)]$$

The first term is the entropy of the agent's predictive distribution. The subtracted term is the average observation entropy within its candidate columns: uncertainty that would remain even if the column were known. Their difference is what one observation can teach it about the column. Policy novelty averages $N(a_{\cdot s})$ over $q(s_\tau\mid\pi)$.

To describe very small counts precisely, write $a=\varepsilon p$ with a fixed positive probability vector $p$. As $\varepsilon\to0^+$, the subtracted term tends to zero and novelty tends to $H(p)$. At exactly zero counts the Dirichlet and its normalised estimate are undefined. As the total grows at fixed $p$, the subtracted term approaches $H(p)$ and novelty tends to zero. It is the difference that shrinks, not the magnitude of the correction.

Two consequences follow:

**Novelty is bounded.** Write $K$ for the number of observations the agent can receive, which is two in the binary column example and three when the sensor also reports marked. The entropy of a distribution over $K$ outcomes is largest when all of them are equally likely, and it cannot exceed $\ln K$. Since novelty is that entropy less a correction that is never negative, novelty cannot exceed it either.

So knowing that you do not know is worth at most $\ln K$ nats, and no amount of ignorance is worth more. That is a sensible property, and it is not one you would guess from the definition.

**Novelty depends on the counts alone.** Not on whether the estimate is correct. A column with large counts has low novelty whether those counts are right or wrong. What that implies comes up shortly, and it is not what you would expect.

> **A note on the shortcut**
>
> A common large-count approximation to the single-observation KL is
>
> $$W_o \approx \frac{1}{2}\left(\frac{1}{a_o}-\frac{1}{\bar{a}}\right).$$
>
> It follows from $\psi(x+1)=\ln x+1/(2x)+O(x^{-2})$. Its accuracy depends on the individual entries being large, not just the column total. For a uniform binary column at $\bar{a}=10$, the averaged shortcut overstates novelty by about 5.24%; for shape $[0.2,0.8]$ the error is 9.36%, and for $[0.02,0.98]$ it is 89.46%.
>
> At $\bar{a}=0.5$, those errors are about 133%, 203%, and 1201%, respectively. As counts tend to zero at fixed shape, the shortcut diverges, while exact novelty tends to $H(p)\leq\ln K$, reaching $\ln K$ only for a uniform shape.
>
> Averaging the shortcut with $p_o=a_o/\bar{a}$ gives $(K-1)/(2\bar{a})$ for every strictly positive shape. Its ratio between small and large columns is therefore the count ratio by construction: 200 for totals 0.5 and 100. The exact ratio is 66.6 for shape $[0.2,0.8]$, 86.3 for $[0.5,0.5]$, and 16.7 for $[0.02,0.98]$. Exact novelty retains a dependence on shape that this averaged shortcut loses.

## The third term

Novelty is a number attached to a policy, so it goes into the score.

$G$ had two terms. It now has three:

$$G(\pi, \tau) = \underbrace{-\,\mathbb{E}_{q(o_\tau\mid\pi)}\big[D_{\mathrm{KL}}[\,q(s_\tau \mid o_\tau, \pi) \,\|\, q(s_\tau \mid \pi)\,]\big]}_{\text{epistemic value}} \; \underbrace{-\,\mathbb{E}_{q(s_\tau\mid\pi)p(o_\tau\mid s_\tau)}\big[D_{\mathrm{KL}}[\,q(\theta \mid o_\tau, s_\tau) \,\|\, q(\theta)\,]\big]}_{\text{novelty}} \; \underbrace{-\,\mathbb{E}_{q(o_\tau \mid \pi)}\big[\log p(o_\tau \mid C)\big]}_{\text{pragmatic value}}$$

Nothing about the other two has changed. The agent still scores what a policy would deliver and what it would reveal about its position, and now also what it would reveal about its model.

The signs work the same way throughout. A divergence is never negative, so the novelty term is never positive: zero for a policy that would teach the agent nothing about its model, negative for one that would teach it something. Like the epistemic term, it can only lower $G$.

Three terms, three reasons to prefer one policy over another, and the agent adds them up and takes the lowest. This score uses a coefficient of one for each term. All are measured in nats, but sharing units does not require equal coefficients or guarantee comparable magnitudes.

As evidence accumulates, familiar columns often offer less to learn. This can favour exploration early and pursuit of preferences later, but it is not a guaranteed sequence. An unexpected observation can increase a column's novelty: adding one count to $[0.01,10]$ to obtain $[1.01,10]$ raises it from 0.004039 to 0.038392 nats. Policy novelty also depends on which columns the policy is predicted to visit, so it changes with the state belief as well as the counts.


## Learning where steps lead

For *B*, each column describes a destination state $i$, given an origin $j$ and an action $u$. Keep positive counts $b_{iju}$ and normalise over destinations:

$$B_{iju}=\frac{b_{iju}}{\sum_k b_{kju}}.$$

If both states are known, add one to the entry for the observed transition. When they are uncertain, use their joint posterior after the new reading:

$$b_{iju}\leftarrow b_{iju}+q(s_t=i,s_{t-1}=j\mid o_{1:t},u_{1:t-1}),\qquad u=u_{t-1}.$$

This is again an expected-count approximation. Multiplying separate state marginals is a further approximation: uncertainty about where the agent started and where it ended is generally correlated. Knowing only the destination does not identify which transition column to update.

The analogous conditional novelty treats the origin and destination as known for each hypothetical transition, then averages over predicted transitions. With $\theta_B$ denoting the uncertain transition array,

$$N_B(\pi,\tau)=\mathbb{E}_{q(s_{\tau-1}\mid\pi)B_{s_\tau,s_{\tau-1},u}}\left[D_{\mathrm{KL}}\left[q(\theta_B\mid s_\tau,s_{\tau-1},u)\,\|\,q(\theta_B)\right]\right],\qquad u=u_{\tau-1}.$$

For independent Dirichlet columns and the same factorised planning approximation, this is $\sum_j q(s_{\tau-1}=j\mid\pi)N(b_{\cdot ju})$. The column formula is unchanged, but its outcomes are destination states, so its bound is the logarithm of the number of possible destinations. The four-term planning score subtracts both $N_A$ and $N_B$ alongside the state information gain and adds the pragmatic cost. These conditional parameter scores assume access to hypothetical states; they need not equal the parameter information actually recoverable from noisy sensor readings.

//BEAT 9 — SECOND DEMONSTRATION: LEARNING A AND B TOGETHER

WHAT THIS HAS TO ESTABLISH:

1. The same machinery handles a second kind of unknown. Four terms, one score,
   no new mechanism.

2. Test whether learning B is harder than learning A under these conditions.
   Both endpoints must be inferred for B; measure the effect rather than
   assuming a performance ordering.

3. The wall is learned, or it is not, and either outcome is worth showing
   honestly. A previous run found the agent never localised well enough to
   learn it at all.

4. Test whether a learned wall makes the available sensor history more
   informative about position. The previous 0.3197-nat calculation, with wall-cell
   belief rising from 0.04 to 0.1814, conditions on knowing that the agent did
   not move, using B[s,s,u]. That event is not directly observed by this sensor.
   It is a counterfactual with an additional movement signal, not a verified
   information gain for the demonstration. Recompute using actual readings and
   their likelihoods before claiming an earned localisation benefit.

HOW:

World: as the first demonstration, plus walls in the works corner. Several,
not one, or the effect is anecdotal.
CONSTRAINTS: the walls must not enclose any cell, or novelty there never
resolves and the demonstration has no ending. They must sit away from both
lamps to avoid an immediate local asymmetry in predicted sensor histories.
Distance alone does not guarantee that longer observation histories cannot
distinguish the lamps. Check any claimed symmetry in the full model.

Both A and B learned, on the same run, at every step.

READER CONTROL: how much of B is unknown, always starting from the same
corner, spanning two conditions — small positive prior counts, and confident
counts from the old layout.

Runs to show:
— A and B error against time, side by side. Claim 2 is whether B lags A, and
  by how much.
— all four terms of G through the run.
— a blocked step, showing the actual noisy sensor reading. Staying in the same
  cell need not repeat the reading, and a repeated reading does not prove a block.
— any earned localisation benefit, comparing inference from the same available
  sensor history with and without the wall in the model.

TO VERIFY RATHER THAN ASSUME:
— whether B converges at all in a run of reasonable length. A previous pass
  found it diverging monotonically, while an oracle that supplied the true
  position allowed both arrays to converge cleanly. This does not by itself
  validate the uncertain-state update; check joint transition beliefs as well.
  If that holds with the pragmatic term restored, the honest figure is the oracle comparison, labelled as such,
  and the prose says B does not converge here and why.
— whether novelty over B dominates the score. If it does, check whether the
  cause is the prior admitting implausible destinations before reaching for a
  precision term.
— whether large, incorrect prior counts slow learning for B as for A. Low
  novelty does not prevent correction when contradictory evidence is collected.
— whether the agent ever learns the wall.

Report what happens. Do not tune the world to produce a clean run. A
demonstration that shows B failing, with the reason identified, teaches more
than one arranged to succeed.//


## Two unknowns, one machinery

The agent began this page trusting two tables. It ends it holding beliefs about both.

Nothing was added to make it curious about its model. The novelty term was not invented and bolted on; it is the same divergence the previous page produced, pointed at a different unknown. Where the epistemic term asks what an observation would tell the agent about where it is, novelty asks what it would tell the agent about what the world is like, and the two are the same question with a different subject.

That is the claim this page was built to make. There are two things an agent can fail to know, and the framework does not treat them as different problems. It writes down what it does not know, scores actions by how much of that they would resolve, and lets the arithmetic decide what to do first.

Novelty does not directly measure whether the model is wrong. Identical counts give identical novelty whether or not they describe the world accurately. Large counts can therefore make a mistaken column look unpromising to revisit. Contradictory observations can still correct finite counts through the updates already given, provided the agent gathers evidence and assigns it to the relevant states. What this page does not provide is a dedicated mechanism for detecting a changed world or discounting old evidence.
