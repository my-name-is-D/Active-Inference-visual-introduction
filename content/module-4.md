# Why an agent looks before it moves

## The agent cannot yet choose

The agent can hold a belief about where it is, carry that belief through a step, and score one belief against another. Every action so far has come from a list settled in advance. $F$ scores a belief against an observation that has already arrived. It says how well the agent has accounted for what happened, and nothing about what to do next.

In other words, nothing scored an action. 

## Two reasons to move

Fetch me a fork from your own kitchen. You go to the drawer, open it, take a fork. Nothing is uncertain and every step is spent getting there. Call that **exploitation**: acting on what you already know.

Fetch me a fork from a kitchen you have never been in. Same task, but you no longer know where to go, so part of the trip is spent finding out. You open the drawer by the sink, because forks are usually near sinks. That drawer is both the one most likely to hold a fork and the one that tells you most about how this kitchen is arranged. Call the second of those **exploration**: acting to learn something.

Now walk into that kitchen needing nothing. You will still open a few cupboards, and you will first skip the ones you can already guess: the glass-fronted cabinet, the one under the sink. What draws you are the drawers you can't predict the content of. Exploration on its own, with nothing but information being sought.

The agent receives no separate, task-specific exploration reward. Its information-seeking behaviour comes from the epistemic term already contained in expected free energy.

## What the agent is after

Consider the exploitation case first: the familiar kitchen, where the agent knows where things are and only needs to know what it is after.

In this module, what the agent is after is encoded by *C*: a **preference distribution** over observations. If seeing the fork is the preferred outcome, that observation receives the largest probability:

$$p(o \mid C)$$

Larger probabilities mark more preferred observations. A sharply concentrated $C$ expresses a strong preference for one outcome; a flatter $C$ treats several outcomes as similarly acceptable. Because $C$ is a probability distribution, its entries sum to one.

This is the convention used in this module, not a universal definition of preference. Active-inference models may instead place preferences over states, and reward-based models may assign rewards to either states or observations. Indexing therefore does not, by itself, distinguish $C$ from reward.

Nor must $C$ and reward express different goals. Given reward scores $r(o)$ over the same observations, an equivalent preference distribution can be formed with

$$p(o \mid C) = \operatorname{softmax}(r(o)) \propto \exp(r(o))$$

The widget shows this mapping. Both panels rank dark, lit, and home in the same order; the reward panel uses scores that need not form a probability distribution, while $C$ expresses their relative desirability as probabilities.

<widget id="c-not-reward"></widget>

Converting reward scores with a softmax allows $C$ to encode the same preferences: outcomes with higher rewards receive higher preference probabilities. If $C = \operatorname{softmax}(r)$, then, for a fixed horizon, minimising the preference term selects the same policy as maximising expected reward: the term is negative expected reward plus a constant.

In this module, the reward-based comparison scores policies only by how well they satisfy the task objective. Expected free energy includes the same preference-seeking component, but also an epistemic component: it values observations expected to reduce uncertainty. The active-inference agent may therefore choose an informative action even when it offers less immediate progress towards its preferred outcome. No separate exploration bonus is added here, because information gain is already part of expected free energy.

$\text{preference fulfilment alone}
\quad\text{VS}\quad
\text{preference fulfilment + expected information gain}.$

Placing $C$ over observations also has a concrete consequence. The agent prefers what it expects to **observe**, not a particular cell as such. If several cells produce the same preferred reading, any of them can satisfy the preference. In this grid, for example, a preference for seeing light can be satisfied by any lit cell.

## A model you act on

With $C$ in hand, the generative model $p(o,s)$ from the previous section acquires a second use.

It was built to explain readings that had already arrived: given this observation, which states could have produced it. Used that way, the agent consults the model and the model reports. 

Put $C$ alongside the model and you have a description of a situation the agent longs to be in. It is a target, and the agent act until the readings it receives look like the ones the model says it should be receiving.

In most frameworks the model and the goal are separate objects, and a planner is what connects them. Here the goal sits inside the model, and belief updating and action selection are expressed within the same probabilistic framework.

What remains is to turn that into a number, so that one action can be compared against another.


## Scoring an action that has not happened

$F$ needs an observation. It scores a belief against an observation that arrived, and every quantity in it is fixed once that reading is in hand.

An action the agent is considering has no observation attached. $F$ cannot be evaluated for it.

But the agent is not without resources. It has *B*, so it can work out where the action would put it. It has *A*, so it can work out what it would be likely to see from there. What it does not have is the one observation that will actually arrive.

The future observation is unknown, so the agent considers every observation that could follow the action and weights it by its predicted probability. This produces an expected score for the action without assuming that any particular observation will occur.

So the observation moves inside the expectation. In $F$ the observation sat outside, fixed; here it varies, and the agent averages over it. Averaging over possible observations lets us evaluate an action before its outcome is known. To obtain the goal-directed score used here, we must also introduce preferences into the distribution against which those predictions are scored.

## Expected free energy
Write $\pi$ for the action under consideration, and $\tau$ for a step in the future. So $s_\tau$ is the state the agent would be in at that step, and $o_\tau$ is what it would see there. The current state stays $s$, as before.

Reminder that $p$ is the generative model, *A* and *B*, fixed in advance. $q$ is what the agent works out from them, given where it currently thinks it is. In the chain below, every $p$ is a table the agent was given and every $q$ is something it computed.

Start from where the agent would end up following the policy under consideration, obtained by carrying the belief through *B*:

$$q(s_\tau \mid \pi) = \sum_{s} p(s_\tau \mid s, \pi)\, q(s)$$

The expectation in $G$ runs over states and observations together, so attach *A* to that prediction. This gives the agent's prediction of the pair:

$$q(o_\tau, s_\tau \mid \pi) = p(o_\tau \mid s_\tau)\, q(s_\tau \mid \pi)$$

Two steps, both from earlier sections: a prediction through *B*, then *A* read forwards.

Start from $F$, written as an expectation rather than a sum:

$$F = \mathbb{E}_{q(s)}\!\left[\log q(s) - \log p(o, s)\right]$$

This scores the belief the agent holds now against a reading that has arrived. Two things have to change for it to score a policy instead.

**First, everything moves to the future and depends on the policy.** The belief becomes the prediction built above, $q(s_\tau \mid \pi)$, and the model's joint becomes its account of the same future pair, $p(o_\tau, s_\tau \mid \pi)$:

$$\mathbb{E}_{q(s_\tau \mid \pi)}\!\left[\log q(s_\tau \mid \pi) - \log p(o_\tau, s_\tau \mid \pi)\right]$$

This is not yet computable. It still contains $o_\tau$, and $o_\tau$ has not happened.

**Second, the unknown observation is averaged over.** The agent cannot say which $o_\tau$ will arrive, but it has a distribution over them: $q(o_\tau, s_\tau \mid \pi)$, the joint prediction built above. Taking the expectation under that instead extends the average across observations as well as states:

$$G(\pi, \tau) = \mathbb{E}_{q(o_\tau, s_\tau \mid \pi)}\!\left[\log q(s_\tau \mid \pi) - \log p(o_\tau, s_\tau \mid \pi)\right]$$

That is expected free energy. Every quantity in it has already been constructed: $q(s_\tau \mid \pi)$ is the state prediction obtained throug *B*, and combining this prediction with *A* gives $q(o_\tau, s_\tau \mid \pi)$ . The remaining term $p(o_\tau, s_\tau \mid \pi)$, is the generative model’s account of the same future state and observation under the policy.

The average now runs over both possible states and possible observations, but the term inside it remains $\log q(s_\tau\mid\pi)$: the agent’s predicted belief about its state. It does not become $\log q(o_\tau,s_\tau\mid\pi)$. The observation is included only because the agent must average the score over everything it might observe; it is not added to the belief being evaluated.

For now, $p(o_\tau,s_\tau\mid\pi)$ remains a single joint distribution, and $C$ has not yet entered the expression. Later, factoring this joint will separate $G$ into two parts: one concerning what the agent expects to learn, and another comparing predicted observations with its preferences $p(o_\tau\mid C)$.
Until then, $G(\pi,\tau)$ can be treated as a single score for an action. The agent computes it for each available action and prefers the one with the lowest value, just as a lower $F$ indicated a better explanation of an observation.



## Scoring a whole policy

The agent can now score an action. To account for what happens after that action, it scores a sequence of steps too.

### Fixed action sequences

Go back to the unfamiliar kitchen. Opening the drawer by the sink is a step that gets you no closer to holding a fork: the drawer may be empty, and even if the fork is there you still have to take it. Whether the drawer is worth opening depends on what the next steps could achieve.

The grid now gives the agent a concrete objective: $C$ strongly favours the home observation, produced at the cell marked `H`. Active inference is often illustrated with an information-seeking detour (an action that scores poorly in the short term but becomes worthwhile because what it reveals improves later decisions). This grid behaves differently. Moving east already has the lowest one-step score, partly because it is likely to produce an informative lamp observation. Looking further ahead is therefore not needed to make an initially inferior exploratory action become preferable. Instead, it lets us ask how complete action sequences are scored and whether future actions can depend on what the agent observes.

So the agent scores sequences rather than single actions. Here we first consider fixed policies: sequences of actions are called **policies**, and $\pi$ now denotes one. An action is something that changes the world, while a policy is a hypothesis about a way of behaving, one of several the agent is weighing up.

The score for a policy is the sum of its per-step scores over the horizon:

$$G(\pi) = \sum_\tau G(\pi, \tau)$$

where $\tau$ runs over the steps the agent is planning across. Each term is the quantity built above, evaluated at that step: the prediction carried one step further through *B*, *A* applied to it, and the model's joint at that point.

For a fixed action sequence, each predicted state belief is obtained by carrying the previous one through $B$. Because these predictions are not corrected by future observations, uncertainty about the agent’s position generally accumulates as it looks further ahead. This does not mean that each successive contribution to $G$ must be larger: the score at each future time also depends on which observations the policy is expected to produce, how strongly they are preferred, and how informative they would be.

## Sophisticated Active Inference

**Planning for what you might learn**
A fixed sequence says: open the drawer, then open the cupboard. But what if the drawer contains the fork? The next useful action depends on what you find.
So far, the $G(\pi)$ is computed over a given sequence of action, this mean the agent has scored each sequence with its later actions held fixed. It already favour informative observations through expected free energy. What this calculation does not yet represent is how those observations could change its subsequent actions.

**Sophisticated active inference** makes that dependence explicit. For each possible next observation, the agent works out the belief it would hold after receiving it. From each of those possible beliefs, it evaluates what to do next. It then averages the continuation scores, weighted by how likely each observation is. Planning therefore branches over possible observations and the beliefs they would produce.

Return to the kitchen: “open the drawer; if there is a fork, take it; otherwise, search elsewhere.” The agent evaluates the first action with these different continuations already in view.
Both agents can update their beliefs and replan after a real observation. The distinction is that the sophisticated agent anticipates this future possible updating while choosing its present action.

For two steps, a fixed policy $\pi=(a_1,a_2)$ has the score:

$$G(\pi)=G(\pi,1)+G(\pi,2)$$

To let the second action depend on the first reading, first compute the belief that each possible $o_1$ would produce. This is the same observation update used earlier, now applied to an imagined reading:

$$
q(s_1\mid o_1,a_1)
=
\frac{p(o_1\mid s_1)\,q(s_1\mid a_1)}
     {q(o_1\mid a_1)}
$$

From that posterior, predict through *B* under a candidate second action $a_2$, apply *A*, and score the resulting prediction as before. Denote this second-step score by $G(a_2\mid a_1,o_1)$: the second-step expected free energy, conditional on the first action and reading.

The two-step score for the first action is then:

$$
G_{\mathrm{soph}}(a_1)
=
G(a_1)
+
\sum_{o_1}q(o_1\mid a_1)
\min_{a_2}G(a_2\mid a_1,o_1)
$$

Before taking $a_1$, the agent considers every observation that might follow it. For each possible observation, it computes the resulting posterior and finds the best second action from that posterior. It then weights these branch-specific continuation scores by the probabilities of their observations and combines them to evaluate the best first action $a$. After $a$ is taken and an observation actually arrives, the agent updates its belief and replans from the posterior associated with that observation.

This recursive rule defines **sophisticated planning**. After each possible future observation, the agent updates its predicted belief and selects the lowest-scoring continuation from that posterior. For longer horizons, it repeats the same branching process at every future step.
The per-step expected free energy is unchanged: it still contains the same preference and information terms. What changes is the structure of the plan. A fixed policy evaluates one predetermined action sequence, whereas a sophisticated policy allows later actions to depend on earlier observations.

The figure below compares these two forms of planning. In the sophisticated view, the branches represent alternative beliefs the agent might hold after different observations. They are possible alternatives, not successive points along a single trajectory.


<widget id="policies-compared"></widget>

figure shows three possible first actions: **east**, the best; **north**, the best action not directed toward a lamp; and **west**, the worst. In the fixed-sequence view, each column contains the best complete policy beginning with that action. The rows show its successive predicted beliefs and cumulative score, and the bottom row compares the policy totals. Changing the horizon may change which continuation is best, so an intermediate cumulative score need not equal the score of the best policy ending at that step.


## What the score is made of

$G$ has been a single number so far. Splitting it shows what the agent is actually weighing, and the split is where the preference enters.

Factor the model’s joint into the observation probability and the state conditional on that observation:

$$p(o_\tau, s_\tau \mid \pi) = p(s_\tau \mid o_\tau, \pi)\, p(o_\tau \mid \pi)$$

Substituting that into $G$ and separating the logarithm gives two terms:

$$G(\pi, \tau) = \underbrace{\mathbb{E}_{q(o_\tau, s_\tau \mid \pi)}\!\left[\log q(s_\tau \mid \pi) - \log p(s_\tau \mid o_\tau, \pi)\right]}_{\text{first}} \; \underbrace{-\; \mathbb{E}_{q(o_\tau \mid \pi)}\!\left[\log p(o_\tau \mid \pi)\right]}_{\text{second}}$$

The second term is where $C$ arrives. As it stands it contains $p(o_\tau \mid \pi)$: what the model says the agent will see under this policy. Replace that with what the agent prefers to see, $p(o_\tau \mid C)$, and the term stops measuring what is likely and starts measuring what is wanted. That replacement is a choice rather than an algebraic step, and it is the move that makes the framework goal-directed.

In the first term, factor the predicted joint as $q(o_\tau,s_\tau \mid \pi)=q(o_\tau \mid \pi)q(s_\tau \mid o_\tau,\pi)$, and approximate the model posterior $p(s_\tau \mid o_\tau,\pi)$ with the agent's posterior $q(s_\tau \mid o_\tau,\pi)$. The expectation over states then becomes a KL divergence, leaving an outer expectation over possible observations:

$$G(\pi, \tau) = \underbrace{-\,\mathbb{E}_{q(o_\tau \mid \pi)}\!\left[D_{\mathrm{KL}}\big[q(s_\tau \mid o_\tau, \pi)\,\|\,q(s_\tau \mid \pi)\big]\right]}_{\text{epistemic value}} \; \underbrace{-\; \mathbb{E}_{q(o_\tau \mid \pi)}\!\left[\log p(o_\tau \mid C)\right]}_{\text{pragmatic value}}$$

The states have not disappeared from the calculation: they are summed over inside the KL divergence. The outer expectation therefore runs only over possible observations. The pragmatic term already depends only on observations, comparing what the policy is expected to produce with what $C$ says the agent prefers.

The next two figures use the same starting situation. The world view shows the agent's true position to the reader; the belief map shows the three positions the agent considers possible.

<widget id="scoring-world"></widget>

**Pragmatic value** compares what the policy is expected to produce, $q(o_\tau\mid\pi)$, with what the agent prefers, $p(o_\tau\mid C)$. Each possible observation incurs the cost $-\log p(o_\tau\mid C)$: strongly preferred observations have a small cost, while observations assigned low preference probability have a large one. Averaging these costs under $q(o_\tau\mid\pi)$ gives the policy’s pragmatic contribution to $G$. Policies expected to produce preferred observations therefore receive a lower score. This is exploitation, written down.

<widget id="pragmatic-term"></widget>

**Epistemic value**, read the divergence inside it: $q(s_\tau \mid \pi)$ is what the agent believes about its position under this policy, and $q(s_\tau \mid o_\tau, \pi)$ is what it would believe after receiving observation $o_\tau$. The divergence between them is how much that observation would change the agent's mind. Averaged over the observations the policy might produce, it is how much the agent expects to learn. The minus sign means a policy that would teach it more lowers $G$.

<widget id="epistemic-term"></widget>


<widget id="efe-signs"></widget>

These are the **one-step** scores for the east, north and west actions, using the starting belief and observation preference shown above. East scores better in both comparisons: it has a more negative epistemic contribution *and* a smaller positive pragmatic cost. The leading minus signs in the formula do not make the two displayed terms both negative. The epistemic term cannot exceed zero; the pragmatic term cannot fall below zero while $C$ is a probability distribution.

An agent scoring $G$ explores, and it explores because of the structure of the quantity rather than because anyone told it to.

To link back to the kitchen example.

A cupboard whose contents are already predictable has little epistemic value because opening it would barely change the agent’s belief. An unfamiliar drawer has high epistemic value only if its contents help distinguish between the hidden situations the agent considers possible. Unpredictability alone is not enough: a noisy sensor may produce surprising readings without resolving any uncertainty.

$\text{surprising observation} \neq \text{informative observation}$

Put a high preference on seeing a fork and the pragmatic term enters, favouring policies whose expected observations include one. In your own kitchen that term decides alone: you can predict every cupboard, so no observation would move your belief and the epistemic term is near zero whatever you do. You go straight to the drawer. In a kitchen you do not know, both terms are live, and the drawer by the sink scores on each: it is the one most likely to hold a fork, and it is informative about the layout either way.

And with a uniform $C$ over all three observations, the pragmatic term is **exactly** $\log 3$ at every step, whatever the policy predicts. Across a horizon of $H$ steps, every fixed policy gets the same pragmatic cost $H\log 3$. Differences in $G$ are then entirely epistemic. It has no preferred observation and can still favour actions expected to teach it more. That is the third kitchen case, and it falls out of the same expression.

<widget id="preference-counterfactual"></widget>

At each selected planning horizon, the widget finds the best fixed policy beginning with east, north, and west under the original non-uniform $C$. Changing the horizon can therefore change those sequences. At any one horizon, the $C$ toggle **does not choose new sequences**: it scores the same ones again with uniform $C$. Their epistemic terms stay unchanged, while the pragmatic column becomes identical across all three rows. This isolates what changing preferences does to $G$ without mixing in a change of policy or sensor.

In Sophisticated Active Inference, the agent considers each reading that could follow its first action, updates its predicted belief with that imagined reading, and scores the remaining actions from the resulting belief. Different readings can therefore lead to different continuations. Their scores are averaged according to how likely each reading is. The per-step quantity $G(\pi,\tau)$ still scores expected preferences and information gain; what changes is how the future beliefs and actions used in those scores are computed. Extending a fixed sequence looks further ahead. Sophisticated planning also anticipates how learning along the way could change what the agent does next.

### Preferring where you are

So far, $C$ describes observations the agent wants to receive. But wanting to see a home marker and wanting to be home are different objectives. Home can remain the destination even when it has no distinctive marker.

To express that preference, give $C$ one entry per state instead of one entry per observation:

$$p(s_\tau\mid C)$$

The entries still sum to one. A large entry for the home cell says that the agent prefers to occupy that cell. It does not say that the agent believes it is already there: $p(s_\tau\mid C)$ describes what is wanted, while $q(s_\tau\mid\pi)$ predicts what would happen under a policy.

Recall the pragmatic term for preferences over observations:

$$
-\mathbb{E}_{q(o_\tau\mid\pi)}
\left[\log p(o_\tau\mid C)\right]
=
-\sum_{o_\tau}q(o_\tau\mid\pi)\log p(o_\tau\mid C)
$$

For preferences over states, score the predicted states against the preferred ones instead:

$$
-\mathbb{E}_{q(s_\tau\mid\pi)}
\left[\log p(s_\tau\mid C)\right]
=
-\sum_{s_\tau}q(s_\tau\mid\pi)\log p(s_\tau\mid C)
$$

The calculation has the same form: weight each preference cost by the probability of encountering it. The prediction through *B* already supplies $q(s_\tau\mid\pi)$, so this pragmatic term does not require converting that prediction into observations through *A*.

For example, assign probability $c_{\mathrm{home}}$ to home and the same smaller probability $c_{\mathrm{other}}$ to each other cell. The pragmatic cost becomes:

$$
-q(s_\tau=\mathrm{home}\mid\pi)\log c_{\mathrm{home}}
-
\big[1-q(s_\tau=\mathrm{home}\mid\pi)\big]\log c_{\mathrm{other}}
$$

As the predicted probability of being home rises, this cost falls. The agent evaluates that probability using its belief; it does not need access to its true position.

Keeping the information-gain term gives the state-preference score used here:

$$
G_{\mathrm{states}}(\pi,\tau)
=
\underbrace{
-\mathbb{E}_{q(o_\tau\mid\pi)}
\left[
D_{\mathrm{KL}}\big[
q(s_\tau\mid o_\tau,\pi)
\,\|\,
q(s_\tau\mid\pi)
\big]
\right]
}_{\text{epistemic contribution}}
\;
\underbrace{
-\mathbb{E}_{q(s_\tau\mid\pi)}
\left[\log p(s_\tau\mid C)\right]
}_{\text{pragmatic contribution}}
$$

This changes the objective being scored; it is not an algebraic rewriting of the observation-preference score. With the same belief, *A* and *B*, the epistemic contribution is unchanged. The pragmatic contribution now favours occupying preferred states.

*A* still matters. A lamp reading can help the agent locate itself, even when the lamp cell has no greater preference than any other non-home cell. Home need not have a distinctive reading: observations elsewhere can help the agent infer where it is and choose actions that lead home.

This choice of preference is separate from the choice between fixed and sophisticated planning. Either planner can use preferences over observations or states. The preference specifies what the agent wants; the planning procedure determines how it evaluates the steps ahead.

> **The other ways $G$ is written**
>
> $G$ can be grouped differently. The chain below relates equivalent expressions and an upper bound, under an explicit assumption: the preference-bearing target distribution uses the same observation likelihood *A* as the predictive model. Thus:
>
> $$p(o,s\mid C)=p(o\mid s)p(s\mid C),\qquad p(o\mid C)=\sum_s p(o\mid s)p(s\mid C)$$
>
> Here the observation and state preferences are linked through *A*, rather than chosen independently. With time indices suppressed, the relationship is:
>
> $$G(\pi) = \underbrace{-\mathbb{E}_{q(o\mid\pi)}\big[D_{\mathrm{KL}}[q(s\mid o,\pi)\,\|\,q(s\mid\pi)]\big]}_{\text{epistemic value}} \underbrace{-\,\mathbb{E}_{q(o\mid\pi)}[\log p(o\mid C)]}_{\text{pragmatic value}}$$
>
> $$= \underbrace{\mathbb{E}_{q(s\mid\pi)}\big[H[p(o\mid s)]\big]}_{\text{ambiguity}} + \underbrace{D_{\mathrm{KL}}\big[q(o\mid\pi)\,\|\,p(o\mid C)\big]}_{\text{risk over outcomes}}$$
>
> $$\leq \underbrace{\mathbb{E}_{q(s\mid\pi)}\big[H[p(o\mid s)]\big]}_{\text{ambiguity}} + \underbrace{D_{\mathrm{KL}}\big[q(s\mid\pi)\,\|\,p(s\mid C)\big]}_{\text{risk over states}}$$
>
> $$= \underbrace{-\mathbb{E}_{q(o,s\mid\pi)}\big[\log p(o,s\mid C)\big]}_{\text{expected energy}} - \underbrace{H\big[q(s\mid\pi)\big]}_{\text{entropy}}$$
>
> The first line is the one used above, though you will often see its epistemic term written with the expectation over $o$ and $s$ together. Both are correct.
>
> The second line says the same thing in terms of how far the expected observations sit from the preferred ones, and how uninformative the sensor is likely to be at the states the policy visits. The third and fourth are where preferences over states appear, and note the sign: the third line is an inequality, so those forms bound $G$ rather than equalling it. 
>
> The state-risk expression in this box is distinct from the state-preference score used in our simulations above, which retains expected information gain and replaces the observation-preference cost with a state-preference cost.
>
> For the derivation and its assumptions, see Champion et al., [*Reframing the Expected Free Energy: Four Formulations and a Unification*, sections 5.3–6, especially equation (6)](https://arxiv.org/html/2402.14460v1#S6).
>
> The second grouping is what most implementations compute, and it is taken up in a later section.



## From scores to a choice

The agent now has a score for each policy. It still has to pick one.

The obvious rule is to take the lowest. That works, and it is what an agent should do when it trusts its own scores. But the scores are built from predictions, and predictions can be wrong: the belief they start from may be broad, and everything computed from it inherits that. A rule that always takes the lowest treats a score of 4.1 against 4.2 as decisively as 1.0 against 9.0.

So one way to turn the scores into a distribution over policies is to negate, exponentiate, and normalise:

$$q(\pi) = \sigma\!\left(-\gamma G(\pi)\right)$$

where $\sigma$ is the **softmax** function. Reading it in pieces: negating makes low $G$ into high value, exponentiating makes everything positive, and normalising makes the results sum to one. What comes out is a probability for each policy; an agent using this rule can sample a policy from that distribution.

After selecting a policy, the agent executes its first action, updates its belief with the resulting observation, and plans again from that updated belief.

$\gamma$ is called the **precision** and is positive. It sets how sharply the best policy is favoured. As $\gamma$ approaches zero from above, the distribution approaches uniform, and sampling becomes nearly random however different the scores are. As $\gamma$ grows the probability concentrates on the lowest-scoring policy, and in the limit sampling always selects it. Precision controls how strongly differences in $G$ affect this choice; it does not change those scores.

<widget id="scores-to-probabilities"></widget>

The policy-probability widget illustrates what $\gamma$ does to their probabilities, not a distribution over every possible policy. The probabilities use the full-precision scores, rather than the rounded numbers printed in the table.
$\gamma$ can also be learned rather than fixed, so that the agent's confidence in its own planning adjusts as it goes. And a second prior over policies, written $E$, encodes a bias towards certain ways of behaving regardless of what they score: habits, in effect. We will introduce this in module 6.

## Watching it run

Everything is now in place. The agent holds a belief, predicts where its candidate actions or policies would take it, and scores them. In the run below, the belief-based agents choose an action greedily from their planning scores; they do not sample from the three-policy softmax shown above.

Here is the world it is in. The agent wakes in the dark with a rough idea of where it went to sleep. It knows the place well enough: two lamps, identical to look at, and a home cell. It cannot see any of them from a distance; its sensor reports only the kind of cell it currently occupies.

Its prior allows three cells, one of which contains the red dot. The belief-based agents begin with that prior and receive their first sensor reading after moving. Tabular Q alone needs an initial noisy reading to index its table, as in the comparison experiment. The agents' beliefs change after subsequent readings, though tabular Q does not consult the belief shown for it.

The right-hand selector changes what $C$ prefers: a home-marked reading or occupying the home cell itself.

<widget id="agent-walk-comparison"></widget>

Both tasks assign **C(home)=0.50**: to the home reading under observation preferences, or to the home cell under state preferences. In the second task, home produces the same readings as an ordinary dark cell. Switching tasks therefore changes both the preference and the sensor model *A*. Physical movement is exact, but the agents’ model *B* assigns only 0.6 probability to the intended move, as in module 2.

A lamp can help localisation in either task. Under the observation preference, its reading is also desirable in itself; under the state preference, every non-home cell has the same immediate pragmatic cost.

The five agents differ in how they choose actions:

- **AIF** scores fixed action sequences using preferences and expected information gain, then replans after each reading.
- **Sophisticated AIF** uses the same terms but allows future actions to depend on possible readings.
- **Reward planner** scores fixed sequences using preference utility alone.
- **Belief Q** learns action values from experience, using its full position belief.
- **Tabular Q** learns action values indexed only by its latest reading.

For both Q learners, training provides utility based on the actual state in the hidden-home task. This signal can reveal home visits during training, although the sensor cannot distinguish home during the displayed runs.

All displayed position beliefs use the same prior and Bayesian filter, updated from each agent’s actions and readings. Identical histories therefore give identical beliefs. For **Tabular Q**, this belief is calculated only for display; the learner does not use it.

## Was your run typical?

As one walk cannot answer that, here are results from 300 runs per agent in each task.

<widget id="figure6-curves"></widget>

In the hidden-home task, beliefs become less concentrated on average, yet AIF’s most likely position becomes more often correct. **Confidence and accuracy are different:** increasing certainty does not necessarily mean correct localisation.

AIF reaches home and identifies its position more often than the fixed-sequence reward planner. With the same model, prior, horizon and preferences, this comparison shows the benefit of including expected information gain in this task. Sophisticated AIF performs similarly to AIF here; anticipating observation-dependent actions adds little on these measures in this simple world.

Tabular Q’s results depend strongly on training: in the home-marker task, 2 training seeds produced no successful evaluation runs, while the third achieved 75% success. The plotted interval does not capture this variation across training seeds.

## What the detour shows

No agent was given an explicit bonus for reaching a lamp. AIF's information term can make a lamp route attractive because a lamp reading can narrow its position belief. A reward planner can also value a useful reading when it plans actions contingent on future observations; the fixed-sequence reward planner here does not make such branches. Which route wins depends on the model, preferences, horizon and actual readings, so one walk is not a general performance claim.


## To go further

- Namjoshi, [*Fundamentals of Active Inference*](https://mitpress.mit.edu/9780262050951/fundamentals-of-active-inference/), Chapter 9, especially §§9.4–9.6, develops expected free energy and policy selection; Chapter 10's two-armed-bandit example isolates the trade between paying for information and exploiting what is already known.

- Smith, Friston and Whyte, [“A step-by-step tutorial on active inference and its application to empirical data”](https://www.sciencedirect.com/science/article/pii/S0022249621000973), follows the same ideas into worked simulations, including exploration and exploitation, and derives the expected-free-energy decompositions in its appendix.

- Millidge, Tschantz and Buckley, [“Whence the Expected Free Energy?”](https://direct.mit.edu/neco/article/33/2/447/95645/Whence-the-Expected-Free-Energy), is the critical companion: it examines where the objective comes from and which conclusions require additional assumptions.
