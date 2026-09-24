# Why an agent looks before it moves

## The agent cannot yet choose

The agent can hold a belief about where it is, carry that belief through a step, and score one belief against another. Every action so far has come from a list settled in advance. $F$ scores a belief against an observation that has already arrived. It says how well the agent has accounted for what happened, and nothing about what to do next.

In other words, nothing scored an action. 

## Two reasons to move

Fetch me a fork from your own kitchen. You go to the drawer, open it, take a fork. Nothing is uncertain and every step is spent getting there. Call that **exploitation**: acting on what you already know.

Fetch me a fork from a kitchen you have never been in. Same task, but you no longer know where to go, so part of the trip is spent finding out. You open the drawer by the sink, because forks are usually near sinks. That drawer is both the one most likely to hold a fork and the one that tells you most about how this kitchen is arranged. Call the second of those **exploration**: acting to learn something.

Now walk into that kitchen needing nothing. You will still open a few cupboards, and you will first skip the ones you can already guess: the glass-fronted cabinet, the one under the sink. What draws you is the drawer you cannot predict what they contain. Exploration on its own, with nothing but information being sought.

In this module, the agent is given only what it is after. No instruction to explore, no exploration bonus.

## What the agent is after

Consider the exploitation case first: the familiar kitchen, where the agent knows where things are and only needs to know what it is after.

In this framework, what the agent is after is stated in *C*, the **preference**: a distribution over observations, with the fork reading carrying the largest entry.

$$p(o \mid C)$$

One entry per observation, and the entries sum to one. The larger the entry, the more the agent favours receiving that observation. A $C$ concentrated on one reading says the agent cares about that outcome and little else; a flatter $C$ says several outcomes would do.

Because the entries sum to one, they compete. The agent cannot care more about one observation without caring less about the others, in exactly the way a belief cannot become more confident everywhere at once.

<widget id="c-not-reward"></widget>

That competition is the first thing separating $C$ from reward, and the second is what it is indexed by.

For comparison, take a simple reward vector $r(s)$: reaching the home cell is worth ten and every other state is worth zero. Its entries are real-valued scores, not probabilities. They need not sum to one and could be negative. A reward need not always be indexed by state alone; this state-only example is used here because it makes the contrast in shape visible. It is not an additional object in the active-inference model in this module.

Reward values can also be encoded as preferences, for example through $p(o\mid C)\propto\exp(r(o))$, where $r(o)$ assigns a reward to each observation. For a fixed horizon, the resulting pragmatic cost is negative expected reward plus a policy-independent constant. The information-gain term is the additional component of the score used here.

In contrast $C$ comes from the agent, sitting inside the generative model alongside *A* and *B*. The world never signals that the agent has arrived; the agent has to recognise the reading it was after. So $C$ says how much the agent favours seeing something. Where several cells produce the same reading, the agent favours the reading, and any cell that produces it will satisfy the preference equally. In the grid world previously presented, if the agent desires to stay under the light, any lit tile would do.

In this tutorial preferences are mainly placed over observations. They can equally be placed over states, as a prior over where the agent would like to end up, and a good deal of the literature does exactly that.

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

This is the situation from the prediction step. An unknown quantity is averaged over, weighted by how likely each of its values is. The agent cannot say what it will see, but it can say how likely each observation is to be seen under the action it is considering.

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

That is expected free energy. Every symbol in it has been built: $q(s_\tau \mid \pi)$ from *B*, $q(o_\tau, s_\tau \mid \pi)$ from that and *A*, and $p(o_\tau, s_\tau \mid \pi)$ the model's own joint with the policy attached.

Note what did not change. Inside the brackets the first term is still over states alone: $\log q(s_\tau \mid \pi)$, not $\log q(o_\tau, s_\tau \mid \pi)$. The agent's belief is about where it is, as it has been throughout, and the observation appears only as something to average over. The expectation widened; the quantity being scored did not.

$p(o_\tau, s_\tau \mid \pi)$ is left whole here. How it splits is what produces the two terms this section is building towards.



$C$ does not appear yet. It arrives with that split, attached to the observation term.

Everything else is unchanged. Same belief, same *A*, same *B*, same expectation, same logarithms.

A lower $G$ is better, as a lower $F$ was better. The agent computes it for each action available to it and compares.

<widget id="pragmatic-term"></widget>

<widget id="epistemic-term"></widget>



## Scoring a whole policy

The agent can now score an action. To account for what happens after that action, it scores a sequence of steps too.

### Fixed action sequences

Go back to the unfamiliar kitchen. Opening the drawer by the sink is a step that gets you no closer to holding a fork: the drawer may be empty, and even if the fork is there you still have to take it. Whether the drawer is worth opening depends on what the next steps could achieve.

The grid lets us check whether that intuition actually applies. With its two lamps and the home marker in their present positions, moving east already has the lowest one-step score. A lamp can therefore attract the agent immediately; this world does **not** demonstrate a detour that first loses and later wins. The question for a longer horizon is which *sequence* scores best, and whether later actions can depend on what the agent observes.

So the agent scores sequences rather than single actions. Here we first consider fixed policies: sequences of actions are called **policies**, and $\pi$ now denotes one. The distinction is worth keeping: an action is something that changes the world, while a policy is a hypothesis about a way of behaving, one of several the agent is weighing up.

The score for a policy is the sum of its per-step scores over the horizon:

$$G(\pi) = \sum_\tau G(\pi, \tau)$$

where $\tau$ runs over the steps the agent is planning across. Each term is the quantity built above, evaluated at that step: the prediction carried one step further through *B*, *A* applied to it, and the model's joint at that point.

One consequence follows from the previous section and is worth stating before it causes confusion. Each step's prediction is built from the one before it. In this world, prediction through *B* cannot make the position belief more concentrated; it often spreads it. Later predictions therefore do not automatically become more precise. Their contributions to $G$ can still rise or fall, because they also depend on which observations and preferred outcomes the policy is expected to produce.

## Sophisticated Active Inferen

**Planning for what you might learn**
A fixed sequence says: open the drawer, then open the cupboard. But what if the drawer contains the fork? The next useful action depends on what you find.
So far, the G(\pi) is computed over a given sequence of action, this mean the agent has scored each sequence with its later actions held fixed. It already favour informative observations through expected free energy. What this calculation does not yet represent is how those observations could change its subsequent actions.

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

From that posterior, predict through *B* under a candidate second action $a_2$, then apply *A* and score the next step as before. Write the resulting score as $G(a_2,2\mid a_1,o_1)$: the second-step expected free energy, conditional on the first action and reading.

The two-step score for the first action is then:

$$
G_{\mathrm{soph}}(a_1)
=
G(a_1,1)
+
\sum_{o_1}q(o_1\mid a_1)
\min_{a_2}G(a_2,2\mid a_1,o_1)
$$

The first term scores the first step. The second averages the best continuation from each possible posterior. The minimum sits **inside the sum**: each reading can lead to a different best second action. Readings with zero predicted probability contribute nothing and require no posterior.

This is the lowest-score continuation rule used in the figure. For longer horizons, the agent repeats the same branching and updating at each future step.

The per-step expected free energy retains the same preference and information terms. What changes is how future beliefs and actions enter the calculation. Extending a fixed sequence alone does not make planning sophisticated.


In the figure below, switch from fixed sequences to sophisticated AIF. The branches show alternative beliefs after possible readings. They are different things the agent might come to believe, rather than successive steps along one trajectory.


<widget id="policies-compared"></widget>

On the Figure above, three of the five first actions are shown: **east**, which wins, **north**, the best move that does not begin toward a lamp, and **west**, the worst. In the fixed-sequence view, each column names the best complete policy starting with that action at the chosen horizon. Reading down it shows the predicted position belief after each action and the running sum of that policy's step scores. The bottom row compares the complete-policy totals. East is best even at horizon one and remains best here at longer horizons; the figure does not claim a delayed crossover. Changing the horizon can select a different complete sequence, so a running total at step two need not be the best *two-step* policy for that first action.


## What the score is made of

$G$ has been a single number so far. Splitting it shows what the agent is actually weighing, and the split is where the preference enters.

The model's joint can be factored two ways. Taking it as the observation times the state given the observation:

$$p(o_\tau, s_\tau \mid \pi) = p(s_\tau \mid o_\tau, \pi)\, p(o_\tau \mid \pi)$$

Substituting that into $G$ and separating the logarithm gives two terms:

$$G(\pi, \tau) = \underbrace{\mathbb{E}_{q(o_\tau, s_\tau \mid \pi)}\!\left[\log q(s_\tau \mid \pi) - \log p(s_\tau \mid o_\tau, \pi)\right]}_{\text{first}} \; \underbrace{-\; \mathbb{E}_{q(o_\tau \mid \pi)}\!\left[\log p(o_\tau \mid \pi)\right]}_{\text{second}}$$

The second term is where $C$ arrives. As it stands it contains $p(o_\tau \mid \pi)$: what the model says the agent will see under this policy. Replace that with what the agent prefers to see, $p(o_\tau \mid C)$, and the term stops measuring what is likely and starts measuring what is wanted. That replacement is a choice rather than an algebraic step, and it is the move that makes the framework goal-directed.

With it, and replacing the true posterior in the first term with the agent's own, the two terms are:

$$G(\pi, \tau) = \underbrace{-\,\mathbb{E}_{q(o_\tau \mid \pi)}\!\left[D_{\mathrm{KL}}\big[q(s_\tau \mid o_\tau, \pi)\,\|\,q(s_\tau \mid \pi)\big]\right]}_{\text{epistemic value}} \; \underbrace{-\; \mathbb{E}_{q(o_\tau \mid \pi)}\!\left[\log p(o_\tau \mid C)\right]}_{\text{pragmatic value}}$$

The state has dropped out of the first expectation. The two logarithms have been written as a KL divergence, and the sum over $s_\tau$ is inside it, so what remains depends on $o_\tau$ alone. The second term never had a state in it: the only quantity there is the observation.

**Pragmatic value** is the straightforward one. $q(o_\tau \mid \pi)$ is what the agent expects to see under this policy, built earlier by carrying the prediction through *A*. $p(o_\tau \mid C)$ is what it prefers to see. Because preference probabilities are at most one, this cost is non-negative. It is *smaller* when the expected observations favour what $C$ prefers, and a smaller cost lowers $G$. This is exploitation, written down.

**Epistemic value**, read the divergence inside it: $q(s_\tau \mid \pi)$ is what the agent believes about its position under this policy, and $q(s_\tau \mid o_\tau, \pi)$ is what it would believe after receiving observation $o_\tau$. The divergence between them is how much that observation would change the agent's mind. Averaged over the observations the policy might produce, it is how much the agent expects to learn. The minus sign means a policy that would teach it more lowers $G$.


<widget id="efe-signs"></widget>

These are the **one-step** scores for the east, north and west first actions in Figure 3, using its same starting belief and observation preference. East scores better in both comparisons: it has a more negative epistemic contribution *and* a smaller positive pragmatic cost. The leading minus signs in the formula do not make the two displayed terms both negative. The epistemic term cannot exceed zero; the pragmatic term cannot fall below zero while $C$ is a probability distribution.

An agent scoring $G$ explores, and it explores because of the structure of the quantity rather than because anyone told it to.

To link back to the kitchen example.

A cupboard you can already guess the contents of has low epistemic value, because the observation would barely move the belief. Opening an unfamiliar drawer can have high epistemic value if seeing its contents resolves uncertainty about what is inside.

An observation has epistemic value when it can resolve uncertainty about the hidden state. Unpredictability alone is insufficient: a noisy sensor may produce surprising observations without helping the agent locate itself.


An observation has epistemic value when it can resolve uncertainty about the hidden state. Unpredictability alone is insufficient: a noisy sensor may produce surprising readings without helping the agent locate itself.


Put a high preference on seeing a fork and the pragmatic term enters, favouring policies whose expected observations include one. In your own kitchen that term decides alone: you can predict every cupboard, so no observation would move your belief and the epistemic term is near zero whatever you do. You go straight to the drawer. In a kitchen you do not know, both terms are live, and the drawer by the sink scores on each: it is the one most likely to hold a fork, and it is informative about the layout either way.

And with a uniform $C$ over the three observations, the pragmatic term is **exactly** $\log 3$ at every step, whatever the policy predicts. Across a horizon of $H$ steps, every fixed policy gets the same pragmatic cost $H\log 3$. Differences in $G$ are then entirely epistemic. It has no preferred observation and can still favour actions expected to teach it more. That is the third kitchen case, and it falls out of the same expression.

<widget id="preference-counterfactual"></widget>

The horizon slider selects the best fixed policies beginning east, north and west under Figure 3's original $C$ **for that length**. Changing the horizon can therefore change those sequences. At any one horizon, the $C$ toggle **does not choose new sequences**: it scores the same ones again with uniform $C$. Their epistemic terms stay unchanged, while the pragmatic column becomes identical across all three rows. This isolates what changing preferences does to $G$ without mixing in a change of policy or sensor.

In Sophisticated Active Inference, the agent considers each reading that could follow its first action, updates its predicted belief with that imagined reading, and scores the remaining actions from the resulting belief. Different readings can therefore lead to different continuations. Their scores are averaged according to how likely each reading is. The per-step quantity \(G(\pi,\tau)\) still scores expected preferences and information gain; what changes is how the future beliefs and actions used in those scores are computed. Extending a fixed sequence looks further ahead. Sophisticated planning also anticipates how learning along the way could change what the agent does next.

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

Figure 5 illustrates what $\gamma$ does to their probabilities, not a distribution over every possible policy. The probabilities use the full-precision scores, rather than the rounded numbers printed in the table.
$\gamma$ can also be learned rather than fixed, so that the agent's confidence in its own planning adjusts as it goes. And a second prior over policies, written $E$, encodes a bias towards certain ways of behaving regardless of what they score: habits, in effect. We will introduce this in module 6.

## Watching it run

Everything is now in place. The agent holds a belief, predicts where its candidate actions or policies would take it, and scores them. In the run below, the belief-based agents choose an action greedily from their planning scores; they do not sample from Figure 5's three-policy softmax.

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
