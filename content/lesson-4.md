# Why an agent looks before it moves

## The agent cannot yet choose

The agent can hold a belief about where it is, carry that belief through a step, and score one belief against another. Every action so far has come from a list settled in advance. $F$ scores a belief against an observation that has already arrived. It says how well the agent has accounted for what happened, and nothing about what to do next.

In other words, nothing scored an action. 

## Two reasons to move

Fetch me a fork from your own kitchen. You go to the drawer, open it, take a fork. Nothing is uncertain and every step is spent getting there. Call that **exploitation**: acting on what you already know.

Fetch me a fork from a kitchen you have never been in. Same task, but you no longer know where to go, so part of the trip is spent finding out. You open the drawer by the sink, because forks are usually near sinks. That drawer is both the one most likely to hold a fork and the one that tells you most about how this kitchen is arranged. Call the second of those **exploration**: acting to learn something.

Now walk into that kitchen needing nothing. You will still open a few cupboards, and you will first skip the ones you can already guess: the glass-fronted cabinet, the one under the sink. What draws you is the drawer you cannot predict what they contain. Exploration on its own, with nothing but information being sought.

In this page, the agent is given only what it is after. No instruction to explore, no exploration bonus. Those will come later.

## What the agent is after

Consider the exploitation case first: the familiar kitchen, where the agent knows where things are and only needs to know what it is after.

In this framework, what the agent is after is stated in **C**, the **preference**: a distribution over observations, with the fork reading carrying the largest entry.

$$p(o \mid C)$$

One entry per observation, and the entries sum to one. The larger the entry, the more the agent favours receiving that observation. A $C$ concentrated on one reading says the agent cares about that outcome and little else; a flatter $C$ says several outcomes would do.

Because the entries sum to one, they compete. The agent cannot care more about one observation without caring less about the others, in exactly the way a belief cannot become more confident everywhere at once.

//FIGURE 1: C shown as a distribution over the observation set, beside the grid. Static, nothing moves. Alongside it, a reward vector over cells, so the reader sees the two objects are different shapes: one indexed by observation, one by state. Label which is which. This figure exists to make the difference visible rather than asserted. If cheap, add a third panel showing preferences over states, the variant named below, so the reader can see all three object shapes at once.//

That competition is the first thing separating $C$ from reward, and the second is what it is indexed by.

A reward is a real number attached to a state. For instance, reaching the goal cell is worth ten, every other cell is worth nothing, and nothing constrains those numbers: one can be a hundred times another, they need not sum to anything, and they can be negative. The quantity says how good it is to be somewhere. This reward comes from the environment: in reinforcement learning the agent discovers it by acting and cannot change it.

In contrast $C$ comes from the agent, sitting inside the generative model alongside *A* and *B*. The world never signals that the agent has arrived; the agent has to recognise the reading it was after. So $C$ says how much the agent favours seeing something. Where several cells produce the same reading, the agent favours the reading, and any cell that produces it will satisfy the preference equally. In the grid world previously presented, if the agent desires to stay under the light, any lit tile would do.

In this tutorial preferences are placed over observations. They can equally be placed over states, as a prior over where the agent would like to end up, and a good deal of the literature does exactly that. The two are not the same thing written differently, and where the difference matters is taken up in a later section.

## A model you act on

With $C$ in hand, the generative model $p(o,s)$ from the previous section acquires a second use.

It was built to explain readings that had already arrived: given this observation, which states could have produced it. Used that way, the agent consults the model and the model reports. 

Put $C$ alongside the model and you have a description of a situation the agent longs to be in. It is a target, and the agent act until the readings it receives look like the ones the model says it should be receiving.

In most frameworks the model and the goal are separate objects, and a planner is what connects them. Here the goal sits inside the model, which means that acting to reach it and inferring where you are stop being two different operations.

What remains is to turn that into a number, so that one action can be compared against another.


## Scoring an action that has not happened

$F$ needs an observation. It scores a belief against an observation that arrived, and every quantity in it is fixed once that reading is in hand.

An action the agent is considering has no observation attached. $F$ cannot be evaluated for it.

But the agent is not without resources. It has *B*, so it can work out where the action would put it. It has *A*, so it can work out what it would be likely to see from there. What it does not have is the one observation that will actually arrive.

This is the situation from the prediction step. An unknown quantity is averaged over, weighted by how likely each of its values is. The agent cannot say what it will see, but it can say how likely each observation is to be seen under the action it is considering.

So the observation moves inside the expectation. In $F$ the observation sat outside, fixed; here it varies, and the agent averages over it. That single change is what turns a score for a belief into a score for an action.

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

//FIGURE 2: score one policy by hand, every term shown separately, on the same grid and starting belief as Figure 1. Nothing moves. Each number traceable to its source: the belief, B, A, and C. Show q(s_tau|pi) as an intermediate quantity before the score, since the prose introduces it separately.//



## One action is not enough

The agent can now score an action. But scoring a single step will not produce the behaviour this section set out to explain.

Go back to the unfamiliar kitchen. Opening the drawer by the sink is a step that gets you no closer to holding a fork: the drawer may be empty, and even if the fork is there you still have to take it. Judged on that step alone, walking towards where you guess the fork is scores better. The drawer is worth opening only because of what comes after it.

The same holds in the grid. One step towards a lamp is one step not taken towards the goal. Score that step on its own and going to the lamp loses. The detour only pays once the steps after it are counted, because that is when the sharper belief gets used.

So the agent scores sequences rather than single actions. A sequence of actions is called a **policy**, and $\pi$ now denotes one. The distinction is worth keeping: an action is something that changes the world, while a policy is a hypothesis about a way of behaving, one of several the agent is weighing up.

The score for a policy is the sum of its per-step scores over the horizon:

$$G(\pi) = \sum_\tau G(\pi, \tau)$$

where $\tau$ runs over the steps the agent is planning across. Each term is the quantity built above, evaluated at that step: the prediction carried one step further through *B*, *A* applied to it, and the model's joint at that point.

One consequence follows from the previous section and is worth stating before it causes confusion. Each step's prediction is built from the one before it, and every pass through *B* spreads the belief. So the prediction at step five is vaguer than the prediction at step two, and everything computed from it is correspondingly less sharp. A policy's later steps contribute less that is definite than its earlier ones. Planning further ahead does not simply give the agent more to go on.

//FIGURE 3: three or four policies scored side by side on the same grid and starting belief, including one straight for the goal and one detouring via a lamp. Show G for each. The detour should win here, before the agent has moved. Expose horizon length: at horizon one the detour loses, and the reader should watch it start winning as the horizon grows. If it never wins in the world as built, report that rather than adjusting the world. Show the predicted belief at each step of each policy alongside the score, so the spreading is visible.//

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

**Pragmatic value** is the straightforward one. $q(o_\tau \mid \pi)$ is what the agent expects to see under this policy, built earlier by carrying the prediction through *A*. $p(o_\tau \mid C)$ is what it prefers to see. The term is large and negative when the two line up, so a policy delivering preferred observations lowers $G$. This is exploitation, written down.

**Epistemic value**, read the divergence inside it: $q(s_\tau \mid \pi)$ is what the agent believes about its position under this policy, and $q(s_\tau \mid o_\tau, \pi)$ is what it would believe after receiving observation $o_\tau$. The divergence between them is how much that observation would change the agent's mind. Averaged over the observations the policy might produce, it is how much the agent expects to learn. The minus sign means a policy that would teach it more lowers $G$.


//FIGURE 3b: show that both terms lower $G$, so that the reader sees the two signs behave the same way.

SIGNS ARE CORRECT AS STATED HERE. Do not "fix" them. Both terms carry a leading minus, and every description below is read *after* that minus is applied, not off the sign in front of the expectation. Read the wrong way round they look inverted; they are not. Verified 2026-09-16, see docs/learning-records/0005.

Pragmatic: $-\mathbb{E}_{q(o_\tau \mid \pi)}[\log p(o_\tau \mid C)]$. The better the expected observations match $C$, the more negative this term, so it pushes $G$ down. Show a policy whose expected observations match $C$ against one whose do not.

Epistemic: $-\mathbb{E}_{q(o_\tau \mid \pi)}[D_{\mathrm{KL}}[\cdots]]$. A KL is never negative, so this term is never positive: it is $0$ for a policy that would teach the agent nothing and negative for one that would teach it something. It can only ever lower $G$, never raise it. Show a policy that stays in the dark ($0$) against one that reaches a lamp (negative).

On the lamp world in site-src/widget.js, from a belief spread over the four top-left cells with $C$ preferring LIT: `down` scores epistemic $-0.2193$, pragmatic $+1.6983$; `right`, `up` and `stay` all score epistemic $0.0000$, pragmatic $+2.0829$. Those are the numbers this figure should reproduce.//

An agent scoring $G$ explores, and it explores because of the structure of the quantity rather than because anyone told it to.

To link back to the kitchen example.

A cupboard you can already guess the contents of has low epistemic value, because the observation would barely move the belief. A drawer you cannot predict has high epistemic value. The agent is drawn to what it cannot predict.

Put a high preference on seeing a fork and the pragmatic term enters, favouring policies whose expected observations include one. In your own kitchen that term decides alone: you can predict every cupboard, so no observation would move your belief and the epistemic term is near zero whatever you do. You go straight to the drawer. In a kitchen you do not know, both terms are live, and the drawer by the sink scores on each: it is the one most likely to hold a fork, and it is informative about the layout either way.

And with a flat $C$, the pragmatic term is nearly the same for every policy, so the agent is left choosing on epistemic value alone. It has nothing it is trying to bring about and it still acts, going to the places that would tell it most. That is the third kitchen case, and it falls out of the same expression.

//FIGURE 4: the table from Figure 3 with G split into its two terms per policy. Same policies, same numbers, one more column each. The reader should see which term the detour wins on and which it loses on. No new run. Add a toggle that flattens C and shows the scores recomputed, so the agent's remaining preference ordering is visible when only epistemic value is doing the work.//

> **The other ways $G$ is written**
>
> $G$ can be grouped differently, and the literature uses several forms. They are the same quantity read from different angles, and the chain below is how they are usually set out.
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
> The second line says the same thing in terms of how far the expected observations sit from the preferred ones, and how uninformative the sensor is likely to be at the states the policy visits. The third and fourth are where preferences over states appear, and note the sign: the third line is an inequality, so those forms bound $G$ rather than equalling it. That is why this tutorial keeps preferences over observations.
>
> The second grouping is what most implementations compute, and it is taken up in a later section.



## From scores to a choice

The agent now has a score for each policy. It still has to pick one.

The obvious rule is to take the lowest. That works, and it is what an agent should do when it trusts its own scores. But the scores are built from predictions, and predictions can be wrong: the belief they start from may be broad, and everything computed from it inherits that. A rule that always takes the lowest treats a score of 4.1 against 4.2 as decisively as 1.0 against 9.0.

So the scores are turned into a distribution over policies instead. Negate, exponentiate, normalise:

$$q(\pi) = \sigma\!\left(-\gamma G(\pi)\right)$$

where $\sigma$ is the **softmax** function. Reading it in pieces: negating makes low $G$ into high value, exponentiating makes everything positive, and normalising makes the results sum to one. What comes out is a probability for each policy, and the agent samples from it.

$\gamma$ is called the **precision**, and it sets how sharply the best policy is favoured. At $\gamma$ near zero the distribution flattens towards uniform, and the agent chooses almost at random however different the scores are. As $\gamma$ grows the probability concentrates on the lowest-scoring policy, and in the limit the agent always takes it. So $\gamma$ expresses how much confidence the agent places in its own scores.

//FIGURE 5: the scores from Figure 4 converted to probabilities, with gamma exposed as a control. At low gamma the distribution is near-uniform; at high gamma nearly all probability sits on the lowest-scoring policy. Same table, one more column, nothing re-run.//

$\gamma$ can also be learned rather than fixed, so that the agent's confidence in its own planning adjusts as it goes. And a second prior over policies, written $E$, encodes a bias towards certain ways of behaving regardless of what they score: habits, in effect. We will introduce this in page 6.

## Watching it run

Everything is now in place. The agent holds a belief, predicts where each policy would take it, scores each one, and samples an action.

Here is the world it is in. The agent wakes in the dark, after a hangover, with a rough idea of where it went to sleep. It knows the place well enough: two lamps, identical to look at, and home, marked by a colour on the floor. What it cannot do is see any of them from a distance. This agent is short-sighted to the point of uselessness: the only thing it senses is the cell it is standing on, which reads as lit, dark, or marked (home).

It looks, and sees dark. That rules out the two lit cells and nothing else, so what it is left with is the rough idea of where it went to sleep: it is somewhere in a handful of cells, and it could not tell you which exactly. Worse, that handful straddles the room. The agent is not slightly unsure of its position. It does not know which side of the place it is on.

The agent wants to get home, so $C$ puts almost all its weight on the marking. But since he is unsure of where it is, it can't quite tell where home is in respect to its position.

//FIGURE 6: the only figure with motion.

WORLD SETUP, fixed and shared with Figures 1 to 5:
5x5 grid, toroidal — the agent wraps from row 5 to row 1 and from column 5 to column 1, in both directions. No walls anywhere.
Coordinates are (row, column), 1-indexed, rows increasing downwards, columns increasing to the right.
Two lamps, at (5,3) and (3,5). Identical: both produce the same observation, and the sensor cannot tell them apart.
Home marking at (1,5). Dark otherwise; the marking is a floor colour visible only from that cell.
Every other cell is dark.
Observation set, three values: lit, dark, marked. The sensor reports the cell the agent occupies and nothing else.
Actions: north, south, east, west, stay. B as built in section 2, with the same spread. Stay is the identity.

AGENT'S TRUE POSITION: (4,2). The agent does not know this. The figure may show it, but it must be visually distinct from the belief, since the prose describes what the agent can work out and the true position is something only the reader has.
(4,2) is two steps from both lamps, and a south-then-east sequence reaches (5,3) from it, so the agent's walk genuinely succeeds.

STARTING BELIEF: a cluster of cells including (4,2), each of them at distance two from both lamps, near-equal among themselves and near zero elsewhere. The cluster must satisfy two properties at once, and they were established by running the numbers rather than assumed:
— every cell in it is several steps from any lamp, so that reaching one requires committing to a direction and a random walk does not stumble on light immediately;
— it straddles the two lamps, so that a lit reading leaves a two-peaked posterior at roughly 0.44 on each lamp rather than resolving outright.
A tight cluster around a cell adjacent to one lamp was tested and rejected: the prior rules out the far lamp, a lit reading resolves the position outright, and the two lamps stop being ambiguous. A cluster adjacent to both lamps was also rejected: it gives the two-peaked posterior but makes one step enough, which any random walk would find.

WHAT TO SHOW:
The grid, the belief as a distribution over cells beside it, updated at every step.
For each sequence under consideration, its G split into epistemic and pragmatic terms, so the reader can see which term is responsible for each choice as it is made.
Both branches at the lamp: the lit outcome, giving the two-peaked belief, and the dark outcome, which eliminates most of the cluster. The prose makes a point of the failed expectation being informative and the reader should see it happen.
The reader selects one comparison agent, so only two are on screen at once:
— the reward maximiser
— tabular Q-learning
— nothing: a clean run of the active inference agent alone.
Horizon and gamma remain available, since the reader has met each separately.
Also expose the flat-C toggle from Figure 4, so the reader can watch the agent with nothing to seek.

TO VERIFY RATHER THAN ASSUME:
Whether a second sequence exists that separates the two remaining candidates after the first lamp, and how many steps it takes. The prose claims one does. If the torus makes the two candidates symmetric under everything the agent can do, that paragraph goes rather than the world.
Whether the epistemic term genuinely separates directions from the starting cluster, so that the sequence ending at a lamp scores clearly above the ones that do not. If every direction scores alike, the cluster is wrong.
Whether the sequence the agent would pick on pragmatic value alone ever wins, at any horizon or gamma.
At what horizon the lamp sequence starts to beat heading straight for home.
Report what the numbers give. If a claim in the prose does not hold, report that rather than adjusting the world.//


//REWRITE BELOW BASED ON WHAT WE ACTUALLY LEARN FROM THE MOTION IN WORLD//

**Heading for home.** The agent has a rough idea where it is, so it has a rough idea which way home lies. It can set off that way. The trouble is that "that way" is only right if its rough idea is right, and what it has is not one idea but several, pointing in different directions. The agent knows this about itself, which is why setting off looks only moderately good to it. And nothing is learned on the way: the cells between here and there are dark, dark is what it expected, and expecting dark and then seeing dark does not inform the agent if it's getting closer or further.

**Walking until it finds a lamp.** There is a direction that pays. Two steps, south then east, and the agent may be standing under a lamp. Not every direction does this: walk the other way and there is nothing to find, and the agent spends the same two steps and arrives no wiser. Committing to the direction that could end somewhere lit is the whole of what this sequence is for.

If it finds one, that rules out every dark cell in the grid at once. If it does not, it still learns where it was not, and the observation it expected and did not get does as much as the one it wanted.

The catch is that the two lamps look the same, so the agent would leave knowing it is under a lamp and no wiser about which. Its belief would narrow to two, one peak on each lamp, and it would still not know which side of the place it is on. 

**Finding a lamp, then working out which.** After the first lamp the agent is down to two candidates, far apart, and from each of them the grid looks different a few steps on. A second sequence, chosen so that what it produces depends on which candidate is true, would settle it. That costs more steps, and while it is happening the agent gets no nearer home.


The reward maximiser has the same map, the same *A*, the same *B*, and the same starting belief. It computes the route that collects the most reward in expectation and walks it. That is the first policy above, and it fails for the reason given: a route computed from a broad belief is a gamble, and nothing in what this agent scores can represent that. It does not plan badly. It plans from an uncertain belief and acts as though the belief were sharp.

//PLACEHOLDER: explain the reward maximiser. Needs to cover what a reward is (a scalar attached to a state, supplied by the environment, discovered by acting), how the agent scores an action by the reward it expects to collect under its current belief, and a pointer out for readers who want the method properly. Keep it short: this tutorial does not teach reinforcement learning, and the agent is here only as a contrast. Note also that the contrast is not good method against bad one — minimising expected free energy and maximising expected reward coincide under conditions, and that should be acknowledged rather than glossed.//

A note on the third option. Tabular Q-learning assumes the state is observed, and here it is not, so this is not a fair comparison and it is not meant to be. What it shows is that the assumption is what breaks: an agent that treats its observation as its state cannot represent being unsure where it is, so it has nothing to resolve and no reason to look. Reinforcement learning has methods that carry belief states or use recurrent policies, and they do considerably better. This tutorial does not cover them. //CHECK IF THAT IS TRUE WITH WORLD AND MOTION//

## The detour was not added

That detour was not put into the objective. Nobody wrote a rule saying to visit a lamp before the marking, or set a bonus on lit cells, or traded exploration against exploitation with a coefficient.

What was specified is one score, and inside it a statement of what the agent wants to see. The rest came from the structure of the quantity: because the observation is unknown and has to be averaged over, the score separates into a term about what a policy would deliver and a term about what it would reveal. 

An agent that scores outcomes must be told to explore. An agent that scores expected free energy explores because of what the score is.


///possible sources

Smith, Friston and Whyte, "A step-by-step tutorial on active inference and its application to empirical data". Worked examples with code, including an explore-exploit task, and it derives the decompositions in an appendix. Closest in spirit to what you are writing, and the natural next step for someone who has finished this section.

Namjoshi, "Fundamentals of Active Inference", chapter 10. The two-armed bandit worked through with actual numbers: two slot machines with hidden payout probabilities, plus a hint the agent can pay for. A risk-averse agent skips the hint and pulls levers; a less risk-averse one takes hints first and then exploits. That is the espresso example and your lamp detour in a third setting, and the chapter shows the 
G
G traces for each policy over time.

The bandit is the better pointer for this section specifically, because it isolates the same trade in a world with no map: the hint costs money and buys only information.

For the reinforcement learning side, the standard text is Sutton and Barto, Reinforcement Learning: An Introduction, which the active inference papers cite throughout. I have not verified an edition or URL, so check that before it goes in.