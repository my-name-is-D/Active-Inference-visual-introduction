# Lesson 1. Where am I?

## Narrowing the flat
The flat turning dark has too much detail for these lessons, each of which focuses on a simpler situation.

A world made of a small number of places, each place a cell in a grid, either dark or lit by a lamp. An agent occupies one of those cells and does not know which. All it can sense is whether the cell it currently stands on is lit or dark.

One thing the agent does have is the map. It knows the layout: which cells are lit, which are dark, and therefore what it would see from any given cell. What it does not know is which of those cells it is standing on. The agent is not exploring an unknown world. It is lost in a world it knows, and the map is what makes the readings interpretable at all: in this example, seeing light would mean nothing unless you already know which places are lit.

That is a scrap of information about where it is, imperfect and not enough to settle the question "where am I exactly?" on its own, but not nothing either. That is the whole world for this lesson: a short strip of tiles, a lamp over some of them, an agent that can tell light from dark and nothing else.

<widget id="legend"></widget>

## Two different kinds of thing
The agent is standing on a tile. Call that tile the **hidden state**. The state is a fact about the world. It is true whether or not the agent knows it, and it does not change because the agent thinks something else. It is called hidden because the agent cannot read it off directly, not because anything is concealing it.

The agent sees light or dark. That is an **observation**: what reaches the agent from where it stands. Which one it gets is not arbitrary. It is determined by the tile it occupies, because that tile is either under the lamp or it is not. Stand somewhere else and you may well see something else. This is the only reason an observation is worth anything: it varies with the state, so it carries information about the state.

It varies, but not uniquely. Several tiles are lit, and they all produce the same observation. So seeing light tells the agent something, and it does not tell the agent it is on tile three. The information is partial. That gap between what the agent receives and what is actually true is called **uncertainty**, and it is not a defect to be engineered away here; it is the permanent condition the agent has to operate in.

Keep the two apart, because almost everything in this tutorial lives in the gap between them. The agent has access to observations. It never has access to the hidden state. If it did, there would be nothing to build. What it can do instead is hold something of its own, assembled from the observations it has received, and the rest of this lesson is about what that something must be.

<widget id="state-vs-belief"></widget>


## Why one answer is not enough
Suppose the agent refuses to carry that uncertainty. It sees light, picks the tile it considers most likely, and treats that as settled: this is where I am, no ambiguity.

Watch what happens when it is wrong. The agent has not moved, but the next reading comes back dark. That can happen: the sensor is unreliable, or the lamp is unreliable, and a lit cell occasionally reports dark. For an agent holding a distribution this is a minor setback. For an agent holding a single answer it is fatal, because the reading contradicts the one cell it committed to, and at the moment it committed it threw away every other cell it might have been on. The alternatives were not ranked and kept, they were discarded.

The agent can only start again from nothing. And starting again does not help: it sees light, it picks a tile, and it is in exactly the position it was in before, with the same information and the same chance of being wrong. Nothing it learned survived the contradiction.

This is what committing fully to one answer costs.

<widget id="commit-vs-distribute"></widget>

## The belief
So the agent will not hold one tile as the truth of where it is. It will hold the probability of being on every tile, all the time.

Concretely: if the world has five tiles, the agent carries five probabilities, one per tile. The larger the probability, the more plausible the agent considers that tile. Together they express not a claim about where the agent is, but a full account of how plausible every position is.

The probabilities must sum to one. In other words, the total is fixed at one and has to be shared out among the tiles, so that becoming more confident about one requires becoming less confident about the others. Without it, the agent could raise its confidence everywhere at once, which corresponds to nothing. Every mechanism presented here depends on that competition.

Call this list of probabilities the **agent's belief**. Two things about it are worth stating plainly.

It belongs to the agent, not to the world. The world has a hidden state. The agent has a belief. They are separate objects and the belief can be wrong.

It is a claim about all tiles at once. Not the best tile with the others implied, all of them, held simultaneously.

A belief with roughly equal probabilities everywhere says the agent has no idea where it is. A belief with one large probability and the rest near zero says the agent is confident. A belief with two large probabilities and the rest near zero says the agent has narrowed the world to two candidates and cannot separate them. That last shape is common and will come up again, and it is not always a halfway stage: if the two tiles look identical to the sensor, that belief is the correct final answer, and the agent will hold it for as long as it stands still.

## The first model: what the world would show
The agent needs to connect observations to states. It cannot do that unless it carries something that says how they relate.

The simplest thing that does the job is a table, and it is the first of two such tables the agent will carry. Call it *A*. For each tile, *A* says what the agent would see if it were standing there. Under the lamp, light is near certain. Away from the lamp, dark is near certain. Tiles at the edge of the pool of light are somewhere in between.

The natural way to read *A* is one tile at a time. Fix a tile. Look down the column belonging to it. What you find is a set of probabilities saying how likely each observation is from that tile. They sum to one, because if the agent is standing there, it will see something.

That is a **conditional probability**, and the notation for it is p(o | s): the probability of observation o, given state s. The bar means given, and everything to the right of it is being held fixed. Reading p(o | s) is exactly what you just did by hand: stand in a column and see how the observations divide up.
The table and the notation are the same object. Writing $A_{ij}$ for the entry in row $i$, column $j$, where the column index is the state and the row index is the observation:

$$A_{ij} = p(o = i \mid s = j)$$

so the column index is the state and the row index is the observation. The sum you read off a column is then

$$\sum_i A_{ij} = 1 \quad \text{for every } j$$

In this world there are only two observations, so *A* has two rows and
twenty-five columns. Take the rows in the order dark, lit. Then every column is
one of two things. A lit tile:

$$A_{:,\,\text{lit}} = \begin{bmatrix} 0.2 \\ 0.8 \end{bmatrix}$$

and a dark tile:

$$A_{:,\,\text{dark}} = \begin{bmatrix} 0.9 \\ 0.1 \end{bmatrix}$$

Both add to one, which is the column rule holding: stand on either tile and you
see something. The whole table is those two columns, repeated. Three of the
twenty-five tiles are lit, so three columns read $[0.2, 0.8]$ and the other
twenty-two read $[0.9, 0.1]$. Nothing else is in *A*.

Read the second entry of each as the sensor's honesty. A lit tile reports light
eight times in ten and lies twice. A dark tile reports dark nine times in ten.
The $0.2$ and the $0.1$ are exactly the false readings this lesson keeps promising.

<widget id="a-columns"></widget>


Each column sums to one. No such statement holds for the rows, which is the point the next section turns on.

So *A* holds one column per tile, each column a proper distribution over observations. Columns are states, rows are observations. Fix that convention now, because it never changes and it is the most common thing to get backwards when reading code or papers.

*A* has a standard name: it is called the **likelihood matrix**, or equivalently the **observation model**. Both refer to this same table. The reason for the first name will be clear after the next section, where *A* is read in the other direction.

Two things about *A* need saying plainly, because everything that follows depends on them and neither is visible in the table itself.

The first is that *A* is given to the agent, not worked out by it. This agent starts with this table and never questions it. That is what it means to know the map. An agent that had to discover which tiles are lit would be solving a different and harder problem, and that problem is not this one.

The second is that the sensor is unreliable (or maybe the lamps are). A lit tile usually reports light, and occasionally reports dark (maybe the sensor failed, maybe the lamp malfunctioned). That is why the columns are distributions rather than single certainties, and it is what makes a false reading possible at all.

The unreliability is fresh each time. When the agent looks twice from the same tile, whether the sensor gets it wrong on the second reading is independent of what happened on the first, so two readings are two separate samples rather than one fact stated twice. This assumption is called conditional independence of observations given the state, and it is what licenses the agent to multiply each new reading into its belief as though it were new evidence. It is standard, and it is not always true of real sensors: a camera pointed at a mirror will misreport the same way every time you look.


## The same table, read the other way
Here is a difficulty, *A* is built to answer a question the agent cannot ask. It answers "if I were on tile three, what would I see?" But the agent does not know it is on tile three. It knows it saw light, and it wants to know which tiles that is evidence for.
So it reads *A* in the other direction. It fixes the observation it actually received and looks across the row: for each tile, how probable would this observation have been there? Tiles where the observation was likely score high. Tiles where it was unlikely score low.
The expression is the same, p(o | s). What has changed is what is held fixed and what varies. Before, s was fixed and you asked about o. Now o is fixed and you run across s.
One warning. That row does not sum to one and is not a distribution. It is a list of scores, one per tile, saying how well each tile explains what happened. Nothing requires those scores to add to anything in particular, because they answer separate questions about separate tiles. Only the columns of *A* are distributions.

<widget id="a-rows"></widget>

## Combining
The agent now has two things: what it believed before the observation, and a score per tile from the observation.

Combine them by multiplying tile by tile. A tile ends up with a large number only if the agent already considered it plausible and it explains what was just seen. A tile that was plausible but explains the observation badly is pulled down. A tile that explains the observation well but was nearly ruled out stays low, though less low than before.

Written out, for each state s:

$$p(s \mid o) \propto p(o \mid s) \, p(s)$$

The symbol $\propto$ means proportional to. It is doing real work here: the left side is what the agent wants, the right side is what it can compute, and the two are not equal, only in proportion to one another.

Put numbers on it. Before the first reading the agent has no reason to prefer
any tile, so the prior is $1/25 = 0.04$ everywhere. Now light arrives. Take the
lit row of *A*, which is $0.8$ on a lit tile and $0.1$ on a dark one, and
multiply tile by tile:

$$\text{lit tile:}\quad 0.8 \times 0.04 = 0.032$$

$$\text{dark tile:}\quad 0.1 \times 0.04 = 0.004$$

That is the whole of the right-hand side, done twice, because there are only two
kinds of column. The lit tiles come out eight times larger than the dark ones,
which is the ratio $0.8 : 0.1$ surviving the multiplication unchanged: the prior
was flat, so it favoured nobody, and the observation did all the work.

Note what these two numbers are not. Add them all up, three tiles at $0.032$ and
twenty-two at $0.004$, and you get $0.184$, not one. They are proportional to the
answer, which is what $\propto$ has been saying.


Then look at what you have. The resulting numbers do not sum to one. Nothing forced them to; you multiplied two lists together and multiplication does not preserve totals. As they stand, they are not a belief.

The fix is to divide every number by their total. This is called **normalisation**, and it is not a tidying step. It is what enforces the competition. Only the relative sizes of the numbers carried any meaning, and dividing through keeps every ratio intact while restoring the constraint that probability is finite and shared out. A tile gains only at the expense of the others.

Dividing by the total turns the proportionality into an equality:

$$p(s \mid o) = \frac{p(o \mid s) \, p(s)}{\sum_{s'} p(o \mid s') \, p(s')}$$

The denominator is nothing more than the total you just computed: the same product, summed over every tile. The primed $s'$ is there only to mark that the sum runs over all states while $s$ in the numerator stays fixed on the one being updated.

The denominator is that same total, written out:

$$\sum_{s'} p(o \mid s')\,p(s') = 3(0.032) + 22(0.004) = 0.096 + 0.088 = 0.184$$

Divide each number through by it:

$$\text{lit tile:}\quad \frac{0.032}{0.184} = 0.174
\qquad
\text{dark tile:}\quad \frac{0.004}{0.184} = 0.0217$$

and check the total: $3(0.174) + 22(0.0217) = 1.000$. That is a belief again.

The ratio is untouched: $0.174$ is still eight times $0.0217$, exactly as $0.032$
was eight times $0.004$. Dividing by a single shared number cannot change how any
two tiles compare. What it changes is what the list means: before the division it
was a set of scores, after it a distribution. And notice that a lit tile went from
$0.04$ to $0.174$ while every dark tile fell from $0.04$ to $0.0217$. Nobody
granted the lit tiles anything. They took it from the others, which is the
competition the fixed total enforces.

<widget id="belief-bars"></widget>


Multiply, then divide by the total. That is the entire update.

## The names

The belief the agent held before the observation is called **the prior**. The scores read across *A* for the received observation are **the likelihood**. The belief that comes out after multiplying and normalising is **the posterior**. The rule that the posterior is proportional to the prior times the likelihood is Bayes' rule, and you have just implemented it.
*A* itself is usually called the likelihood matrix or the observation model. Both mean the table you have been reading.
The posterior of one step becomes the prior of the next. That is the only sense in which the agent has a memory: everything it has learned is in the belief it is carrying.

## What the shapes tell you
Run the update repeatedly and four things happen, all of them readable directly off the belief.

**Repeated evidence sharpens, up to a point.** Each reading is a fresh sample, so consistent readings multiply the same tiles up again and again, and the belief narrows towards the tiles that survive all of them. But look at what it narrows to. If several tiles are lit, every one of them explains a reading of light equally well, so every reading multiplies all of them by the same factor and their relative plausibility never changes. The agent can become entirely certain that it is standing somewhere lit, and no amount of further looking will tell it which of the lit tiles it is on.

**An uninformative observation does nothing.** If a tile is dark and every candidate tile is dark, the likelihood is nearly the same across all of them, and multiplying by a near-constant leaves the ratios where they were. The agent looked and learned nothing. This is not a failure of the machinery; it is the correct response to evidence that fails to distinguish anything.

**An informative observation collapses the belief.** Light in a place where only one candidate is lit gives that tile a much larger probability than the rest, and after normalisation the others are left with a really low probability.

**A false reading tilts the belief without destroying it.** If a wrong observation arrives, the tiles it favours are multiplied up and the belief moves the wrong way. But the tiles that had accumulated support from every previous observation are still there, and the next few correct readings multiply them back to the top. One reading is one factor among many, and it is outvoted.

That last point is the reason for all of this. A single guess is destroyed by one contradiction. A belief is only tilted by it, and the tilt is repaired by ordinary use, without any special handling for mistakes. Nothing in the update knows that an error occurred, and nothing needs to. Keeping the distribution is what buys the recovery.

Worth stating: this is the sequence the widget below steps through. The agent
stands on a lit tile for all seven readings and the sensor reports lit, lit,
dark, lit, dark, lit, lit.

| reading | 1 lit | 2 lit | 3 dark | 4 lit | 5 dark | 6 lit | 7 lit |
|---|---|---|---|---|---|---|---|
| each lit tile | 0.174 | 0.299 | 0.220 | 0.313 | 0.258 | 0.322 | 0.332 |
| each dark tile | 0.0217 | 0.0047 | 0.0155 | 0.0028 | 0.0102 | 0.0016 | 0.0002 |

Three things are readable straight off the top row. It climbs while the readings
agree. It falls at readings three and five, the two false ones, and it falls
without collapsing: $0.299$ to $0.220$ is a setback, not a contradiction, and the
next reading more than repairs it. And it converges towards a third, not towards
one, because three tiles are lit and no amount of standing still separates them.
A third each is the correct answer here, and the agent is most of the way to it.

<widget id="sequence-stepper"></widget>


One restriction has been in force throughout, and it is worth stating before it is lifted. The agent has not moved. Every reading came from the same tile, and the belief was revised on the assumption that the tile did not change between readings. That assumption is what makes the sequence of multiplications correct, and it is also what puts the ceiling on how much the agent can learn: it can resolve the noise in its sensor, but it cannot resolve an ambiguity that is built into where it happens to be standing. Moving is what breaks that, and moving needs machinery this lesson does not have.

That is what your hand was doing in the dark when it found a table and you carried on walking.

