# Module 1. Where am I?

## Narrowing the flat
The flat turning dark has too much detail for these modules, so we narrow it down to a small grid of cells, they represent the vicinty of your home, by night, some lit by a lamp and the rest dark. An agent stands on one of those cells but does not know which. All it can sense is whether its own cell is lit or dark.

The agent does have the map. It knows which cells are lit, so it knows what it would see from any cell. It is not exploring an unknown world. It is lost in one it knows. The map is what gives a reading its meaning: seeing light tells you nothing about where you are unless you know which places are lit. Even with the map, one reading is only a scrap of information. It cannot answer *"where am I exactly?"* on its own, but it is not nothing.

<widget id="legend"></widget>

## Two different kinds of thing
The agent is standing on a cell. Call that cell the **hidden state**. The state is a fact about the world. It is true whether or not the agent knows it, and it does not change because the agent thinks something else. It is called hidden because the agent cannot read it off directly, not because anything is concealing it.

The agent sees light or dark. That is an **observation** (or reading if we consider the sensor): what reaches the agent from where it stands. It is determined by the cell it occupies, because that cell is either under the lamp or it is not. An observation varies with the state, so it carries information about the state.

But several cells are lit, and they all produce the same observation. Seeing light tells the agent something, but not that it is on cell three. That gap between what the agent receives and what is true is called **uncertainty**. It is not a defect to engineer away. It is the permanent condition the agent works in.


The agent can read observations but never the hidden state. If it could, it would simply know where it is, and there would be nothing to work out. Instead, it must build its own picture of where it might be from the observations it has received. The rest of this module is about what that picture must be.

<widget id="state-vs-belief"></widget>


## Why one answer is not enough
Suppose the agent refuses to carry that uncertainty. It sees light, picks the cell it considers most likely, and treats that as settled: this is where I am, no ambiguity.

Now suppose a reading contradicts it. The agent has not moved, but the next reading comes back dark. That can happen, because a lit cell occasionally reports dark. An agent that kept every cell in play, weighted by how likely it is, just adjusts the weights. An agent holding a single answer is stuck: the reading contradicts the one cell it kept, and it threw away every other cell the moment it committed.

It can only start again from nothing, and that does not help. It sees light, picks a cell, and is back where it was, with the same information and the same chance of being wrong. Nothing it learned survived the contradiction.

This is what committing fully to one answer costs.

<widget id="commit-vs-distribute"></widget>

## The belief
So the agent will not hold one cell as the truth of where it is. It will hold the probability of being on every cell, all the time.

Concretely: if the world has five cells, the agent carries five probabilities, one per cell. The larger the probability, the more plausible the agent considers that cell. Together they express not a claim about where the agent is, but a full account of how plausible every position is.

The probabilities must sum to one. The agent has a fixed amount of confidence to share among the cells, so becoming more confident about one cell means becoming less confident about the others. Without this rule, it could grow more confident about every cell at once, which would mean nothing.

Call this list of probabilities the **agent's belief**. Two things about it:

- It is provisional. Unlike the hidden state (which is fixed in the world and does not change) the belief changes in response to each new observation. The agent revises it as it learns, which means the belief can be wrong, and it will be corrected by subsequent readings.

- It is a claim about all cells at once. Not the best cell with the others implied, all of them, held simultaneously.

Roughly equal probabilities everywhere means the agent has no idea where it is. One large probability with the rest near zero means it is confident. Two large probabilities mean it has narrowed its position to two cells it cannot tell apart. That last shape will come up again, and it is not always a halfway stage. If the two cells look identical to the sensor, that belief is the correct final answer, and the agent will hold it for as long as it stands still.

## The first model: what the world would show
The agent needs to connect observations to states. It cannot do that unless it carries something that says how they relate.

The simplest thing that does the job is a table, and it is the first of two such tables the agent will carry. Call it *A*. For each cell, *A* says what the agent would see if it were standing there. Under the lamp, light is near certain. Away from the lamp, dark is near certain. 

The natural way to read *A* is one cell at a time. Fix a cell. Look down the column belonging to it. What you find is a set of probabilities saying how likely each observation is from that cell. 

That is a **conditional probability**, and the notation for it is p(o | s): the probability of observation o, given state s. The bar means given, and everything to the right of it is being held fixed. Reading p(o | s) is exactly what you just did by hand: stand in a column and see how the observations divide up.
The table and the notation are the same object. Writing $A_{ij}$ for the entry in row $i$, column $j$, where the column index is the state and the row index is the observation:

$$A_{ij} = p(o = i \mid s = j)$$

so the column index is the state and the row index is the observation. The sum you read off a column is then

$$\sum_i A_{ij} = 1 \quad \text{for every } j$$

In this world there are only 2 observations, so *A* has 2 rows and 25 columns. Take the rows in the order dark, lit. Then every column is one of two things. A lit cell:

$$A_{:,\,\text{lit}} = \begin{bmatrix} 0.2 \\ 0.8 \end{bmatrix}$$

and a dark cell:

$$A_{:,\,\text{dark}} = \begin{bmatrix} 0.9 \\ 0.1 \end{bmatrix}$$

Both add to one, which is the column rule holding: stand on either cell and you see something. The whole table is those two columns, repeated. 

Read the second entry of each as the sensor's honesty. A lit cell reports light eight times in ten and disfunction twice. A dark cell reports dark nine times in ten and hallucinate once.
The $0.2$ and the $0.1$ are exactly the false readings.

<widget id="a-columns"></widget>


Each column sums to one. No such statement holds for the rows.

*A* has a standard name: it is called the **likelihood matrix**, or equivalently the **observation model**. Both refer to this same table. 

Two things about *A* need saying plainly:

- The first is that *A* belongs to the agent. It is the agent's model of what each cell produces, not the world itself. Here the model is exact: the agent's table matches how the world really behaves, and that is what it means to know the map. Nothing guarantees this. The agent could very well not know which observation to expect in the world, this will be covered in module 5, where the agent need to learn A from its readings.

- The second is that the sensor is unreliable (or maybe the lamps are). A lit cell usually reports light, and occasionally reports dark. That is why the columns are distributions rather than single certainties, and it is what makes a false observation tolerable for the model.

Given the cell, readings are independent: whether the second is wrong does not depend on whether the first was. This **conditional independence** is what lets the agent multiply each new reading into its belief as fresh evidence. It is a standard assumption, but real sensors can break it: a smudged lens makes the same mistake every time.

## The same table, read the other way
Here is a difficulty, *A* is built to answer a question the agent cannot ask. It answers *"if I were on cell three, what would I see?"* But the agent does not know it is on cell three. It knows it saw light, and it wants to know which cells that is evidence for.

So it reads *A* the other way. It fixes the observation it received and looks along that row: for each cell, how probable would this observation have been there? Cells where it was likely score high, and cells where it was unlikely score low. The expression is still $p(o \mid s)$. Only the roles have swapped: before, $s$ was fixed and $o$ varied, and now $o$ is fixed and $s$ varies.

One warning: the row is not a distribution and need not sum to one. It is a list of separate scores, one per cell, each saying how well that cell explains the reading.

<widget id="a-rows"></widget>

## Combining
The agent now has two things: what it believed before the observation, and a score per cell from the observation.

Combine them by multiplying cell by cell. A cell ends up with a large number only if the agent already considered it plausible and it explains what was just seen. A cell that was plausible but explains the observation badly is pulled down. A cell that explains the observation well but was nearly ruled out stays low, though less low than before.

Written out, for each state s:

$$p(s \mid o) \propto p(o \mid s) \, p(s)$$

The symbol $\propto$ means proportional to. The left side is what the agent wants, the right side is what it can compute.

Put numbers on it. Before the first reading the agent has no reason to prefer any cell, so the prior is $1/25 = 0.04$ everywhere. Now light arrives. Take the lit row of *A*, which is $0.8$ on a lit cell and $0.1$ on a dark one, and multiply cell by cell:

$$\text{lit cell:}\quad 0.8 \times 0.04 = 0.032$$

$$\text{dark cell:}\quad 0.1 \times 0.04 = 0.004$$

That is the whole of the right-hand side, done twice, because there are only two kinds of column. The lit cells come out eight times larger than the dark ones, which is the ratio $0.8 : 0.1$ surviving the multiplication unchanged: the prior was flat, so it favoured nobody, and the observation did all the work.

Each 3 lit cell has a probability of $0.032$ to explain the lit observation and each 22 dark cells has a $0.004$ probability to explain the same observation, Add them all up and you get $0.184$, not one. They are proportional to the answer, which is what $\propto$ has been saying. As they stand, they are not a belief.

The fix is to divide every number by their total. This is called **normalisation**. Only the ratios between the numbers carry meaning, and dividing keeps every ratio while making the total one again. That restores the rule that probabilities sum to one: a cell gains only at the expense of the others.

Dividing by the total turns the proportionality into an equality:

$$p(s \mid o) = \frac{p(o \mid s) \, p(s)}{\sum_{\tilde{s}} p(o \mid \tilde{s}) \, p(\tilde{s})}$$

The $\tilde{s}$ in the denominator runs over every cell, while $s$ in the numerator stays fixed on the one being updated.
The denominator is that same total, written out:

$$\sum_{\tilde{s}} p(o \mid \tilde{s})\,p(\tilde{s}) = 3(0.032) + 22(0.004) = 0.096 + 0.088 = 0.184$$

Divide each number through by it:

$$\text{lit cell:}\quad \frac{0.032}{0.184} = 0.174
\qquad
\text{dark cell:}\quad \frac{0.004}{0.184} = 0.0217$$

and check the total: $3(0.174) + 22(0.0217) = 1.000$. That is a belief again.


<widget id="belief-bars"></widget>



## The names

The belief before an observation is called the **prior**. The scores from *A* for the observed outcome are the **likelihood**. The belief after multiplying and normalising is the **posterior**. The rule that posterior is proportional to prior times likelihood is **Bayes' rule**, which you have just implemented.

The posterior of one step becomes the prior of the next. 

## What the shapes tell you
Run the update repeatedly and four things happen, all of them readable directly off the belief.

**Repeated evidence sharpens, up to a point.** Consistent readings keep narrowing the belief towards the cells that explain them best, but cells that look alike to the sensor stay tied. The agent can become certain it is somewhere lit, but not which lit cell it is on.

**An uninformative observation does nothing.** If every cell still in play would give the same reading, that observation cannot favour any of them, and the belief stays as it was.

**An informative observation collapses the belief.** Light in a place where only one candidate is lit gives that cell a much larger probability than the rest, and after normalisation the others are left with a really low probability.

**A false reading tilts the belief without destroying it.** A wrong observation pushes the belief the wrong way, but the cells supported by every earlier observation are still there, and the next correct readings bring them back. One observation is outvoted by the rest.

That last point is the reason for all of this. A single guess is destroyed by one contradiction. A belief is only tilted, and ordinary updating repairs it, with no special handling for mistakes.

Here is an example of sequences of observations presented in the figure below. The agent stands on a lit cell for all seven readings and the sensor reports lit, lit, dark, lit, dark, lit, lit.

| reading | 1 lit | 2 lit | 3 dark | 4 lit | 5 dark | 6 lit | 7 lit |
|---|---|---|---|---|---|---|---|
| each lit cell | 0.174 | 0.299 | 0.220 | 0.313 | 0.258 | 0.322 | 0.332 |
| each dark cell | 0.0217 | 0.0047 | 0.0155 | 0.0028 | 0.0102 | 0.0016 | 0.0002 |


<widget id="sequence-stepper"></widget>


One restriction has been in force throughout. The agent has not moved. Every observation came from the same cell, and the belief was revised on the assumption that the cell did not change between readings. That assumption is what makes the sequence of multiplications correct, and it is also what puts the ceiling on how much the agent can learn: it can resolve the noise in its sensor, but it cannot resolve an ambiguity that is built into where it happens to be standing. Moving is what breaks that, and this is the next module.


## To go further

- Namjoshi, [*Fundamentals of Active Inference*](https://mitpress.mit.edu/9780262050951/fundamentals-of-active-inference/), Chapter 9, especially §§9.1–9.3, develops the categorical generative model and hidden-state inference used here.

- Parr, Pezzulo and Friston, [*Active Inference*](https://direct.mit.edu/books/oa-monograph/5299/Active-InferenceThe-Free-Energy-Principle-in-Mind), Chapters 4 and 7, place likelihoods and belief updating inside the full generative model and its discrete-time formulation.
