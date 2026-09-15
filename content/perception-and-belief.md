# Where am I?

A robot vacuum sits in the corner of a room. You turn it on, and now, every
second it checks where it is, picks a direction, then moves. It will keep doing
this loop: localising itself, choosing where to go, then moving in that
direction, again and again, until the battery runs out.

That loop is the whole problem, and the interesting part of it is the middle
step: how should the agent pick a direction? This tutorial builds an answer to
that over several lessons.

But before an agent can choose where to go, it has to work out where it
already is. That turns out to be the harder half, and it is what this lesson
is about.

The pieces:

- **The world**: a 5x5 grid for the room
- **The agent**: somewhere on it
- **The loop**: look, decide, move

## The world

The grid knows where the agent is, where the goal is, and which tiles are lit.
It holds no beliefs and no model of itself. Moving into a wall leaves the agent
where it was.

Every tile has a number, running along each row from the top left, so tile
`(row, col)` is number `row * 5 + col`. Those numbers matter: they are how the
picture of the room becomes a list of probabilities.

## The loop

Every agent in this tutorial runs the same three steps. Only the middle one
ever changes.

```python
observation = world.observe()   # look
action = plan(observation)      # decide
world.step(action)              # move
```

Notice what the first line does not do. It hands back what the sensor said, and
nothing more. On a fully lit grid with a perfect sensor that is harmless,
because what the sensor says is where the agent is.

That equivalence is about to break.

## A sensor that lies sometimes

Real sensors are wrong sometimes. Set the sensor noise and watch what the agent
gets told as it walks a fixed route: right, right, down, down, right, down.

<widget id="sensor-noise-walk"></widget>

An agent that simply stores what it was told believes it is somewhere it is
not, and has no way to notice.

## Keeping a distribution instead

The fix is to stop storing one tile. Instead the agent keeps a number for every
tile: how likely is it that I am here? Twenty-five numbers that add up to one.

That is a **belief**. Not a position, a distribution over positions.

Knowing nothing at all looks like this: every tile equally likely, so each
holds probability 1/25 = 0.04. The two pictures below are the same twenty-five
numbers. On the left they sit on the grid, so you can see where the agent
thinks it is. On the right they are one bar per tile, so you can see how
strongly. Bar 7 is the tile labelled 7.

<figure src="knowing-nothing"></figure>

## What the agent expects to see

To use an observation, the agent needs to know what each tile would look like.
That is one array, called `A` by convention in every paper on the subject.

`A[observation, tile]` is the probability of getting that observation while
standing on that tile. With a perfect sensor it is the identity: on tile 7 you
see 7. With a noisy sensor, some of that certainty is spread over the
neighbouring tiles. Read a column to ask "standing here, what might I see?"

<widget id="a-matrix"></widget>

## The update, in three multiplications

Now the belief can absorb an observation. The agent starts with what it thought
before (the **prior**), multiplies in how likely the observation was from each
tile (the **likelihood**, a row of `A`), and divides by the total so the
numbers add to one again.

```python
unnormalised = likelihood * prior
posterior = unnormalised / unnormalised.sum()
```

That is it. Multiply, then rescale.

<figure src="after-seeing-tile-7"></figure>

The belief has collapsed onto tile 7 and its neighbours. Where a perfect sensor
would leave all the mass on one tile, a noisy one leaves the agent fairly sure
but not certain, which is exactly right: it should not be certain.

## Watching it move

Now put the two halves together. The agent walks the fixed route. After each
move its belief spreads out, because moving is uncertain, and then sharpens
again when it looks. Step through it.

<widget id="watch-it-move"></widget>

The ring is where the agent actually is, which it never gets to know. The bars
are what it believes. When the sensor lies, the belief leans the wrong way for
a step and then recovers, because one bad reading cannot outvote everything
that came before it.

That recovery is the entire reason for keeping a distribution.

## Surprise

One more quantity, because everything later is built on it. When the agent
looks, how unexpected was what it saw?

$$\text{surprise} = -\log p(\text{observation})$$

Zero if it was certain, and larger the less it saw it coming. The logarithm
turns "one in a thousand" into a bigger number than "one in two" in a way that
adds up sensibly across steps.

<figure src="surprise-curve"></figure>

<details>
<summary>Why this is called free energy</summary>

Surprise, $-\log p(o)$, is the quantity an agent would like to keep small. The
awkwardness is that computing it means summing over every state the observation
could have come from, which on this grid is fine and in general is not.

Variational free energy is a quantity that is always at least as large as the
surprise, and which can be computed without that sum. Minimising it therefore
pushes the surprise down without ever evaluating it:

$$F = \underbrace{-\mathbb{E}_{q}[\log p(o \mid s)]}_{\text{accuracy}}
    + \underbrace{D_{KL}[q(s) \,\|\, p(s)]}_{\text{complexity}}$$

The two terms pull against each other. Accuracy wants a belief that explains
what was seen. Complexity charges the agent for moving away from what it
believed beforehand. A belief that lurches to fit one observation pays for it.

On this grid the exact posterior is computable, so nothing here needed free
energy. It is written down because the same two-term shape reappears when the
agent starts scoring futures rather than explaining the present, and that is
the lesson where the argument actually lands.

</details>

<details>
<summary>Glossary: the four letters</summary>

| Symbol | Means | Where |
|---|---|---|
| `A` | what the agent expects to see in each state | here |
| `B` | how states change under each action | here |
| `C` | which observations it prefers | lesson 2 |
| `D` | where it believes it starts | here |

These four letters are kept because every paper and every implementation uses
them. Everything else in this repository is spelled out in full:
`expected_free_energy`, never `efe`.

</details>

---

The agent now knows roughly where it is. It still has no reason to prefer one
tile over another, and no way to decide where to go.

**Next:** what it means to imagine a sequence of moves before making them.
