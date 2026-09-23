# Active inference, by running it

<p class="sub">Walking in the dark</p>

The power goes out in a flat you have lived in for years. You get up and head for the door.

You reach out, expecting the bookshelf. Your hand finds the table. You must be further along than you thought. For a moment, you stop. If this is the table, the door must be further to your left. You adjust your course, turn, feel for the wall to check, and carry on.

Getting to the door takes more than remembering the room. You have to revise your sense of where you are, and sometimes reach out to decide where to step next.

You do not need to be certain where you are to take the next step. What you believe guides what you do; what you discover changes what you believe. Sometimes the next move takes you closer to the door. Sometimes it helps you work out where you are.

Active inference gives us a way to describe this: acting on an uncertain picture of the world, using what happens to update it, and choosing actions both to make progress and to find out more. Believing, checking, and acting.


In robotics, uncertainty is not an exception. A robot may receive the same reading in several places, see only part of a scene, or be unsure whether its own model is correct. It must decide not only how to approach its goal, but when to look again, change viewpoint, revisit a landmark, or test what it thinks it knows.
A Q-table does not represent that uncertainty unless we add machinery around it. Frontier exploration, as must models, handles one special case by sending the robot towards given objevtive only or toward unmapped boundaries or toward. More general reinforcement-learning and POMDP methods can reason about information, but often require an explicit belief model or an additional exploration objective.

Active inference puts the belief itself at the centre. It evaluates actions by both the outcomes they are expected to produce and the uncertainty they are expected to resolve. The same calculation can make the robot move towards its goal, seek a better viewpoint, localise itself, or investigate a part of its model that it has not yet learned.
That does not make active inference automatically better, simpler, or cheaper. Its promise is a single probabilistic language for perception, learning and action under uncertainty—and a precise question to ask of every movement: **what might this achieve, and what might it teach me?**

This tutorial turns that loop into intuition you can see.
