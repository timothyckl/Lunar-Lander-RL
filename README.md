# Reinforcement-Learning-CA2

Algorithms in this project were implemented with help from [Reinforcement Learning: An Introduction by Richard S. Sutton and Andrew G. Barto](#http://incompleteideas.net/book/bookdraft2017nov5.pdf)

## Lunar Lander

### Action Space

The lunar lander has two types of action spaces, discrete and continuous.

**Discrete $A$**

| action |    description   |
|--------|------------------|
| $a_0$  |    Do nothing    |
| $a_1$  | Fire left engine |
| $a_2$  | Fire main engine |
| $a_3$  | Fire right engine|

### State Space

| state  |          description      |
|--------|---------------------------|
| $s_0$  |   x-axis coord of agent   |
| $s_1$  |   y-axis coord of agent   |
| $s_2$  |   x-axis linear velocity  |
| $s_3$  |   y-axis linear velocity  |
| $s_4$  |       Agent's angle       |
| $s_5$  |  Agent's angular velocity |
| $s_6$  |  Right leg touched ground |
| $s_7$  |  Left leg touched ground  |

### Rewards

An episode is considered a solution if it scores at least 200 points.

|   points  |                     condition             |
|-----------|------------------------------------------ |
|     +/-   |        Agent's distance to landing pad    |
|     +/-   |                Agent's speed              |

|      -    |       Agent's tilt (angle not horizontal) |
|     +10   |       For each leg that contacts ground   |
|    -0.03  |   For each frame that a side engine fires |
|     -0.3  |   For each frame that main engine fires   |
|   +/- 100 |   For crashing/landing safely             |

## Episode Termination

1. Lander crashes
2. Lander exits viewport (x-coord is gt 1)
3. Lander is not awake