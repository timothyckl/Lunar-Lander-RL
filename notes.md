# Reinforcement Learning

Reinforcement learning algorithms study the behavior of subjects in such environments and learn to optimize that behavior (e.g. maximizing some reward).

## Table of Contents

1. Markov Decision Processes
2. Value-Learning
3. Policy-Learning

## 1. Markov Decision Processes

MDPs are a way of formalizing sequential decision making and acts as the basis for structuring problems solved with RL.

### 1.1 Components of an MDP

In an MDP, the decision maker, called an agent, interacts with the environment it's placed in. These interactions occur sequentially over time. At each time step, the agent will get some representation of the environment's state. Given this representation, the agent selects an action to take. The environment is then transitioned into a new state, and the agent is given a reward as a consequence of the previous action.

- Agent (Goal is to maximize cumulative future rewards)
- Environment
- State
- Action 
- Reward

### 1.2 MDP Notations

- $S_t\in S$, where $S_t$ is the state at time $t$ in the state space
- $A_t\in A$, where $A_t$ is the action taken at time $t$ in the action space
- $R_{t+1}\in R$, where $R_{t+1}$ is the reward from taking action $A_t$ in state $S_t$

We can think of the process of receiving a reward as an arbitrary function $f$ that maps state-action pairs to rewards. At each time , we have

$$
f(S_t, A_t)=R_{t+1}
$$

The trajectory representing the sequential process of selecting an action from a state, transitioning to a new state, and receiving a reward can be represented as

$$
S_0,A_0,R_1,S_1,A_1,R_2,S_2,A_2,...
$$

### 1.3 Expected Return

$G_t$, is the sum of future rewards aka expected return at time $t$ and is defined as

$$
G_t=R_{t+1}+R_{t+2}+R_{t+3}+...+R_T
$$

where $T$ is the final time step.

### 1.4 Episodic and Continuing Tasks

The notion of a final time step $T$ is defined when a task lasts a finite amount of time, called epsiodes. Formally, tasks with episodes are called episodic tasks.

A continuous task never ends, making the definition of the return at each time $t$ problematic as $T=\infty$. Which is why the idea of discounted returns needs to be introduced.

### 1.5 Discounted Return

Rather than the agent's goal being to maximize the expected return of rewards, it will instead be to maximize the expected *discounted* return of rewards by taking action $A_t$ at each time $t$.

The discount rate $\gamma$ is set between 0 and 1. This will discount future rewards and determine the present value of future rewards at time $t$. Discounted return is defined as 

$$
G_t=R_{t+1}+\gamma R_{t+2}+\gamma^2 R_{t+3}+...
$$

$$
=\sum_{k=0}^{\infty} \gamma^k R_{t+k+1}
$$

The effect of adjusting $\gamma$ will make it such that the agent will care more about immediate rewards than future rewards, since future rewards will be more heavily discounted. $\gamma=0$ will only consider immediate rewards and $\gamma=1$ will cause immediate and all future rewards to be equally weighted.

We can see from below that returns at successive time steps are related to each other

$$
G_t=R_{t+1}+\gamma R_{t+2}+\gamma^2 R_{t+3}+\gamma^3 R_{t+4}+...
$$

$$
=R_{t+1}+\gamma (R_{t+2}+\gamma R_{t+3}+\gamma^2 R_{t+4}+...)
$$

$$
=R_{t+1}+\gamma G_{t+1}
$$

Even though the return at time $t$ is a sum of an infinite number of terms, the return is actually finite as long as the reward is nonzero and constant, and $\gamma<1$.

$$
G_t=\sum_{k=0}^{\infty} \gamma^k=\frac{1}{1-\gamma}
$$

### 1.6 Policies 

Denoted by $\pi$, a policy is a function that outputs a probablity distribution of selecting each action given a state. In other words, what action an agent should take given a certain state.

If an agent follows a policy $\pi$, at time $t$, then $\pi (a|s)$. This means that at time $t$, under policy $\pi$, $\pi (a|s)$ is the probability of taking action $a$ given state $s$.

Note that for each state $s \in S$, $\pi$ is a probability distribution over $a \in A(s)$.

![](https://i.imgur.com/CifMjfR.png)


### 1.7 Value Functions

Value functions give us a way to measure "how good" it is for an agent to be in a given state OR to select a given action within a particular state. 

Value functions are defined with respect to specific ways of acting. Since the way an agent acts is influenced by the policy it's following, hence we can see that value functions are defined with respect to policies.

![](https://i.imgur.com/9cRJ6M7.png)

#### 1.7.1 V-function (State-Value)

The V-function gives the expected total return from starting from state $s$ at time $t$ and following policy $\pi$ thereafter and is defined as

$$
V_\pi (s)=E_\pi[G_t|S_t=s]
$$

$$
=E_\pi[\sum_{k=0}^\infty \gamma^k R_{t+k+1}|S_t=s]
$$

#### 1.7.2 Q-function (Action-Value)

The Q-function captures the expected total future reward an agent in state $s$ at time $t$ can receive by executing a certain action $a$ under policy $\pi$ and is defined as

$$
Q_\pi (s,a)=E_\pi[G_t|S_t=s,A_t=a]
$$

$$
=E_\pi[\sum_{k=0}^\infty \gamma^k R_{t+k+1}|S_t=s,A_t=a]
$$

### 1.8 Optimal Policies

In terms of return, a policy $\pi$ is considered to be better than or the same as policy $\pi'$ if the expected return of $\pi$ is greater than or equal to the expected return of $\pi'$ for all states. In other words,

$$
\pi \geq \pi' \\; iff \\; v_\pi(s) \geq v_{\pi'}(s) \\; for \\; all \\; s \in S
$$

A policy that is better than or at least the same as all other policies is called the *optimal policy*.

#### 1.8.1 Optimal V-function

$V_\*$ denotes the optimal state-value function and is defined as

$$
V_\*(s)=\underset{\pi}{\operatorname{max}} V_\pi(s)
$$

for all $s \in S$. In other words, $V_\*$ gives the largest expected return achievable by any policy $\pi$ for each state.

#### 1.8.2 Optimal Q-function
Similarly, $Q_\*$ denotes the optimal action-value function is defined as

$$
Q_\*(s,a)=\underset{\pi}{\operatorname{max}} Q_\pi(s,a)
$$

for all $s \in S$ and $a \in A(s)$. In other words, $Q_\*$ gives the largest expected return achievable by any policy $\pi$ for each possible state-action pair.

### 1.9 Bellman Equation

The Bellman Equation states that, for any state-action pair $(s,a)$ at time $t$, the Q-value is going to be the expected reward we get from taking action $a$ in state $s$, which is $R_{t+1}$, plus the maximum expected discounted return that can be achieved from any possible next state-action pair $(s',a')$.

$$
Q_\*(s,a)=E[R_{t+1}+\gamma \underset{a'}{\operatorname{max}} Q_\*(s',a')]
$$

We can use this equation to find $Q_\*$, and in turn determine the optimal policy for any state $s$. All we have to do next is find an action $a$ that maximizes $Q_*(s,a)$.

[Additional Resource](https://www.youtube.com/watch?v=14BfO5lMiuk)

## 2. Value-Learning (Q-Learning)

## 3. Policy Learning






<!-- 
## Value-Learning

Find $Q(s,a)$

$$
a=\underset{a}{\operatorname{argmax}} Q(s,a)
$$


## Policy Learning

Find $\pi(s)$
Sample $a\sim\pi(s)$ -->
