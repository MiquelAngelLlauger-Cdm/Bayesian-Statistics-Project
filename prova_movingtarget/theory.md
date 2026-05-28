# Constrained Bayesian Search for a Moving Target: Theoretical Formulation

This document provides a rigorous mathematical and conceptual foundation for the **Optimal Constrained Search for a Moving Target**, based on the seminal paper by **James N. Eagle (1984)**: *"The Optimal Search for a Moving Target When the Search Path is Constrained"*.

This theoretical guide serves as a comprehensive reference to bridge the static Bayesian spatial search explored in class (e.g., the Palomares case study in *Deliverable 2*) with dynamic, physically constrained search scenarios typical in maritime rescue, tracking, and defence applications.

---

## 1. Introduction and Course Connection

In *Deliverable 2*, we analyzed the **static spatial search problem** (inspired by the Palomares bomb recovery). The target (bomb) was assumed to be stationary, and our task was to build a spatial prior distribution $\pi_j$, model a detection function given effort, update our beliefs using Bayes' rule after failed missions, and select the next search cell.

While static search models are useful for fixed wreckages, many real-world applications involve a **moving target** (e.g., drifting ships, lost hikers, submarines) and a **physically constrained searcher** (e.g., a boat, aircraft, or rescue team that can only travel a limited distance in one time period). 

This project extends the static search model in two major ways:
1. **Target Dynamics**: The target moves according to a discrete-time Markov chain with a known transition matrix $P$.
2. **Searcher Constraints**: The searcher's physical movement is restricted: the search cell chosen in the current period must be selected from a set of reachable cells determined by the cell searched in the previous period.

---

## 2. Mathematical Problem Formulation

### 2.1 The Environment and Target Dynamics
Let the search area be discretized into a finite set of $N$ cells, denoted by $C = \{1, 2, \ldots, N\}$. 
Time is discrete and indexed by $t \in \{1, 2, \ldots, T\}$, where $T$ is the finite search horizon.

At the beginning of each time step $t$, the target occupies some cell $Z_t \in C$. The target moves between time steps according to a stationary **Markov process** with a known transition matrix $P = [p_{ij}] \in \mathbb{R}^{N \times N}$, where:
$$p_{ij} = P(Z_{t+1} = j \mid Z_t = i)$$
and $\sum_{j=1}^N p_{ij} = 1$ for all $i \in C$.

### 2.2 Searcher Kinematics and Constraints
The searcher is located in a single cell in each period. If the searcher was in cell $i$ during period $t-1$, the cell $j$ searched in period $t$ must belong to a constrained set of accessible cells $C_i \subseteq C$. 

This formulation naturally models:
- **Local Grid Search**: $C_i$ consists of cell $i$ and its immediate horizontal/vertical/diagonal neighbors (limited travel speed).
- **Relocation Overheads**: Grid topologies where some cells are unreachable due to physical barriers (e.g., islands, shallow waters).

### 2.3 Detection Model
Let $q_j \in [0, 1]$ be the probability of detecting the target in cell $j$ during a single time period, given that the target is indeed in cell $j$ and the searcher searches cell $j$. If the target is not in the cell currently searched, the probability of detection is $0$.

---

## 3. Bayesian Updating (The Failed Search Filter)

Let $r(t) = \left( r_1(t), \ldots, r_N(t) \right)$ be a row vector representing the probability distribution of the target's location at the start of period $t$, given that all searches in periods $1, \ldots, t-1$ were unsuccessful.
$$r_k(t) = P(Z_t = k \mid \text{unsuccessful search in periods } 1, \ldots, t-1)$$
where $\sum_{k=1}^N r_k(t) = 1$ and $r_k(t) \ge 0$.

Suppose the searcher decides to search cell $j \in C$ at time step $t$. If the search is unsuccessful (which occurs with probability $1 - q_j r_j(t)$), we update our target distribution using Bayes' rule.

### 3.1 Posterior Before Target Transition
Immediately after the failed search in cell $j$, but *before* the target moves, the intermediate probability $r'_k(t)$ that the target is in cell $k$ is:
- **For $k \neq j$** (cells not searched):
  $$r'_k(t) = P(Z_t = k \mid \text{failed search in } j) = \frac{r_k(t) \cdot 1}{1 - q_j r_j(t)}$$
- **For $k = j$** (the searched cell):
  $$r'_j(t) = P(Z_t = j \mid \text{failed search in } j) = \frac{r_j(t) \cdot (1 - q_j)}{1 - q_j r_j(t)}$$

### 3.2 Prior for the Next Step (After Transition)
The target then transitions according to the Markov matrix $P$. The target location prior at the beginning of period $t+1$ is obtained by multiplying the intermediate vector $r'(t)$ by $P$:
$$r_k(t+1) = \sum_{m=1}^N r'_m(t) p_{mk}$$

In vector notation, we can write this recursive update function $T_j(r)$ as:
$$T_j(r) = (1 - q_j r_j)^{-1} r P_j$$
where $P_j \in \mathbb{R}^{N \times N}$ is the modified transition matrix obtained by multiplying the $j$-th row of $P$ by $(1 - q_j)$.

*Proof of Vector Form:*
Let $D_j = \text{diag}(1, \ldots, 1 - q_j, \ldots, 1)$ be a diagonal matrix where the $j$-th diagonal element is $1-q_j$. The intermediate row vector $r'(t)$ is:
$$r'(t) = \frac{r(t) D_j}{1 - q_j r_j(t)}$$
Applying the target transition matrix $P$:
$$r(t+1) = r'(t) P = \frac{r(t) D_j P}{1 - q_j r_j(t)}$$
Since $D_j$ is a diagonal matrix with $1 - q_j$ at row $j$ and $1$ elsewhere, the product $D_j P$ is exactly the transition matrix $P$ with its $j$-th row scaled by $(1-q_j)$. Letting $P_j = D_j P$ yields:
$$T_j(r) = \frac{r P_j}{1 - q_j r_j}$$
This completed filter represents the Bayesian update for a moving target under a failed search.

---

## 4. Partially Observable Markov Decision Process (POMDP)

Because the target's true cell $Z_t$ is hidden (partially observable) and our knowledge is summarized by the continuous belief state $r(t) \in \Pi = \{r \in \mathbb{R}^N \mid \sum_k r_k = 1, r_k \ge 0\}$, the constrained search problem is a **Partially Observable Markov Decision Process (POMDP)**.

### 4.1 The Dynamic Programming Bellman Equation
We use dynamic programming, labeling "backward in time" to denote the remaining search steps.
Let $V_n(r, i)$ be the **maximum probability of detecting the target in the remaining $n$ time periods**, given that the current target distribution is $r$ and the cell searched in the previous period was $i$.

The Bellman optimality equation is:
$$V_n(r, i) = \max_{j \in C_i} \left\{ q_j r_j + (1 - q_j r_j) V_{n-1}(T_j(r), j) \right\}$$
with the boundary condition:
$$V_0(r, i) = 0 \quad \forall r \in \Pi, \ i \in C$$

- $q_j r_j$ is the probability of immediate detection in cell $j$.
- $1 - q_j r_j$ is the probability of failing to detect.
- $V_{n-1}(T_j(r), j)$ is the optimal probability of detection in the remaining $n-1$ periods, starting from updated target distribution $T_j(r)$ and previous search cell $j$.

---

## 5. Value Function Convexity and the Vector Update Rule

The belief space $\Pi$ is continuous and infinite, which usually makes dynamic programming equations intractable. However, we can prove a crucial structural property: the value function $V_n(r, i)$ is **piecewise linear and convex (PWLC)** in the belief state $r$.

### 5.1 Theorem (Piecewise Linearity & Convexity)
For all $n \in \{0, \ldots, T\}$ and $i \in C$, the value function $V_n(r, i)$ can be written in the form:
$$V_n(r, i) = \max_{a \in A(n, i)} r a$$
where $A(n, i)$ is a finite collection of $N$-dimensional column vectors, and $r a = \sum_{k=1}^N r_k a_k$ is the dot product.

### 5.2 Proof (By Induction)
*Base Case ($n=0$):*
$$V_0(r, i) = 0 = r \mathbf{0}$$
Hence, $A(0, i) = \{\mathbf{0}\}$, where $\mathbf{0}$ is the zero vector. The theorem holds.

*Base Case ($n=1$):*
Using the Bellman equation:
$$V_1(r, i) = \max_{j \in C_i} \left\{ q_j r_j \right\}$$
Let $e_j \in \mathbb{R}^N$ be the $j$-th standard basis column vector (a vector with $1$ at index $j$ and $0$ elsewhere). Then $q_j r_j = r (q_j e_j)$. Therefore:
$$V_1(r, i) = \max_{j \in C_i} r (q_j e_j)$$
Thus, $A(1, i) = \{ q_j e_j \mid j \in C_i \}$. The theorem holds.

*Inductive Step:*
Assume the theorem holds for $n-1$, i.e., $V_{n-1}(r, j) = \max_{a_j \in A(n-1, j)} r a_j$. 
Substitute this induction hypothesis into the Bellman recurrence for step $n$:
$$V_n(r, i) = \max_{j \in C_i} \left\{ q_j r_j + (1 - q_j r_j) \max_{a_j \in A(n-1, j)} T_j(r) a_j \right\}$$
Bring the scalar term $(1 - q_j r_j)$ inside the maximization over $a_j$:
$$V_n(r, i) = \max_{j \in C_i} \max_{a_j \in A(n-1, j)} \left\{ q_j r_j + (1 - q_j r_j) T_j(r) a_j \right\}$$
Substitute the filter definition $T_j(r) = (1 - q_j r_j)^{-1} r P_j$:
$$V_n(r, i) = \max_{j \in C_i} \max_{a_j \in A(n-1, j)} \left\{ q_j r_j + (1 - q_j r_j) \frac{r P_j a_j}{1 - q_j r_j} \right\}$$
The denominator terms cancel perfectly!
$$V_n(r, i) = \max_{j \in C_i} \max_{a_j \in A(n-1, j)} \left\{ q_j r_j + r P_j a_j \right\}$$
Rewrite $q_j r_j$ as $r (q_j e_j)$ and factor out the row vector $r$:
$$V_n(r, i) = \max_{j \in C_i} \max_{a_j \in A(n-1, j)} r \left( q_j e_j + P_j a_j \right)$$
Thus, we can write $V_n(r, i) = \max_{a \in A(n, i)} r a$, where the new vector set $A(n, i)$ is defined by:
$$A(n, i) = \bigcup_{j \in C_i} \left\{ q_j e_j + P_j a_j \;\middle|\; a_j \in A(n-1, j) \right\}$$
This completes the proof. $\blacksquare$

### 5.3 Physical Interpretation of Vector Components
Every vector $a \in A(n, i)$ is associated with a specific chronological sequence of search cells (a search path) of length $n$.
The $k$-th component of a vector $a \in A(n, i)$, denoted $a_k$, represents the **conditional probability of detecting the target during the $n$-period search, given that the target starts in cell $k$** at the beginning of this $n$-step horizon, and the searcher follows the path associated with $a$.

---

## 6. Pruning and Vector Dominance

If the searcher has $M$ accessible choices at each step ($|C_i| \approx M$), then the size of the set $A(n, i)$ grows exponentially as:
$$|A(n, i)| \approx M^n$$
For a 10-period search with 5 actions per cell, this yields $5^{10} \approx 9.7 \times 10^6$ vectors. This exponential explosion makes dynamic programming computationally expensive. 

Fortunately, many vectors are **dominated**—meaning there is another vector or a combination of vectors that yields a higher detection probability for *all* possible prior distributions $r \in \Pi$. We can prune these vectors at each step $n$ to keep $|A(n, i)|$ small.

### 6.1 Simple (Pairwise) Dominance
A vector $a \in A(n, i)$ is strictly pairwise dominated if there exists another vector $b \in A(n, i)$ such that:
$$b_k \ge a_k \quad \forall k \in \{1, \ldots, N\} \quad (\text{with at least one strict inequality } b_m > a_m)$$
If $b \ge a$ element-wise, then for any belief state $r \ge 0$, we have $r b \ge r a$. Thus, $a$ can never be the unique argmax of the value function and can be discarded safely. This check is computationally cheap ($\mathcal{O}(|A|^2)$ vector comparisons) and extremely effective.

### 6.2 Linear Programming (LP) Dominance (Convex Hull Pruning)
Sometimes a vector $a$ is not dominated by any *single* vector, but is dominated by a *convex combination* of other vectors. That is, the hyperplanes associated with other vectors collectively "cover" $a$ over the entire belief simplex $\Pi$.

To find if $a$ is dominated in this complete sense, we solve the following linear program for each vector $a \in A(n, i)$:
$$\min_{\pi, x} \left( x - \pi a \right)$$
$$\text{subject to: } x \ge \pi b \quad \forall b \in A(n, i) \setminus \{a\}$$
$$\sum_{k=1}^N \pi_k = 1, \quad \pi_k \ge 0 \quad \forall k$$

If the minimum objective value of this LP is **non-negative** ($\ge 0$), then there is no belief state $\pi$ where $a$ outperforms all other vectors. Thus, $a$ is dominated and can be pruned. If the minimum is negative, $a$ is optimal for at least one belief state and must be kept.

#### The LP Dual Formulation
Solving the primal LP is standard, but its dual is highly illuminating and often faster. The dual is:
$$\max v$$
$$\text{subject to: } \sum_{b \neq a} \lambda_b b - v \mathbf{1} \ge a$$
$$\sum_{b \neq a} \lambda_b = 1, \quad \lambda_b \ge 0 \quad \forall b$$

*Dual Interpretation:*
A vector $a$ is dominated if and only if there exists a convex combination of the other vectors in $A(n, i)$ (with coefficients $\lambda_b$) that is element-wise greater than or equal to $a$ (i.e., lies strictly "above" $a$ in the vector space).

---

## 7. Reconstruction of the Optimal Search Path

Once the vector sets $A(t, i)$ have been generated and pruned for all $t = 1, \ldots, T$ and starting cells $i \in C$, we can find the optimal search path for any initial target distribution $r(1)$:

1. **Calculate Maximum $P_d$**: Given initial searcher starting cell $i_0$, find:
   $$P_d^* = \max_{a \in A(T, i_0)} r(1) a$$
2. **Retrieve Search Path**: The vector $a^*$ that achieves this maximum is associated with a specific sequence of search cells $s_1^*, s_2^*, \ldots, s_T^*$.
   - $s_1^*$ is the cell $j \in C_{i_0}$ used to construct $a^*$.
   - The remaining sequence $s_2^*, \ldots, s_T^*$ is recursively retrieved from the vector $a_{s_1}^* \in A(T-1, s_1^*)$ that generated $a^*$.

This structure guarantees that we obtain the optimal search path without having to re-run the dynamic program if the initial prior $r(1)$ changes!
