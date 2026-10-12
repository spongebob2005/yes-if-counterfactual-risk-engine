"""LinUCB contextual bandit (M5). Numpy only, so the API can load the learned state.

`gamma` < 1 discounts old observations (D-LinUCB), so the policy can follow a world that
changes, which is the point of the drift experiment.
"""

from __future__ import annotations

import numpy as np


class LinUCB:
    def __init__(self, n_actions: int, d: int, alpha: float = 1.0, gamma: float = 1.0):
        self.A = np.stack([np.eye(d) for _ in range(n_actions)])
        self.b = np.zeros((n_actions, d))
        self.alpha = alpha
        self.gamma = gamma

    def scores(self, x: np.ndarray, alpha: float | None = None) -> np.ndarray:
        alpha = self.alpha if alpha is None else alpha
        out = np.empty(len(self.A))
        for a, (A, b) in enumerate(zip(self.A, self.b)):
            theta = np.linalg.solve(A, b)
            width = np.sqrt(x @ np.linalg.solve(A, x))
            out[a] = theta @ x + alpha * width
        return out

    def choose(self, x: np.ndarray, allowed: np.ndarray | None = None, alpha: float | None = None) -> int:
        s = self.scores(x, alpha)
        if allowed is not None:
            s = np.where(allowed, s, -np.inf)
        return int(np.argmax(s))

    def update(self, a: int, x: np.ndarray, reward: float) -> None:
        if self.gamma < 1.0:  # forget a little, but keep the identity prior
            eye = np.eye(len(x))
            self.A[a] = self.gamma * self.A[a] + (1 - self.gamma) * eye
            self.b[a] *= self.gamma
        self.A[a] += np.outer(x, x)
        self.b[a] += reward * x

    def state(self) -> dict:
        return {"A": self.A.tolist(), "b": self.b.tolist(), "alpha": self.alpha, "gamma": self.gamma}

    @classmethod
    def from_state(cls, state: dict) -> "LinUCB":
        A = np.asarray(state["A"], dtype=float)
        bandit = cls(A.shape[0], A.shape[1], state["alpha"], state.get("gamma", 1.0))
        bandit.A, bandit.b = A, np.asarray(state["b"], dtype=float)
        return bandit


def context(risk: float, amount: float) -> np.ndarray:
    """Expected cost of every action is linear in (1, amount, risk x amount), so that is the context.

    The plan's sketch (log amount, hour, sharing) underperformed every baseline here:
    fraud loss scales with the amount itself, not its log.
    """
    return np.array([1.0, amount / 100.0, risk * amount / 100.0, risk])


class ThompsonFriction:
    """Learns how well each friction works, from outcomes, and keeps learning as the world changes.

    For every action it keeps discounted Beta counts for two rates: the share of fraud it
    catches and the share of legitimate customers who abandon. Both are observable in
    production (a failed verification or a later chargeback; a customer who walks away).
    It starts from the configured beliefs (worth `prior_weight` observations), then picks
    the action with the lowest expected cost under sampled rates, given the calibrated risk.
    Friction costs and the transaction value are known, not learned.
    """

    def __init__(self, sim: dict, margin: float, actions: tuple[str, ...], prior_weight: float = 20.0,
                 gamma: float = 0.999):
        self.actions, self.margin, self.gamma = actions, margin, gamma
        self.fric = np.array([sim[a]["fric"] for a in actions], dtype=float)
        catch = np.array([sim[a]["catch"] for a in actions], dtype=float)
        abandon = np.array([sim[a]["abandon"] for a in actions], dtype=float)
        self.catch = np.column_stack([catch * prior_weight + 1e-3, (1 - catch) * prior_weight + 1e-3])
        self.abandon = np.column_stack([abandon * prior_weight + 1e-3, (1 - abandon) * prior_weight + 1e-3])

    def expected(self, risk: float, amount: float, catch: np.ndarray, abandon: np.ndarray) -> np.ndarray:
        return self.fric + risk * (1 - catch) * amount + (1 - risk) * abandon * self.margin * amount

    def choose(self, risk: float, amount: float, allowed: np.ndarray, rng: np.random.Generator | None = None) -> int:
        if rng is None:  # serving: posterior means, deterministic
            catch = self.catch[:, 0] / self.catch.sum(axis=1)
            abandon = self.abandon[:, 0] / self.abandon.sum(axis=1)
        else:
            catch = rng.beta(self.catch[:, 0], self.catch[:, 1])
            abandon = rng.beta(self.abandon[:, 0], self.abandon[:, 1])
        cost = np.where(allowed, self.expected(risk, amount, catch, abandon), np.inf)
        return int(np.argmin(cost))

    def update(self, a: int, is_fraud: bool, slipped: bool, abandoned: bool) -> None:
        if is_fraud:
            self.catch[a] = self.gamma * self.catch[a] + np.array([not slipped, slipped], dtype=float)
        else:
            self.abandon[a] = self.gamma * self.abandon[a] + np.array([abandoned, not abandoned], dtype=float)

    def rates(self) -> dict[str, dict[str, float]]:
        return {a: {"catch": float(self.catch[i, 0] / self.catch[i].sum()),
                    "abandon": float(self.abandon[i, 0] / self.abandon[i].sum())}
                for i, a in enumerate(self.actions)}

    def state(self) -> dict:
        return {"kind": "thompson", "catch": self.catch.tolist(), "abandon": self.abandon.tolist(),
                "fric": self.fric.tolist(), "margin": self.margin, "gamma": self.gamma, "actions": list(self.actions)}

    @classmethod
    def from_state(cls, state: dict) -> "ThompsonFriction":
        policy = cls.__new__(cls)
        policy.actions, policy.margin, policy.gamma = tuple(state["actions"]), state["margin"], state["gamma"]
        policy.fric = np.asarray(state["fric"], dtype=float)
        policy.catch = np.asarray(state["catch"], dtype=float)
        policy.abandon = np.asarray(state["abandon"], dtype=float)
        return policy
