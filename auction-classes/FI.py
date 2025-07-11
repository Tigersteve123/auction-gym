import numpy as np
from scipy.optimize import minimize

class FI:
    def __init__(self, Gamma_func, q_func, p_func, mu_func):
        """
        Gamma_func: A function of (s, A) that returns a cost
        q_func: A function of (s, A) that returns a penalty
        p_func: A function of s that returns a probability
        mu_func: Expected return rate, function of s
        """
        # Exogenous variables
        self.Gamma = Gamma_func  # Gamma(s, A)
        self.q = q_func          # q(s, A)
        self.p = p_func          # p(s)
        self.mu = mu_func        # mu(s)
        
        # Endogenous variables (decision variables)
        self.optimal_A = None
        self.optimal_m = None
        self.optimal_r = None
        
        # Auction outcome
        self.s = 0
        self.r = 0
        
        # Final outcomes
        self.A = 0
        self.profit = 0
        self.entry_decision = True

    def calculate_expected_profit_risk_neutral(self, A, s, r):
        """
        Compute expected profit given A, s, r
        """
        gross_returns = self.mu(s) * s
        funding_cost = self.Gamma(s, A)
        bid_cost = s * r
        audit_penalty = self.q(s, A)

        base_profit = gross_returns - funding_cost - bid_cost
        expected_profit = (1 - self.p(s)) * base_profit + self.p(s) * (base_profit - audit_penalty)

        return expected_profit

    def calculate_optimal_action(self, s, r):
        """
        Optimize expected profit over A \in [0, 1]
        """
        def objective(A):
            return -self.calculate_expected_profit_risk_neutral(A, s, r)

        res = minimize(objective, x0=[0.5], bounds=[(0, 1)], method='L-BFGS-B')

        if res.success:
            self.optimal_A = res.x[0]
            return res.x[0]
        else:
            raise RuntimeError("Optimization of A failed")

    def calculate_optimal_bid(self):
        """
        Jointly optimize over A \in [0,1], r \in [0,1], m <= 0
        """
        def objective(x):
            A, r, m = x
            if not (0 <= A <= 1 and r >= 0 and m >= 0):
                return np.inf
            s = m
            return -self.calculate_expected_profit_risk_neutral(A, s, r)

        x0 = [0.5, 0.05, 10]  # initial guess
        bounds = [(0, 1), (0.01, 1.0), (1e-3, None)]

        res = minimize(objective, x0=x0, bounds=bounds, method='L-BFGS-B')

        if res.success:
            self.optimal_A = res.x[0]
            self.optimal_r = res.x[1]
            self.optimal_m = res.x[2]
        else:
            raise RuntimeError("Joint optimization of (A, r, m) failed")

    def calculate_optimal_entry(self):
        if self.optimal_A is None or self.optimal_m is None or self.optimal_r is None:
            self.calculate_optimal_bid()
        expected_profit = self.calculate_expected_profit_risk_neutral(
            self.optimal_A, self.optimal_m, self.optimal_r
        )
        self.entry_decision = expected_profit >= 0
        return self.entry_decision

    def set_auction_winnings(self, s, r):
        self.s = s
        self.r = r

    def calculate_profit(self):
        real_A = self.calculate_optimal_action(self.s, self.r)
        self.A = real_A
        self.profit = self.calculate_expected_profit_risk_neutral(self.A, self.s, self.r)

    def __str__(self):
        return (
            f"Financial Intermediary State:\n"
            f"  Optimal Bid:\n"
            f"    - Quantity (m): {self.optimal_m:.4f}\n"
            f"    - Rate (r): {self.optimal_r:.4f}\n"
            f"    - Action (A): {self.optimal_A:.4f}\n"
            f"  Auction Outcome:\n"
            f"    - Entry: {str(self.entry_decision)}\n"
            f"    - Won Amount (s): {self.s:.4f}\n"
            f"    - Rate Accepted (r): {self.r:.4f}\n"
            f"  Realized:\n"
            f"    - Final Action (A): {self.A:.4f}\n"
            f"    - Profit: {self.profit:.4f}\n"
        )

# Test case
if __name__ == '__main__':
    def Gamma_func(s, A):
        return s*(np.exp(A)-1)
        
    def q_func(s, A):
        return 10*(1-A)*s
    
    def p_func(s):
        return .5
    
    def mu_func(s):
        return .05
    
    example_FI = FI(Gamma_func, q_func, p_func, mu_func)
    example_FI.calculate_optimal_entry()
    example_FI.set_auction_winnings(example_FI.optimal_m, example_FI.optimal_r)
    example_FI.calculate_profit()
    print(example_FI)
