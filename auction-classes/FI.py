import numpy as np
from scipy.optimize import minimize, minimize_scalar

class FI:
    def __init__(self, Gamma_func, q_func, p_func, mu_func):
        """
        Gamma_func: A function of (A, s) that returns a cost rate
        q_func: A function of (A, s) that returns a penalty
        p_func: A function of s that returns a probability
        mu: Expected return rate, function of s
        """
        # Exogenous variables
        self.Gamma = Gamma_func  # Function for cost Gamma(s, A)
        self.q = q_func # Penalty function q(s, A)
        self.p = p_func # Audit probability function p(s)
        self.mu = mu_func
        
        # Choice/endogenous variables
        self.optimal_A = None
        self.optimal_m = None
        self.optimal_r = None
        
        # Auction-determined variables
        self.s = None
        self.r = None
        
        self.profit = None
        self.A = None

    def calculate_expected_profit_risk_neutral(self, A, s, r):
        """
        Expected profit with continuous s (dollar amount of funds won).
        """
        gross_returns = self.mu(s) * s
        funding_cost = self.Gamma(A, s)
        bid_cost = s * r
        audit_penalty = self.q(A, s)

        base_profit = gross_returns - funding_cost - bid_cost
        expected_profit = (1 - self.p(s)) * base_profit + self.p(s) * (base_profit - audit_penalty)

        return expected_profit

    def calculate_optimal_action(self, s, r):
        """
        Optimize expected profit over A \in [0, 1] for continuous s.
        """
        def objective(A):
            return -self.calculate_expected_profit_risk_neutral(A, s, r)

        res = minimize_scalar(objective, bounds=(0, 1), method='bounded')

        if res.success:
            self.optimal_A = res.x
            return res.x
        else:
            raise RuntimeError("Optimization of A failed")

    
    def calculate_optimal_bid(self):
        """
        Optimize over bid rate r \in [0,1] and bid size m \geq 0.
        We treat s = m here since the intermediary wins that amount.
        """
        def objective(x):
            r, m = x
            s = m  # if bid succeeds, they win m dollars
            A_star = self.calculate_optimal_action(s, r)
            self.optimal_A = A_star
            expected_profit = self.calculate_expected_profit_risk_neutral(A_star, s, r)
            return -expected_profit

        bounds = [(0.01, 1.0), (1e-3, None)]  # avoid zero size
        x0 = [0.05, 10]  # initial guess

        res = minimize(objective, x0=x0, bounds=bounds)

        if res.success:
            self.optimal_r = res.x[0]
            self.optimal_m = res.x[1]
        else:
            raise RuntimeError("Optimization of (r, m) failed")

    
    def calculate_optimal_entry(self):
        if self.optimal_A is None or self.optimal_m is None or self.optimal_r is None:
            self.calculate_optimal_bid()
        expected_profit = self.calculate_expected_profit_risk_neutral(self.optimal_A, self.optimal_m, self.optimal_r)
        return expected_profit >= 0
        
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
        return A*s
    
    def p_func(s):
        return .5
    
    def mu_func(s):
        return .05
    
    example_FI = FI(Gamma_func, q_func, p_func, mu_func)
    example_FI.calculate_optimal_entry()
    example_FI.set_auction_winnings(example_FI.optimal_m, example_FI.optimal_r)
    example_FI.calculate_profit()
    print(example_FI)
