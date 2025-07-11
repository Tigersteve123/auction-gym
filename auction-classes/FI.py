import numpy as np
from scipy.optimize import minimize
from typing import List, Callable

class FI:
    def __init__(self, Gamma_func, q_func, p_func, mu_func, f=0):
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
        self.f = f
        
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

        base_profit = gross_returns - funding_cost - bid_cost - self.f
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
            return (self.optimal_m, self.optimal_r)
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
        return self.profit

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

class Auction:
    def __init__(self, N: int, q_func: Callable, Gammas: List[Callable], mus: List[Callable], ps: List[Callable], total_funds: int, rate_floor: float):
        """
        N: Number of financial intermediaries
        q_func: Shared penalty function q(s, A)
        Gammas: List of Gamma functions, one per FI (each Gamma(s, A))
        mus: List of expected return functions mu(s), one per FI
        ps: List of audit probability functions p(s), one per FI
        """
        assert len(Gammas) == len(mus) == len(ps) == N, "Length of input lists must match N"

        self.Gammas = Gammas
        self.mus = mus
        self.ps = ps
        self.q = q_func
        self.N = N
        self.total_funds = total_funds
        self.rate_floor = rate_floor

        # Create FI instances
        self.FIs = [
            FI(Gamma_func=Gammas[i], q_func=q_func, p_func=ps[i], mu_func=mus[i])
            for i in range(N)
        ]
        
        self.auction_participants = []
        self.winners = []
    
    def run_auction(self):
        self.auction_participants.clear()
        self.winners = []

        # Entry decisions
        for FI in self.FIs:
            if FI.calculate_optimal_entry():
                self.auction_participants.append(FI)

        # Sort participants by descending optimal_r, then descending optimal_m
        self.auction_participants.sort(
            key=lambda fi: (fi.optimal_r, fi.optimal_m),
            reverse=True
        )

        # Fund allocation with rate floor
        remaining_funds = self.total_funds
        for fi in self.auction_participants:
            if fi.optimal_r < self.rate_floor:
                continue  # Bid below rate floor: disqualified

            if remaining_funds <= 0:
                break  # No more funds to allocate

            allocated = min(fi.optimal_m, remaining_funds)
            fi.set_auction_winnings(s=allocated, r=fi.optimal_r)
            fi.calculate_profit()

            self.winners.append(fi)
            remaining_funds -= allocated

        # Print auction results
        self.print_results()

    def print_results(self):
        print("\nAuction Results:")
        if not self.winners:
            print("  No winning bids met the rate floor.")
        else:
            for idx, fi in enumerate(self.winners):
                print(f" Winner {idx+1}:")
                print(f"   - Allocated: {fi.s:.2f}")
                print(f"   - Rate: {fi.r:.4f}")
                print(f"   - Action A: {fi.A:.4f}")
                print(f"   - Profit: {fi.profit:.4f}")

        remaining_funds = self.total_funds - sum(winner.s for winner in self.winners)
        print(f"Remaining funds after auction: {remaining_funds:.2f}")

    def __str__(self):
        output = ["\nAuction Summary:"]
        output.append(f"Total Funds: {self.total_funds}")
        output.append(f"Rate Floor: {self.rate_floor:.4f}")
        output.append(f"Total Participants: {len(self.auction_participants)}")
        output.append(f"Winning Bids: {len(self.winners)}")

        if self.winners:
            output.append("\nWinners:")
            for idx, fi in enumerate(self.winners):
                output.append(
                    f"  Winner {idx+1}: Allocated={fi.s:.2f}, Rate={fi.r:.4f}, Profit={fi.profit:.4f}"
                )
        
        losers = [
            fi for fi in self.auction_participants
            if fi not in self.winners or fi.optimal_r < self.rate_floor
        ]
        if losers:
            output.append("\nLosers:")
            for idx, fi in enumerate(losers):
                status = "Disqualified (rate too low)" if fi.optimal_r < self.rate_floor else "Not allocated"
                output.append(
                    f"  Loser {idx+1}: Requested={fi.optimal_m:.2f}, Rate={fi.optimal_r:.4f}, Reason: {status}"
                )
        return "\n".join(output)

# Test case
if __name__ == '__main__':
    # Define symmetric function components
    def Gamma_func(s, A):
        return s * (np.exp(A) - 1)

    def q_func(s, A):
        return 10 * (1 - A) * s

    def p_func(s):
        return 0.5

    def mu_func(s):
        return 0.05

    # Parameters
    N = 5  # Number of FIs
    total_funds = 100
    rate_floor = 0.03

    # Create symmetric lists of functions
    Gammas = [Gamma_func] * N
    mus = [mu_func] * N
    ps = [p_func] * N

    # Create and run auction
    auction = Auction(
        N=N,
        q_func=q_func,
        Gammas=Gammas,
        mus=mus,
        ps=ps,
        total_funds=total_funds,
        rate_floor=rate_floor
    )

    auction.run_auction()
    print(auction)  # Optional summary printout

