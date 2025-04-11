from .base import Strategy

class EnsembleStrategy(Strategy):
    def __init__(self, data, strategies, vote_rule="majority"):
        super().__init__(data)
        self.strategies = strategies  # List of Strategy instances
        self.vote_rule = vote_rule

    def generate_signals(self):
        # Generate signals from each sub-strategy
        for strategy in self.strategies:
            strategy.generate_signals()

        # Combine signals
        combined = []
        for i in range(len(self.data)):
            signals_at_i = [strat.signals[i] for strat in self.strategies if i < len(strat.signals)]

            # Apply voting rule
            vote = self.apply_vote(signals_at_i)
            combined.append(vote)

        self.signals = combined

    def apply_vote(self, signals):
        # Remove Nones
        signals = [s for s in signals if s is not None]
        if not signals:
            return None

        if self.vote_rule == "majority":
            counts = {"buy": 0, "sell": 0}
            for s in signals:
                if s in counts:
                    counts[s] += 1
            if counts["buy"] > counts["sell"]:
                return "buy"
            elif counts["sell"] > counts["buy"]:
                return "sell"
            else:
                return None  # tie

        elif self.vote_rule == "consensus":
            return signals[0] if all(s == signals[0] for s in signals) else None

        elif self.vote_rule == "weighted":
            # Add weights if needed (e.g., HMM: 0.6, NLP: 0.4)
            # Not implemented yet
            return None

        return None
