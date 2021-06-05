from typing import List, Tuple, Dict


class ScoreContributionSpecification:
    """
    Specifies how to calculate the score of a goal-directed benchmark.

    The global score will be a weighted average of top-x scores.
    This class specifies which top-x to consider and what the corresponding weights are.
    """

    def __init__(self, contributions: List[Tuple[int, float]]) -> None:
        """
        Args:
            contributions: List of tuples (top_count, weight) for the score contributions
        """
        self.contributions = contributions

    @property
    def top_counts(self) -> List[int]:
        return [x[0] for x in self.contributions]

    @property
    def weights(self) -> List[float]:
        return [x[1] for x in self.contributions]


def uniform_specification(*top_counts: int) -> ScoreContributionSpecification:
    """
    Creates an instance of ScoreContributionSpecification where all the top-x contributions have equal weight

    Args:
        top_counts: list of values, where each value x will correspond to the top-x contribution
    """
    contributions = [(x, 1.0) for x in top_counts]
    return ScoreContributionSpecification(contributions=contributions)


def compute_global_score(contribution_specification: ScoreContributionSpecification,
                         scores: List[float]) -> Tuple[float, Dict[str, float]]:
    """
    Computes the global score according to the contribution specification.

    Args:
        contribution_specification: Score contribution specification
        scores: List of all scores - list must be long enough for all top_counts in contribution_specification

    Returns:
        Tuple with the global score and a dict with the considered top-x scores
    """
    sorted_scores = sorted(scores, reverse=True)

    global_score = 0.0
    top_x_dict = {}

    for top_count, weight in contribution_specification.contributions:
        score = sum(sorted_scores[:top_count]) / top_count
        top_x_dict[f'top_{top_count}'] = score
        global_score += score * weight

    global_score /= sum(contribution_specification.weights)

    return global_score, top_x_dict
