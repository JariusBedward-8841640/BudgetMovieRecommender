"""Environment package for BudgetMovieRecommender."""

__all__ = ["BudgetMovieEnv"]


def __getattr__(name: str):
    """Lazy import to avoid importing Gymnasium on package import."""
    if name == "BudgetMovieEnv":
        from .movie_env import BudgetMovieEnv

        return BudgetMovieEnv
    raise AttributeError(f"module 'env' has no attribute '{name}'")
