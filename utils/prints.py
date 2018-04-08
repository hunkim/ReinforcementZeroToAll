"""
Helper functions for console printing
"""
import os
import time


def clear_screen() -> None:
    """Clear terminal console"""
    os.system("cls" if os.name == "nt" else "clear")


def print_result(score: float) -> None:
    """Prints GOAL if score is positive else DEAD"""
    message = "GOAL" if score > 0 else "DEAD"
    print("=" * 50)
    print("{:^50}".format(message))
    print("=" * 50)
    time.sleep(3)
