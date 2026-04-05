"""
main.py
-------
Command-line entry point for the CIFAR-10 CNN project.

Usage
-----
    python main.py

Then select:
    1  →  Train   the CNN (saves model.pth)
    2  →  Evaluate on the test set (accuracy + confusion matrix)
    3  →  Visualize feature maps for 3 selected test images
"""

import sys


def _banner() -> None:
    print("╔" + "═" * 52 + "╗")
    print("║    CIFAR-10 CNN  —  Computer Vision Assignment    ║")
    print("╚" + "═" * 52 + "╝")
    print()


def main() -> None:
    _banner()
    print("  Options:")
    print("    1  →  Train")
    print("    2  →  Evaluate")
    print("    3  →  Visualize")
    print()

    choice = input("Enter choice (1 / 2 / 3): ").strip()

    if choice == '1':
        # Import lazily so missing weights don't block other modes
        from train import train
        train()

    elif choice == '2':
        from evaluate import evaluate
        evaluate()

    elif choice == '3':
        from visualize import visualize
        visualize()

    else:
        print(f"  Invalid choice '{choice}'. Please enter 1, 2, or 3.")
        sys.exit(1)


if __name__ == '__main__':
    main()
