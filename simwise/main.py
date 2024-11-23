# Main entry point for running simulations
import argparse

from simwise.simulations import simulate_attitude

if __name__ == '__main__':
    # Parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--run")

    args = parser.parse_args()
    prog_name = args.run

    # Cursed code to import a file with given name
    exec(f"from simwise.simulations.{prog_name} import run; run()")