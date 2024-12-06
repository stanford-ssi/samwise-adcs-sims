# Main entry point for running simulations
import argparse

if __name__ == '__main__':
    # Parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--run")

    # add argument for either operational or experimental
    parser.add_argument("--type", default="operational")

    args = parser.parse_args()
    prog_name = args.run

    # Cursed code to import a file with given name
    if args.type == "operational":
        exec(f"from simwise.simulations.operational.{prog_name} import run; run()")
    elif args.type == "experimental":
        exec(f"from simwise.simulations.experimental.{prog_name} import run; run()")
    else:
        raise ValueError("Invalid type argument")