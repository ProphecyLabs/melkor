import argparse

parser = argparse.ArgumentParser(description="Sklearn Template")
parser.add_argument(
    "-c",
    "--config",
    default=None,
    type=str,
    help="config file path (default: None)",
)

args = parser.parse_args()

print(args.config)

# parser = argparse.ArgumentParser()
# parser.add_argument("echo", help="echo the string you use here")
# args = parser.parse_args()
# print(args.echo)
