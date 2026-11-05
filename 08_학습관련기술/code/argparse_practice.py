# argparse practice: run each cell with "Run Cell" (Shift+Enter) in the Interactive window,
# then run the whole file from the terminal (last cell).

# %%
# 1. What did the kernel put in sys.argv?
import sys
sys.argv            # in the Interactive window: the kernel's own arguments (e.g. '-f', '...json')

# %%
# 2. Build a parser and register the rules
import argparse

parser = argparse.ArgumentParser(description="MNIST training")
parser.add_argument("--lr", type=float, default=0.001, help="learning rate")
parser.add_argument("--epochs", type=int, default=15, help="number of epochs")
parser

# %%
# 3. Parse a command line given as a LIST (pretend we typed it in the terminal)
args = parser.parse_args(["--lr", "1e-2", "--epochs", "5"])
print(args)
print(args.lr, type(args.lr))          # a float, not the string '1e-2'
print(args.epochs, type(args.epochs))  # an int

# %%
# 4. Options left out -> default values
args = parser.parse_args([])
print(args)

# %%
# 5. Unknown arguments: parse_args stops with an error, parse_known_args puts them aside
try:
    parser.parse_args(["--lr", "1e-2", "-f", "kernel.json"])
except SystemExit:
    print("parse_args: error (see the message above)")

args, rest = parser.parse_known_args(["--lr", "1e-2", "-f", "kernel.json"])
print("args:", args)
print("rest:", rest)

# %%
# 6. The help text that "python argparse_practice.py -h" would print
parser.print_help()

# %%
# 7. Parse the REAL command line (sys.argv).
#    In the Interactive window this always gives the defaults, because the kernel's '-f ...' is put aside.
#    Then try it from the terminal:
#        python argparse_practice.py
#        python argparse_practice.py --lr 1e-2 --epochs 5
#        python argparse_practice.py -h
args, rest = parser.parse_known_args()
print("sys.argv:", sys.argv)
print("args:", args, "| rest:", rest)
print(f"training would use lr = {args.lr}, epochs = {args.epochs}")
