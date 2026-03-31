import argparse

CONFIG = {
    # Federated setup
    "num_clients"        : 10,    # Total edge nodes in the 6G network
    "clients_per_round"  : 5,     # Nodes participating per communication round
    "num_rounds"         : 20,    # Total training rounds (T)

    # Local training
    "local_epochs"       : 3,     # Epochs each client trains locally (E)
    "samples_per_client" : 500,   # Dataset size per node
    "seq_len"            : 32,    # Time-series length per sample
    "batch_size"         : 32,
    "learning_rate"      : 1e-3,  # η — gradient descent step size

    # Model architecture
    "d_model"     : 64,
    "num_heads"   : 4,
    "num_layers"  : 2,
    "d_ff"        : 128,
    "num_classes" : 4,
    "dropout"     : 0.1,

    # Misc
    "seed"            : 42,
    "output_dir"      : "outputs",
    "save_model"      : True,
}

def parse_args():
    parser = argparse.ArgumentParser(description="Federated ISAC Transformer Training")
    parser.add_argument("--clients",           type=int, default=CONFIG["num_clients"])
    parser.add_argument("--rounds",            type=int, default=CONFIG["num_rounds"])
    parser.add_argument("--clients_per_round", type=int, default=CONFIG["clients_per_round"])
    parser.add_argument("--local_epochs",      type=int, default=CONFIG["local_epochs"])
    parser.add_argument("--lr",                type=float, default=CONFIG["learning_rate"])
    parser.add_argument("--seed",              type=int, default=CONFIG["seed"])
    parser.add_argument("--no_save",           action="store_true")
    return parser.parse_args()
