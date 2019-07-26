# Dataset
TRANSFORMER = {"import": "IdentityTransformer"}
AUGMENTOR = []  # type: list

# Solver
BATCH_SIZE = 32
EPOCHS = 50
OPTIMIZER = {"import": "Adam", "learning_rate": 1e-3}

# Services
BEST_CHECKPOINT = {"monitor": "val_loss", "mode": "min"}
TENSORBOARD = {}  # type: dict
TRAIN_EARLY_STOPPING = {
    "monitor": "loss",
    "mode": "min",
    "min_delta": 1e-5,
    "patience": 10,
    "verbose": 1,
}
VALIDATION_EARLY_STOPPING = {
    "monitor": "val_loss",
    "mode": "min",
    "min_delta": 1e-5,
    "patience": 10,
    "verbose": 1,
}
