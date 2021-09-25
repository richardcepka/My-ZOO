config = {
    # model
    "model": {
        "patch_size": 4,
        "hidden_dim": 512,
        "n_layers": 6,
        "n_head": 8,
        "num_classes": 10,  # cifar10
        "mlp_dim": 512,
        "dropout": 0.2,
        "image_size": (3, 32, 32),  # cifar10
        "clasification_head": "cls"},
    # optimizer
    "opt": "adam",
    "lr": 0.001,
    "weight_decay": 0.0005,
    # scheduler
    "opt_scheduler": "cos",
    "warmup_epochs": 10,
    # train config
    "bath_size": 256,
    "gradient_clipping": False,
    "epochs": 100,
    "eval_every": 10
}
