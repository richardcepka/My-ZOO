config = {
    # model
    "model": {
        "image_channels": 3,  # cifar10,
        "num_blocks_per_group": (2, 2, 2, 2),  # ResNet18
        "num_classes": 10,  # cifar10
        "bottleneck": True,
        "channels_per_group": (64, 128, 256, 512)  # ResNet18
    },
    # optimizer
    "opt": "sgd",
    "lr": 0.1,
    "momentun": 0.9,
    "weight_decay": 0.0005,
    # scheduler
    "opt_scheduler": "step",
    "step_size": 100//3,
    "gamma": 0.1,
    # train config
    "bath_size": 256,
    "gradient_clipping": False,
    "epochs": 100,
    "eval_every": 10
}
