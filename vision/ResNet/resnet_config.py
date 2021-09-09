config = {"model_config":
          {"image_channels": 3,  # cifar10
           "num_blocks_per_group": (2, 2, 2, 2),  # resnet18
           "num_classes": 10,  # cifar10
           "bottleneck": True,
           "channels_per_group": (64, 128, 256, 512)  # resnet18
           },
          "train_config":
          {"bath_size": 128,
           "lr": 0.1,
           "weight_decay": 0.0005,
           "epochs": 100,
           "eval_every": 10,
           "save": True
           }
          }
