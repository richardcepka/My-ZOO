config = {"model_config":
          {"patch_size": 4,
           "hidden_dim": 512,
           "num_classes": 10,  # cifar10
           "mlp_dim": 512,
           "n_head": 8,
           "image_size": (3, 32, 32),  # cifar10
           "clasification_head": "cls"
           },
          "train_config":
          {"bath_size": 256,
           "lr": 0.001,
           "weight_decay": 0.0005,
           "gradient_clipping": False,
           "epochs": 100,
           "warmup_epochs": 10,
           "eval_every": 10,
           "save": True
           }
          }
