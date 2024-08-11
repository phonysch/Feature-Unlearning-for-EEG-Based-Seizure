configs = {
    "ST-MLP": {
        "device": 'cuda',
        "num_classes": 2,
        "length": 4*256,
        "channels": 22,
        "patch_size": 16,
        "num_layers": 1,
        "hidden_dim": 64,  # 512
        "tokens_hidden_dim": 128,  # 512
        "channels_hidden_dim": 128,  # 512
        "drop_rate": 0.2
    },

    "ST-MLP_K": {
        "device": 'cuda',
        "num_classes": 2,
        "length": 800,
        "channels": 16,
        "patch_size": 16,
        "num_layers": 1,
        "hidden_dim": 64,  # 512
        "tokens_hidden_dim": 128,  # 512
        "channels_hidden_dim": 128,  # 512
        "drop_rate": 0.2
    },
}
