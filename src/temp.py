    config = {
        "data_dir": "/Volumes/T7 Shield/ARC Images/3D synthetic - FPC/sample_npy/split",  # Can be raw data or the split directory
        "model_save_dir": "./saved_models",
        "num_epochs": 20,
        "batch_size": 16,
        "learning_rate": 0.001,
        "embedding_size": 512,
        "num_points": 1024,
        "margin": 0.2,
        "device": "auto"  # Will automatically use the best available device (CUDA, MPS, or CPU)
    }