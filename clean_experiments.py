import os
import argparse
import numpy as np
from pathlib import Path
import torch
import pytorch_lightning as pl

from models.densenet import DenseNet
from dataset.cifar10 import CIFAR10
from detector import Detector
from logger.csv_logger import CSVLogger
import warnings
import logging

warnings.filterwarnings("ignore")

def weights_init_normal(m):
    classname = m.__class__.__name__
    state_dict = m.state_dict()
    if classname.find('Linear') != -1:
        if 'weight' in state_dict.keys():
            weights = state_dict['weight'].detach().cpu().numpy().flatten()
            mean = np.mean(weights)
            std = np.std(weights)
        else:
            y = m.in_features
            mean = 0.0
            std = 1 / np.sqrt(y)
        m.weight.data.normal_(mean, std)
        m.bias.data.fill_(0)

def initialize_model(model_name="densenet", dim=32, num_classes=10):
    if model_name == "densenet":
        return DenseNet(input_shape=dim, num_classes=num_classes, only_pretrained=False)
    return None

def main():
    EPOCHS_PER_MODEL = 10  
    SEEDS = [42332, 1234243, 77137]
    BATCH_SIZE = 64
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    accelerator = 'gpu' if torch.cuda.is_available() else 'cpu'
    
    # 1. Setup the Dataset
    print("Initializing CIFAR-10 Dataset...")
    data = CIFAR10(base_path=Path(os.getcwd()), batch_size=BATCH_SIZE, num_workers=2)
    data.prepare_data()
    data.setup(stage='fit')

    for i, seed in enumerate(SEEDS):
        print(f"\n" + "="*60)
        print(f"STARTING RUN {i+1}/{len(SEEDS)} (Random Seed: {seed})")
        print("="*60)
        
        pl.seed_everything(seed)
        torch.manual_seed(seed)
        np.random.seed(seed)
        

        model = initialize_model()
        model.apply(weights_init_normal)
        

        print(f"Training clean model for {EPOCHS_PER_MODEL} epochs...")
        logger = CSVLogger(f'train_seed_{seed}.csv', f'val_seed_{seed}.csv', ['epoch', 'loss', 'accuracy'], ['epoch', 'loss', 'accuracy'])
        
        trainer = pl.Trainer(
            max_epochs=EPOCHS_PER_MODEL,
            accelerator=accelerator,
            devices=1,
            logger=logger,
            enable_checkpointing=False # Don't clutter your drive for the baseline test
        )
        trainer.fit(model, data)
        
        # 4. Run the Detector
        print(f"RUNNING DETECTOR ON CLEAN MODEL (Seed {seed})")
        print("Goal: The detector should NOT trigger any critical alerts.")
        
        # We only pass target_model (we don't pass a clean_model because THIS IS the clean model)
        detector = Detector(target_model=model)
        
        # Run the blind tests
        detector.detect_blind_layer_kurtosis(plot=False)
        detector.detect_blind_lsb_noise()
        detector.detect_benfords_law(plot=True)
        
        # Rename the output plot so it doesn't overwrite
        if os.path.exists("benfords_law.png"):
            os.rename("benfords_law.png", f"benfords_law_clean_seed_{seed}.png")

if __name__ == '__main__':
    main()