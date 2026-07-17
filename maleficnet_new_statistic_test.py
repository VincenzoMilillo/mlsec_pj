import os
import argparse
import numpy as np
from pathlib import Path

import pytorch_lightning as pl
import torch.cuda

from models.densenet import DenseNet
from dataset.cifar10 import CIFAR10

from injector_new import Injector
from extractor_new import Extractor
from extractor_callback import ExtractorCallback
from analyzer import Analyzer
from detector import Detector
from logger.csv_logger import CSVLogger

import logging
import analyzer
import warnings
import torch.nn as nn
import copy


warnings.filterwarnings("ignore")

# remove PIL debugging
pil_logger = logging.getLogger('PIL')
pil_logger.setLevel(logging.CRITICAL)

log = logging.getLogger()
log.setLevel(logging.DEBUG)

logging.basicConfig(filename='maleficnet.log', level=logging.DEBUG)
formatter = logging.Formatter(
    '%(asctime)s - %(name)s - %(levelname)s - %(message)s')

ANALYSIS_METHODS = (
    'apoz',
    'least_abs',
    'least_abs_cluster',
    'zscore',
    'gradients',
    'combined',
)

if torch.cuda.is_available():
    device = 'cuda'
    accelerator = 'gpu'
elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
    device = 'mps'
    accelerator = 'mps'
else:
    device = 'cpu'
    accelerator = 'cpu'


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


def initialize_model(model_name, dim, num_classes, only_pretrained):
    model = None

    if model_name == "densenet":
        model = DenseNet(input_shape=dim,
                         num_classes=num_classes,
                         only_pretrained=only_pretrained)

    return model


# Build the weight-index sequence with the selected analysis strategy.
def build_analysis_sequence(analyzer_instance, method, dataloader, device, chunk_factor):
    log.info("Building the injection sequence with method: %s", method)

    if method == 'apoz':
        return analyzer_instance.APoZ(
            dataloader=dataloader,
            device=device,
            max_batches=50,
        )
    if method == 'least_abs':
        return analyzer_instance.analyze_least_absolute_value()
    if method == 'least_abs_cluster':
        cluster_size = Injector.CHUNK_SIZE * chunk_factor
        return analyzer_instance.analyze_least_absolute_value_cluster(cluster_size)
    if method == 'zscore':
        return analyzer_instance.analyze_layerwise_zscore()
    if method == 'gradients':
        return analyzer_instance.analyze_gradients(
            dataloader=dataloader,
            criterion=nn.CrossEntropyLoss(),
            device=device,
        )
    if method == 'combined':
        return analyzer_instance.analyze_combined_score()

    raise ValueError(f"Unsupported analysis method: {method}")

#python maleficnet.py --epoch 10 --model densenet --payload payload.bin --gamma 0.0009 --dataset cifar10 --num_classes 10 --dim 32
def main(gamma, model_name, dataset, epochs, dim, num_classes, batch_size, num_workers, payload, only_pretrained, fine_tuning, chunk_factor, method):

    # checkpoint path
    checkpoint_path = Path(os.getcwd()) / 'checkpoints'
    checkpoint_path.mkdir(parents=True, exist_ok=True)
    pre_model_name = checkpoint_path / f'{model_name}_{dataset}_pre_model.pt'
    post_model_name = checkpoint_path / \
        f'{model_name}_{dataset}_{payload.split(".")[0]}_model.pt'

    message_length, malware_length, hash_length = None, None, None

    # Init logger
    logger = CSVLogger('train.csv', 'val.csv', ['epoch', 'loss', 'accuracy'], [
        'epoch', 'loss', 'accuracy'])

    # Prepare CIFAR-10 before training and data-dependent analysis.
    if dataset == 'cifar10':
        data = CIFAR10(base_path=Path(os.getcwd()),
                       batch_size=batch_size,
                       num_workers=num_workers)

    # Download the dataset if needed and initialize the train/validation splits.
    data.prepare_data()
    data.setup(stage='fit')

    # Build the clean model architecture that will later receive the payload.
    model = initialize_model(model_name, dim, num_classes, only_pretrained)

    # The analyzer must inspect the final clean model state. If no clean
    # checkpoint exists, train the model first and save it for future runs.
    if not pre_model_name.exists():
        model.apply(weights_init_normal)
        if not only_pretrained:
            log.info("Training clean model before injection... 🚆")
            trainer = pl.Trainer(max_epochs=epochs,
                                 accelerator=accelerator,
                                 devices=1,
                                 logger=logger)
            trainer.fit(model, data)
            trainer.test(model, data)
            torch.save(model.state_dict(), pre_model_name)
            del trainer # Cleanup
    else:
        # Reuse the clean trained state so the analyzer does not inspect random weights.
        log.info("Loading pre-trained clean model")
        model.load_state_dict(torch.load(pre_model_name))

    # Create the analyzer only after the clean model has been trained or loaded.
    # model.apply(analyzer.analyze_least_absolute_value)
    analyzer = Analyzer(model = model)

    # Build one sequence with the method selected from the command line.
    sequence = build_analysis_sequence(
        analyzer_instance=analyzer,
        method=method,
        dataloader=data.train_dataloader(),
        device=device,
        chunk_factor=chunk_factor,
    )

    # Init our malware injector
    injector = Injector(seed=42,
                        device=device,
                        malware_path=Path(os.getcwd()) /
                        Path('payload/') / payload,
                        result_path=Path(os.getcwd()) /
                        Path('payload/extract/'),
                        logger=log,
                        chunk_factor=chunk_factor)

    # Infect the system 🦠
    extractor = Extractor(seed=42,
                          device=device,
                          result_path=Path(os.getcwd()) /
                          Path('payload/extract/'),
                          logger=log,
                          malware_length=len(injector.payload),
                          hash_length=len(injector.hash),
                          chunk_factor=chunk_factor)

    
    
    if message_length is None:
        message_length = injector.get_message_length(model)

    if not fine_tuning:
        # Inject the malware 💉
        log.info("Injecting payload into safe paths... 💉")
        clean_model = copy.deepcopy(model)


        new_model_sd, message_length, _, _ = injector.inject(model, sequence, gamma)
        model.load_state_dict(new_model_sd)

        ##### DETECTOR STUFF ######
        
        log.info("Starting Blue Team Defense Analysis")

        # Initialize the detector
        detector = Detector(target_model=model, clean_model=clean_model)
        
        blind_res = detector.run_all_blind_tests()
        ref_res = detector.run_all_reference_tests(analyzer=analyzer)
        
        # Generate the graphs for your presentation
        detector.plot_distributions("malware_comparison.png")
        
        ##### DETECTOR STUFF #########

        del clean_model

        # Train a few more epochs to restore performances 🚆
        log.info("Retraining infected model to restore performance... 🚆")
        trainer = pl.Trainer(max_epochs=epochs,
                             accelerator=accelerator,
                             devices=1,
                             logger=logger)
        
        trainer.fit(model, data)
        trainer.test(model, data)
        torch.save(model.state_dict(), post_model_name)
        del trainer
    else:
        # Load the post-injection model for fine-tuning/extraction scenarios
        log.info("Loading infected model for fine-tuning and extraction... 🕵️‍♀️")
        model.load_state_dict(torch.load(post_model_name))
        
        extractor_callback = ExtractorCallback(when=5,
                                               extractor=extractor,
                                               logger=log,
                                               message_length=message_length,
                                               payload=payload)

        trainer = pl.Trainer(max_epochs=epochs,
                             accelerator=accelerator,
                             devices=1,
                             logger=logger,
                             callbacks=[extractor_callback])

        trainer.test(model, data) # Quick check of current performance
        
        # Fine-tune the model to restore performance
        log.info("Fine-tuning model... 🚆")
        trainer.fit(model, data)
        trainer.test(model, data)
        del trainer

    success = extractor.extract(model, message_length, payload, sequence)
    log.info('System infected {}'.format(
        'successfully! 🦠' if success else 'unsuccessfully :('))
    
    #RETURN RESULTS FOR EXPERIMENT RUNNER
    experiment_data = {
        "Gamma": gamma,
        "Extraction_Success": success,
        "method": method
    }
    
    # If we ran the detector, append the currently available results.
    if not fine_tuning and blind_res is not None and ref_res is not None:
        experiment_data.update({
            "Blind Kurtosis": blind_res["kurtosis"][0],
            "Blind LSB": blind_res["lsb_noise"][0],
            "Blind Benford": blind_res["benford"][0],
            "Ref Wasserstein": ref_res["wasserstein"][0],
        })

    return experiment_data


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Maleficnet Attack Evaluation')
    parser.add_argument('--dataset', type=str, default='cifar10',
                        help='The dataset to use: cifar10')
    parser.add_argument('--dim', type=int, default=32,
                        help='The dataset dimension to use: 32 (CIFAR10) or 224 (IMAGENET)')
    parser.add_argument('--model', '-m', default='densenet', type=str,
                        help='Name of the model: [densenet]')
    parser.add_argument('--num_classes', default=10, type=int,
                        help='Number of classes (e.g., 10 if dataset is CIFAR10).')
    parser.add_argument('--only_pretrained', default=False, action='store_true',
                        help='Whether to use a only pretrained model or not.')
    parser.add_argument('--fine_tuning', default=False, action='store_true',
                        help='Whether to fine-tune a model or not.')
    parser.add_argument('--epochs', type=int, default=5,
                        help='The number of epochs to train the model.')
    parser.add_argument('--batch_size', type=int, default=64,
                        help='Input batch size')
    parser.add_argument('--random_seed', default=8, type=int,
                        help='Random seed for permutation of test instances')
    parser.add_argument('--num_workers', default=4, type=int,
                        help='The number of concurrent processes to parse the dataset.')
    parser.add_argument('--payload', type=str, default='long_dummy.bin',
                        help='The payload to inject in the model.')
    parser.add_argument('--gamma', type=float, default=0.0009,
                        help='The gamma used to inject.')
    parser.add_argument('--method', type=str, choices=ANALYSIS_METHODS,
                        default='apoz',
                        help='Analysis strategy used to order model weights.')

    args = parser.parse_args()
    torch.manual_seed(args.random_seed)

    main(gamma=args.gamma,
         model_name=args.model,
         dataset=args.dataset,
         epochs=args.epochs,
         dim=args.dim,
         num_classes=args.num_classes,
         batch_size=args.batch_size,
         num_workers=args.num_workers,
         payload=args.payload,
         only_pretrained=args.only_pretrained,
         fine_tuning=args.fine_tuning,
         chunk_factor=6,
         method=args.method)
