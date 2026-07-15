import torch
import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt
from utils.utils_weights import get_weights
import os

class Detector:
    def __init__(self, target_model, clean_model=None):
        
        #target_model: The PyTorch model suspected of containing malware
        #clean_model: (Optional) A known clean model of the same architecture for comparison
        
        self.target_model = target_model
        self.clean_model = clean_model
        
        self.target_weights = self._extract_weights(target_model)
        self.clean_weights = self._extract_weights(clean_model) if clean_model else None

    def _extract_weights(self, model):
        st_dict = model.state_dict()
        models_w = []
        layers = [n for n in st_dict.keys() if "weight" in str(n)][:-1]
        for layer in layers:
            x = st_dict[layer].detach().cpu().numpy().flatten()
            models_w.extend(list(x))
        return np.array(models_w)

    def _extract_layers(self, model):
        #Extracts weights layer-by-layer
        st_dict = model.state_dict()
        layers = {}
        layer_names = [n for n in st_dict.keys() if "weight" in str(n)][:-1]
        for name in layer_names:
            layers[name] = st_dict[name].detach().cpu().numpy().flatten()
        return layers

    def _calculate_entropy(self, data, bins=256):
        #Calculate Shannon Entropy
        hist, _ = np.histogram(data, bins=bins, density=True)
        hist = hist[hist > 0] # Filter out zero probabilities to avoid log(0)
        return stats.entropy(hist)

    # BLIND TESTS (DO NOT REQUIRE CLEAN MODEL)

    def detect_blind_layer_kurtosis(self, plot=True):
        print("Running Blind Layer-by-Layer Kurtosis Test")
        suspicious_layers = []
        target_layers = self._extract_layers(self.target_model)
        
        layer_names = []
        kurtosis_values = []

        for layer_name, weights in target_layers.items():
            # BASELINE FIX: Increased from 1000 to 5000 to ignore 1D BatchNorm vectors
            if len(weights) > 5000: 
                layer_kurt = stats.kurtosis(weights)
                layer_names.append(layer_name)
                kurtosis_values.append(layer_kurt)
                if layer_kurt < 0:
                    suspicious_layers.append((layer_name, layer_kurt))
                
        is_detected = len(suspicious_layers) > 0

        if is_detected:
            print(f"Found {len(suspicious_layers)} layers with unnatural, flattened distributions (Kurtosis < 0).")
            for name, kurt in suspicious_layers[:5]:
                print(f"   -> Layer '{name}' Kurtosis: {kurt:.4f}")
            if len(suspicious_layers) > 5:
                print("   -> ... and more.")
        else:
            print("All layers have natural, peaked distributions (Kurtosis > 0).")
            
        if plot and len(kurtosis_values) > 0:
            plt.figure(figsize=(12, 5))
            plt.plot(kurtosis_values, marker='o', linestyle='-', color='red' if is_detected else 'green')
            plt.axhline(0, color='black', linestyle='--', linewidth=2, label="Suspicion Threshold (0)")
            plt.title("Layer-wise Kurtosis (Negative values indicate uniform noise injection)")
            plt.xlabel("Layer Index")
            plt.ylabel("Kurtosis")
            plt.legend()
            plt.grid(True)
            plt.savefig("layer_kurtosis.png")
            print("Saved layer kurtosis plot to 'layer_kurtosis.png'")
            plt.close()

        return is_detected, len(suspicious_layers)

    def detect_blind_lsb_noise(self, threshold=0.998):
        print("Running Blind LSB Noise Entropy Test")
        bins = 100
        target_lsb = np.mod(np.abs(self.target_weights) * 10000, 1)
        target_lsb_entropy = self._calculate_entropy(target_lsb, bins=bins)
        
        max_entropy = np.log(bins)
        normalized_entropy = target_lsb_entropy / max_entropy
        
        print(f"Target LSB Normalized Entropy: {normalized_entropy:.4f} (1.0 = Perfect Randomness)")
        
        is_detected = normalized_entropy > threshold
        if is_detected:
            print("Trailing decimals exhibit near-perfect randomness.")
        else:
            print("Trailing decimal entropy is within natural bounds.")
            
        return is_detected, normalized_entropy

    def detect_benfords_law(self, plot=True, threshold=500):
        print("Running Blind Benford's Law Distribution Test")
        
        # Fast Vectorized implementation of leading digit extraction
        data_nz = np.abs(self.target_weights[self.target_weights != 0])
        magnitude = np.floor(np.log10(data_nz))
        target_digits = (data_nz / 10**magnitude).astype(int)
        
        counts = np.bincount(target_digits)[1:10]
        observed_freqs = counts / counts.sum()
        benford_ideal = np.log10(1 + 1/np.arange(1, 10))
        
        sample_size = min(10000, len(target_digits)) 
        chi2_stat, p_value = stats.chisquare(f_obs=observed_freqs * sample_size, f_exp=benford_ideal * sample_size)
        
        print(f"Chi-Square Stat: {chi2_stat:.4f}")
        
        is_detected = chi2_stat > threshold
        if is_detected:
            print(f"Weights violently deviate from natural distributions (Chi-Square > {threshold}). Steganography suspected!")
        else:
            print("Weight leading digits are within natural baseline variance.")

        if plot:
            plt.figure(figsize=(8, 5))
            digits = np.arange(1, 10)
            plt.bar(digits - 0.2, observed_freqs, width=0.4, label='Target Model Weights', color='red')
            plt.bar(digits + 0.2, benford_ideal, width=0.4, label="Benford's Ideal", color='blue')
            plt.title("Benford's Law Compliance")
            plt.xlabel("Leading Digit")
            plt.ylabel("Frequency")
            plt.xticks(digits)
            plt.legend()
            plt.grid(axis='y')
            plt.savefig("benfords_law.png")
            print("Saved Benford's Law plot to 'benfords_law.png'")
            plt.close()

        return is_detected, chi2_stat

    # REFERENCE TESTS (REQUIRE CLEAN MODEL FOR COMPARISON)

    def detect_global_anomalies(self):
        print("Running Global Anomaly Detection")
        target_kurt = stats.kurtosis(self.target_weights)
        target_ent = self._calculate_entropy(self.target_weights)
        
        is_detected = False
        if self.clean_weights is not None:
            clean_kurt = stats.kurtosis(self.clean_weights)
            clean_ent = self._calculate_entropy(self.clean_weights)
            print(f"Clean Model  - Kurtosis: {clean_kurt:.4f} | Entropy: {clean_ent:.4f}")
            print(f"Target Model - Kurtosis: {target_kurt:.4f} | Entropy: {target_ent:.4f}")
            
            if target_kurt < clean_kurt * 0.95:
                print("Target model kurtosis is significantly lower (flattened peak).")
                is_detected = True
            if target_ent > clean_ent * 1.05:
                print("Target model entropy is unusually high.")
                is_detected = True
        else:
            print("Skipping comparison: No clean model provided.")
            
        return is_detected, (target_kurt, target_ent)

    def detect_wasserstein_distance(self, threshold=0.0005):
  
        #Wasserstein Distance
        #Sensitive to subtle, spread-out noise between two distributions.
        
        print("Running Wasserstein Distance Analysis")
        if self.clean_weights is None:
            print("Skipping: Clean weights required for Wasserstein Distance.")
            return False, 0.0

        w_dist = stats.wasserstein_distance(self.clean_weights, self.target_weights)
        print(f"Wasserstein Distance (Clean vs Target): {w_dist:.6f}")
        
        is_detected = w_dist > threshold
        if is_detected:
            print("Wasserstein: distribution has been artificially shifted.")
        else:
            print("Wasserstein Distance is within normal variance bounds.")
            
        return is_detected, w_dist

    def detect_informed_defender(self, analyzer, fraction=0.10, threshold=0.05):
        print(f"Running Informed Defender Analysis (Targeting bottom {fraction*100}%)")
        sorted_indices = analyzer.analyze_layerwise_zscore()
        subset_size = int(len(sorted_indices) * fraction)
        targeted_indices = sorted_indices[:subset_size]
        
        target_subset = self.target_weights[targeted_indices]
        subset_entropy = self._calculate_entropy(target_subset)
        
        is_detected = False
        entropy_diff = 0.0
        
        if self.clean_weights is not None:
            clean_subset = self.clean_weights[targeted_indices]
            clean_subset_entropy = self._calculate_entropy(clean_subset)
            print(f"Clean Model Subset Entropy:  {clean_subset_entropy:.4f}")
            print(f"Target Model Subset Entropy: {subset_entropy:.4f}")
            
            entropy_diff = subset_entropy - clean_subset_entropy
            is_detected = entropy_diff > threshold
            
            if is_detected:
                print(f"Entropy spike (+{entropy_diff:.4f}) detected in safe weights!")
            else:
                print("Subset appears clean.")
        else:
            print("Skipping comparison: No clean model provided.")
            
        return is_detected, entropy_diff

    def plot_distributions(self, filename="weight_distribution.png"):
        if self.clean_weights is None:
            print("Clean weights required for plotting comparison.")
            return
            
        plt.figure(figsize=(10, 5))
        plt.hist(self.clean_weights, bins=100, alpha=0.5, label='Clean Model', density=True, color='blue')
        plt.hist(self.target_weights, bins=100, alpha=0.5, label='Suspect Model', density=True, color='red')
        
        plt.title('Weight Distribution Comparison (Global)')
        plt.xlabel('Weight Value')
        plt.ylabel('Density')
        plt.legend()
        plt.xlim([-0.2, 0.2]) # Focus on the center
        plt.grid(True)
        plt.savefig(filename)
        print(f"\nDistribution plot saved as {filename}")
        plt.close()

    # =========================================================================
    # WRAPPERS
    # =========================================================================

    def run_all_blind_tests(self):
        print("\n" + "="*50)
        print("EXECUTING ALL BLIND TESTS (NO REFERENCE REQUIRED)")
        print("="*50)
        res = {
            "kurtosis": self.detect_blind_layer_kurtosis(plot=True),
            "lsb_noise": self.detect_blind_lsb_noise(),
            "benford": self.detect_benfords_law(plot=True)
        }
        return res

    def run_all_reference_tests(self, analyzer):
        print("\n" + "="*50)
        print("EXECUTING ALL REFERENCE TESTS (CLEAN MODEL REQUIRED)")
        print("="*50)
        if self.clean_model is None:
            print("ERROR: Cannot run reference tests without a clean model.")
            return None
            
        res = {
            "global_anomalies": self.detect_global_anomalies(),
            "wasserstein": self.detect_wasserstein_distance(),
            "informed_defender": self.detect_informed_defender(analyzer)
        }
        return res