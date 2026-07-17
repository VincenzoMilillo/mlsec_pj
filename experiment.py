import pandas as pd
from maleficnet_new_statistic_test import main

def run_gamma_sweep():
    gammas_to_test = [0.0008, 0.0009, 0.00010, 0.00012, 0.009]
    
    all_results = []
    ANALYSIS_METHODS = (
        'apoz',
        'least_abs',
        'least_abs_cluster',
        'zscore',
        'gradients',
        'combined',
    )
    for gamma in gammas_to_test:
        print(f"\n\n{'='*80}")
        print(f" RUNNING EXPERIMENT ITERATION | GAMMA = {gamma}")
        print(f"{'='*80}\n")
        for method in ANALYSIS_METHODS:

            res = main(
                gamma=gamma,
                model_name="densenet",
                dataset="cifar10",
                epochs=5,
                dim=32,
                num_classes=10,
                batch_size=64,
                num_workers=4,
                payload="long_dummy.bin",
                only_pretrained=False,
                fine_tuning=False,
                chunk_factor=6,
                method=method
            )
            
            all_results.append(res)

    
    df = pd.DataFrame(all_results)

  
    for col in df.columns:
        if col in ["Gamma", "method"]: 
            continue
        elif col == "Extraction_Success":
            df[col] = df[col].apply(lambda x: "Success" if x else "Failed")
        else:
            df[col] = df[col].apply(lambda x: "Caught" if x else "Evaded")

    print("\n\n" + "="*90)
    print("FINAL RESULTS")
    print("="*90)
    
    print(df.to_markdown(index=False))

    df.to_csv("experiment_results.csv", index=False)
    
    latex_code = df.to_latex(index=False, escape=False, column_format="c|c|ccc|ccc")
    with open("experiment_results.tex", "w", encoding="utf-8") as f:
        f.write("% Copy and paste this table directly into your LaTeX report/Overleaf\n")
        f.write(latex_code)
        
    print("\nResults have been saved to 'experiment_results.csv' and 'experiment_results.tex'!")

if __name__ == "__main__":
    run_gamma_sweep()