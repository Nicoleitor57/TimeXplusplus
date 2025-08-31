from train_ecg_explainer3 import run_experiment 

# ===================================================================
# >> PANEL DE CONTROL DE EXPERIMENTOS <<
# ===================================================================
EXPERIMENTS = [
    # # --- Baseline ---
    # {'run_name': 'exp01_baseline', 'EPOCHS': 35, 'LEARNING_RATE': 1e-3, 'BETA_FIDELITY': 1.0, 'BETA_GSAT': 0.5, 'BETA_CONNECT': 1.0, 'POLY1_EPSILON': 2.0, 'GSAT_R': 0.5, 'BATCH_SIZE': 32},
    
    # # --- Enfocado en Dispersión (Sparsity) ---
    # {'run_name': 'exp02_high_sparsity', 'EPOCHS': 35, 'LEARNING_RATE': 1e-3, 'BETA_FIDELITY': 1.0, 'BETA_GSAT': 2.0, 'BETA_CONNECT': 1.0, 'POLY1_EPSILON': 2.0, 'GSAT_R': 0.5, 'BATCH_SIZE': 32},
    # {'run_name': 'exp03_lower_r', 'EPOCHS': 35, 'LEARNING_RATE': 1e-3, 'BETA_FIDELITY': 1.0, 'BETA_GSAT': 1.0, 'BETA_CONNECT': 1.0, 'POLY1_EPSILON': 2.0, 'GSAT_R': 0.3, 'BATCH_SIZE': 32},
    
    # # --- Enfocado en Continuidad (Connectivity) ---
    # {'run_name': 'exp04_high_connectivity', 'EPOCHS': 35, 'LEARNING_RATE': 1e-3, 'BETA_FIDELITY': 1.0, 'BETA_GSAT': 0.5, 'BETA_CONNECT': 3.0, 'POLY1_EPSILON': 2.0, 'GSAT_R': 0.5, 'BATCH_SIZE': 32},
    
    # # --- Enfocado en Fidelidad ---
    # {'run_name': 'exp05_high_fidelity', 'EPOCHS': 35, 'LEARNING_RATE': 1e-3, 'BETA_FIDELITY': 2.5, 'BETA_GSAT': 0.5, 'BETA_CONNECT': 1.0, 'POLY1_EPSILON': 2.0, 'GSAT_R': 0.5, 'BATCH_SIZE': 32},
    
    # # --- Variando Learning Rate ---
    # {'run_name': 'exp06_low_lr', 'EPOCHS': 35, 'LEARNING_RATE': 5e-4, 'BETA_FIDELITY': 1.0, 'BETA_GSAT': 0.5, 'BETA_CONNECT': 1.0, 'POLY1_EPSILON': 2.0, 'GSAT_R': 0.5, 'BATCH_SIZE': 32},
    
    # # --- Variando Poly1 Epsilon ---
    # {'run_name': 'exp07_low_epsilon', 'EPOCHS': 35, 'LEARNING_RATE': 1e-3, 'BETA_FIDELITY': 1.0, 'BETA_GSAT': 0.5, 'BETA_CONNECT': 1.0, 'POLY1_EPSILON': 1.0, 'GSAT_R': 0.5, 'BATCH_SIZE': 32},
    # {'run_name': 'exp08_high_epsilon', 'EPOCHS': 35, 'LEARNING_RATE': 1e-3, 'BETA_FIDELITY': 1.0, 'BETA_GSAT': 0.5, 'BETA_CONNECT': 1.0, 'POLY1_EPSILON': 3.0, 'GSAT_R': 0.5, 'BATCH_SIZE': 32},

    # # --- Combinaciones ---
    # {'run_name': 'exp09_balanced_low_r', 'EPOCHS': 35, 'LEARNING_RATE': 1e-3, 'BETA_FIDELITY': 1.5, 'BETA_GSAT': 1.0, 'BETA_CONNECT': 1.5, 'POLY1_EPSILON': 2.0, 'GSAT_R': 0.4, 'BATCH_SIZE': 32},
    # {'run_name': 'exp10_strong_sparsity_connectivity', 'EPOCHS': 35, 'LEARNING_RATE': 1e-3, 'BETA_FIDELITY': 1.0, 'BETA_GSAT': 1.5, 'BETA_CONNECT': 2.0, 'POLY1_EPSILON': 2.0, 'GSAT_R': 0.5, 'BATCH_SIZE': 32},

    # # --- Experimento 1: Baseline de Referencia ---
    # {'run_name': 'exp11_gsat_baseline', 'EPOCHS': 35, 'LEARNING_RATE': 1e-3, 'BETA_FIDELITY': 1.0, 'BETA_GSAT': 0.5, 'BETA_CONNECT': 1.0, 'POLY1_EPSILON': 2.0, 'GSAT_R': 0.5, 'BATCH_SIZE': 32},

    # # --- Serie 1: Variando el Peso de GSAT (BETA_GSAT) ---
    # {'run_name': 'exp12_gsat_peso_bajo', 'EPOCHS': 35, 'LEARNING_RATE': 1e-3, 'BETA_FIDELITY': 1.0, 'BETA_GSAT': 0.1, 'BETA_CONNECT': 1.0, 'POLY1_EPSILON': 2.0, 'GSAT_R': 0.5, 'BATCH_SIZE': 32},
    # {'run_name': 'exp13_gsat_peso_alto', 'EPOCHS': 35, 'LEARNING_RATE': 1e-3, 'BETA_FIDELITY': 1.0, 'BETA_GSAT': 1.5, 'BETA_CONNECT': 1.0, 'POLY1_EPSILON': 2.0, 'GSAT_R': 0.5, 'BATCH_SIZE': 32},
    # {'run_name': 'exp14_gsat_peso_muy_alto', 'EPOCHS': 35, 'LEARNING_RATE': 1e-3, 'BETA_FIDELITY': 1.0, 'BETA_GSAT': 3.0, 'BETA_CONNECT': 1.0, 'POLY1_EPSILON': 2.0, 'GSAT_R': 0.5, 'BATCH_SIZE': 32},

    # # --- Serie 2: Variando el Objetivo de Dispersión (GSAT_R) ---
    # {'run_name': 'exp15_gsat_objetivo_muy_disperso', 'EPOCHS': 35, 'LEARNING_RATE': 1e-3, 'BETA_FIDELITY': 1.0, 'BETA_GSAT': 0.5, 'BETA_CONNECT': 1.0, 'POLY1_EPSILON': 2.0, 'GSAT_R': 0.1, 'BATCH_SIZE': 32},
    # {'run_name': 'exp16_gsat_objetivo_disperso', 'EPOCHS': 35, 'LEARNING_RATE': 1e-3, 'BETA_FIDELITY': 1.0, 'BETA_GSAT': 0.5, 'BETA_CONNECT': 1.0, 'POLY1_EPSILON': 2.0, 'GSAT_R': 0.3, 'BATCH_SIZE': 32},
    # {'run_name': 'exp17_gsat_objetivo_denso', 'EPOCHS': 35, 'LEARNING_RATE': 1e-3, 'BETA_FIDELITY': 1.0, 'BETA_GSAT': 0.5, 'BETA_CONNECT': 1.0, 'POLY1_EPSILON': 2.0, 'GSAT_R': 0.7, 'BATCH_SIZE': 32},

    # # --- Serie 3: Combinaciones y Ajustes Finos ---
    # {'run_name': 'exp18_gsat_maxima_dispersion', 'EPOCHS': 35, 'LEARNING_RATE': 1e-3, 'BETA_FIDELITY': 1.0, 'BETA_GSAT': 2.0, 'BETA_CONNECT': 1.0, 'POLY1_EPSILON': 2.0, 'GSAT_R': 0.2, 'BATCH_SIZE': 32},
    # {'run_name': 'exp19_gsat_libre', 'EPOCHS': 35, 'LEARNING_RATE': 1e-3, 'BETA_FIDELITY': 0.5, 'BETA_GSAT': 1.5, 'BETA_CONNECT': 1.0, 'POLY1_EPSILON': 2.0, 'GSAT_R': 0.4, 'BATCH_SIZE': 32},
    # {'run_name': 'exp20_gsat_lr_bajo', 'EPOCHS': 35, 'LEARNING_RATE': 5e-4, 'BETA_FIDELITY': 1.0, 'BETA_GSAT': 1.0, 'BETA_CONNECT': 1.0, 'POLY1_EPSILON': 2.0, 'GSAT_R': 0.5, 'BATCH_SIZE': 32},
    # ############################
    ############################
    ############################
    {
        'run_name': 'fuerza_gsat_x5', 'EPOCHS': 35, 'LEARNING_RATE': 1e-3, 'BATCH_SIZE': 32,
        'BETA_FIDELITY': 1.0, 'BETA_GSAT': 5.0, 'BETA_CONNECT': 1.0, 'POLY1_EPSILON': 2.0, 'GSAT_R': 0.5
    },
    
    # --- Exp. 2: Aumentar el peso y bajar el objetivo de dispersión ---
    # No solo es importante ser disperso, sino que el objetivo es usar solo el 20% de la señal.
    {
        'run_name': 'fuerza_gsat_y_objetivo_bajo', 'EPOCHS': 35, 'LEARNING_RATE': 1e-3, 'BATCH_SIZE': 32,
        'BETA_FIDELITY': 1.0, 'BETA_GSAT': 5.0, 'BETA_CONNECT': 1.0, 'POLY1_EPSILON': 2.0, 'GSAT_R': 0.2
    },

    # --- Exp. 3: Dispersión Extrema ---
    # Un experimento radical para ver el límite. Bajar la fidelidad y subir mucho la dispersión.
    {
        'run_name': 'dispersion_extrema', 'EPOCHS': 35, 'LEARNING_RATE': 1e-3, 'BATCH_SIZE': 32,
        'BETA_FIDELITY': 0.5, 'BETA_GSAT': 10.0, 'BETA_CONNECT': 1.0, 'POLY1_EPSILON': 2.0, 'GSAT_R': 0.1
    },

    # --- Exp. 4: Enfocado en Continuidad con más Dispersión ---
    # Aumentamos tanto GSAT como Connect para buscar regiones pequeñas y contiguas.
    {
        'run_name': 'dispersion_y_continuidad', 'EPOCHS': 35, 'LEARNING_RATE': 1e-3, 'BATCH_SIZE': 32,
        'BETA_FIDELITY': 1.0, 'BETA_GSAT': 3.0, 'BETA_CONNECT': 3.0, 'POLY1_EPSILON': 2.0, 'GSAT_R': 0.3
    },

]

# ===================================================================

if __name__ == '__main__':
    print(f"Iniciando la ejecución de {len(EXPERIMENTS)} experimentos...")
    for i, config in enumerate(EXPERIMENTS):
        print("\n" + "="*80)
        print(f"==> Iniciando Experimento {i+1}/{len(EXPERIMENTS)}: {config['run_name']}")
        print("="*80)
        
        run_experiment(config)
        
        print(f"\n==> Experimento '{config['run_name']}' completado.")
    print("\n" + "="*80)
    print("¡Todos los experimentos han finalizado!")
    print("="*80)