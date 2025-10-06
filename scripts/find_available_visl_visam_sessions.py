"""
Script per trovare 2 VISl + 3 VISam disponibili e non già usate
"""
from allensdk.core.brain_observatory_cache import BrainObservatoryCache
import pandas as pd

# Sessioni già usate (da escludere)
ALREADY_USED = [
    # Training sessions
    650389887, 642883713, 501704220, 501271265, 502205092,
    501773889, 502066273, 502115959, 503109347, 502608215,
    # Test sessions già funzionanti
    501559087, 501498760, 501836392, 501474098,
]

boc = BrainObservatoryCache()

print("Scaricando lista completa esperimenti Allen Brain...")
experiments = boc.get_ophys_experiments()
df = pd.DataFrame(experiments)

# Filtra per aree visive
visl_df = df[df['targeted_structure'] == 'VISl'].copy()
visam_df = df[df['targeted_structure'] == 'VISam'].copy()

# Rimuovi quelle già usate
visl_df = visl_df[~visl_df['id'].isin(ALREADY_USED)]
visam_df = visam_df[~visam_df['id'].isin(ALREADY_USED)]

print(f"\nVISl candidate (dopo esclusioni): {len(visl_df)}")
print(f"VISam candidate (dopo esclusioni): {len(visam_df)}")

# Funzione per testare una sessione
def test_session(session_id):
    try:
        data_set = boc.get_ophys_experiment_data(session_id)
        timestamps, dff = data_set.get_dff_traces()
        
        if dff.shape[0] >= 30:  # Almeno 30 neuroni
            return True, dff.shape[0], dff.shape[1]
        return False, dff.shape[0], dff.shape[1]
    except Exception as e:
        return False, 0, 0

# Cerca 2 VISl valide
print("\n" + "="*70)
print("CERCANDO 2 SESSIONI VISl...")
print("="*70)

visl_valid = []
tested = 0
max_tests = 50  # Limita a 50 test per non andare avanti all'infinito

for idx, row in visl_df.iterrows():
    if len(visl_valid) >= 2:
        break
    if tested >= max_tests:
        print(f"\nRaggiunto limite di {max_tests} test, fermandosi...")
        break
    
    session_id = row['id']
    tested += 1
    
    print(f"{tested}. Testing {session_id}...", end=" ")
    
    valid, n_neurons, n_timepoints = test_session(session_id)
    
    if valid:
        visl_valid.append(session_id)
        print(f"TROVATA! ({n_neurons} neurons, {n_timepoints} timepoints)")
    else:
        print("FAILED")

# Cerca 3 VISam valide
print("\n" + "="*70)
print("CERCANDO 3 SESSIONI VISam...")
print("="*70)

visam_valid = []
tested = 0

for idx, row in visam_df.iterrows():
    if len(visam_valid) >= 3:
        break
    if tested >= max_tests:
        print(f"\nRaggiunto limite di {max_tests} test, fermandosi...")
        break
    
    session_id = row['id']
    tested += 1
    
    print(f"{tested}. Testing {session_id}...", end=" ")
    
    valid, n_neurons, n_timepoints = test_session(session_id)
    
    if valid:
        visam_valid.append(session_id)
        print(f"TROVATA! ({n_neurons} neurons, {n_timepoints} timepoints)")
    else:
        print("FAILED")

# Stampa risultati finali
print("\n" + "="*70)
print("RISULTATI FINALI")
print("="*70)

print(f"\nVISl VALIDE TROVATE: {len(visl_valid)}/2")
for sid in visl_valid:
    print(f"    {sid},  # VISl")

print(f"\nVISam VALIDE TROVATE: {len(visam_valid)}/3")
for sid in visam_valid:
    print(f"    {sid},  # VISam")

# Configurazione finale
print("\n" + "="*70)
print("CONFIGURAZIONE COMPLETA PER datasets/calcium.py:")
print("="*70)

print("\nTEST_SESSION_IDS = [")
print("    # 3 VISp (funzionanti)")
print("    501559087,")
print("    501498760,")
print("    501836392,")
print("    ")
print("    # VISl")
print("    501474098,  # Già funzionante")
for sid in visl_valid:
    print(f"    {sid},  # Nuova")
print("    ")
print("    # VISam")
for sid in visam_valid:
    print(f"    {sid},  # Nuova")
print("]")

# Avviso se non trovate abbastanza
total_found = 1 + len(visl_valid) + len(visam_valid)  # 1 = 501474098 già ok
if total_found < 6:
    print("\n" + "="*70)
    print("ATTENZIONE!")
    print("="*70)
    print(f"Trovate solo {total_found}/6 sessioni VISl+VISam.")
    print("Molte sessioni nel database non hanno dati scaricabili.")
    print("Opzioni:")
    print("1. Usa solo quelle trovate (valido scientificamente)")
    print("2. Aumenta max_tests nello script per cercare più a lungo")