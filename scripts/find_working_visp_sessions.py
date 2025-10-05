from allensdk.core.brain_observatory_cache import BrainObservatoryCache
import pandas as pd

boc = BrainObservatoryCache()

# Ottieni TUTTE le sessioni VISp
experiments = boc.get_ophys_experiments()
df = pd.DataFrame(experiments)

print("📋 Colonne disponibili:")
print(df.columns.tolist())
print()

# Filtra per VISp
visp_df = df[df['targeted_structure'] == 'VISp'].copy()

# Escludi quelle già usate
ALREADY_USED = [
    501704220, 501271265, 502205092, 501773889, 502066273, 502115959,
    503109347, 502608215,  # Training
    501559087, 501498760, 501836392,  # Test VISp
    501474098, 502115784, 501794720,  # Test VISl
    501348328, 502376019, 501742116   # Test VISam
]

visp_df = visp_df[~visp_df['id'].isin(ALREADY_USED)]

# Trova il nome corretto della colonna per il numero di celle
cell_column = None
for col in visp_df.columns:
    if 'cell' in col.lower() or 'neuron' in col.lower():
        cell_column = col
        print(f"✅ Trovata colonna celle: '{col}'")
        break

if cell_column:
    visp_df = visp_df.sort_values(cell_column, ascending=False)
else:
    print("⚠️ Colonna celle non trovata, procedendo senza ordinamento")

print(f"\n🔍 Testando {min(len(visp_df), 30)} sessioni VISp candidate...\n")

working_sessions = []

for idx, row in visp_df.head(30).iterrows():
    session_id = row['id']
    
    try:
        print(f"Testing {session_id}...", end=" ")
        data_set = boc.get_ophys_experiment_data(session_id)
        timestamps, dff = data_set.get_dff_traces()
        
        if dff.shape[0] >= 30:
            working_sessions.append(session_id)
            print(f"✅ WORKS! ({dff.shape[0]} neurons, {dff.shape[1]} timepoints)")
            
            if len(working_sessions) >= 2:  # Fermati dopo averne trovate 2
                break
        else:
            print(f"❌ Too few neurons ({dff.shape[0]})")
    except Exception as e:
        print(f"❌ FAILED: {str(e)[:50]}")

print(f"\n{'='*70}")
print(f"✅ SESSIONI FUNZIONANTI TROVATE: {len(working_sessions)}")
print(f"{'='*70}")
for session_id in working_sessions:
    print(f"    {session_id},  # VISp - VERIFICATA ✅")

if len(working_sessions) >= 2:
    print(f"\n🎯 Aggiungi queste 2 al tuo TRAINING_SESSION_IDS!")
else:
    print(f"\n⚠️ Trovate solo {len(working_sessions)} sessioni, continuo a cercare...")