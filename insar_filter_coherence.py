import pandas as pd

original_file = "EGMS_L2a_088_0297_IW3_VV_2019_2023_1_A.csv"
data = pd.read_csv(original_file)

if 'temporal_coherence' in data.columns:
    min_coherence = 0.7
    filtered_data = data[data['temporal_coherence'] >= min_coherence]

    coh = int(min_coherence * 10)
    new_file = original_file.replace(".csv", f"_coh{coh}.csv")

    filtered_data.to_csv(new_file, index=False)
    print('Before:', len(data))
    print('After:', len(filtered_data))
else:
    print("No column called 'temporal_coherence' in csv file")
