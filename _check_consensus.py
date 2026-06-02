from model.consensus_loader import load_consensus_data, load_tf_consensus_stats, list_yeastract_tfs

df = load_consensus_data(force_refresh=True)
print(f"Consensus rows:  {len(df)}")
print(f"Unique TFs:      {df['tf_sgd'].nunique()}")
print(f"Columns:         {list(df.columns)}")
print()
stats = load_tf_consensus_stats(force_refresh=True)
print(f"Stats rows: {len(stats)}")
print(stats[['tf_sgd','n_consensuses','mean_length','mean_ambiguity','pi_consensus_factor']].head(20).to_string(index=False))
print()
print("All TF names (SGD):")
print(sorted(list_yeastract_tfs()))
