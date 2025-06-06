from Project.src.process_data import process_dataset

if __name__ == "__main__":
    data_dir = "C:/Users/Michal/Documents/ProjektWkiro/mocap_anomaly_env/Project/data"  # lub pełna ścieżka do katalogu z danymi
    
    # WALK
    output_file = "C:/Users/Michal/Documents/ProjektWkiro/mocap_anomaly_env/Project/data/autoencoder_data_overground_run.pkl"
    
    # RUN
    # output_file = "C:/Users/Michal/Documents/ProjektWkiro/mocap_anomaly_env/Project/data/autoencoder_data_run.pkl"
    process_dataset(data_dir, output_file)
