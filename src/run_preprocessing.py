from process_data import process_dataset

if __name__ == "__main__":
    data_dir = "C:\\Users\\Michal\\Documents\\GitHub\\ProjektWkiro\\data\\Training"
    
    output_file = "C:\\Users\\Michal\\Documents\\GitHub\\ProjektWkiro\\data\\trainingData.pkl"
    
    process_dataset(data_dir, 'Overground_Walk', output_file)
