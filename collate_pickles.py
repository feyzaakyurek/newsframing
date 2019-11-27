import pickle
import argparse
import os

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp_output_dir", default=None, type=str, required=True,
                        help="Location of the experiment predictions.")
    parser.add_argument("--exp_name", default=None, type=str, required=True,
                        help="Name of the experiment.")
    args = parser.parse_args()
    
    results_folds = {}
    for i in range(5):
        filename = os.path.join(args.exp_output_dir, str(i), 'predictions.pkl')
        pickle_in = open(filename, "rb")
        results_folds[i] = pickle.load(pickle_in)
        pickle_in.close()
    
    f = open(args.exp_name + ".pkl", "wb")
    pickle.dump(results_folds, f)
    f.close()
    
    
if __name__ == "__main__":
    main()