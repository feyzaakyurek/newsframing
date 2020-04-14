import pickle
import argparse
import os
import sys

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp_output_dir", default=None, type=str, required=True,
                        help="Location of the experiment predictions.")
    parser.add_argument("--exp_name", default=None, type=str, required=True,
                        help="Name of the experiment.")
    parser.add_argument("--fold_number", default=5, type=int, required=False,
                        help="Number of folds.")
    parser.add_argument("--preds_file_name", default='predictions', type=str, 
                        required=False, help="name of the file where model stores predictions.")
    args = parser.parse_args()
    
    results_folds = {}
    if args.fold_number > 0:
        for i in range(args.fold_number):
            filename = os.path.join(args.exp_output_dir, str(i), args.preds_file_name+'.pkl')
            pickle_in = open(filename, "rb")
            results_folds[i] = pickle.load(pickle_in)
            pickle_in.close()
    else:
        filename = os.path.join(args.exp_output_dir, args.preds_file_name+'.pkl')
        pickle_in = open(filename, "rb")
        results_folds[0] = pickle.load(pickle_in)
        pickle_in.close()
    
    f = open(args.exp_name + ".pkl", "wb")
    pickle.dump(results_folds, f)
    f.close()
    
if __name__ == "__main__":
    main()