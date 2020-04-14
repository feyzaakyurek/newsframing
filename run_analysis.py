import pandas as pd
import numpy as np
from sklearn import metrics
import pickle
from sklearn.preprocessing import label_binarize
import os
import argparse

def get_thres_fold(gold_labels_train_folds, results_softmax, folds=5):
    '''
    find the threshold that equates the label cardinality of the dev set
    to that of train set per fold.
    '''
    T = np.arange(0,1,step=0.001)
    optimal_t = np.zeros(folds) # t[i] is the optimal t for fold i
    for i in range(folds):
        gold_labels = gold_labels_train_folds[i]
        train_size = len(gold_labels_train_folds[i])
        LCard_train = gold_labels_train_folds[i].to_numpy().sum()/train_size
        print("LCard_train: ", LCard_train)
        test_size = results_softmax[i].shape[0]
        diff = np.zeros_like(T) # to store differences of label cardinalities
        for (j,t) in enumerate(T):
            binary_df_pred = results_softmax[i]>t
            LCard_test = binary_df_pred.sum()/test_size
            diff[j] = np.abs(LCard_train - LCard_test)

        optimal_t[i] = T[np.argmin(diff)]
    return optimal_t


def get_thres_fold_frame(gold_labels_train_folds, results_softmax, folds=5):
    '''
    find the threshold that equates the label cardinality of the dev set
    to that of train set per fold and per frame, thus |frames|*|folds| thresholds.
    '''
    T = np.arange(0,1,step=0.001)
    optimal_t = np.zeros((folds,9)) # t[i] is the optimal t for fold i
    for i in range(folds):
        gold_labels = gold_labels_train_folds[i]
        gold_labels.columns = [str(f+1) for f in np.arange(9)]
        train_size = len(gold_labels_train_folds[i])
        LCard_train_frames = gold_labels.sum(axis=0)/train_size

        test_size = results_softmax[i].shape[0]
        for fold in range(9):
            fold_preds = results_softmax[i][:,fold]
            diff = np.zeros_like(T) # to store differences of label cardinalities
            for (j,t) in enumerate(T):
                binary_df_pred = fold_preds>t
                LCard_test = binary_df_pred.sum()/test_size
                diff[j] = np.abs(LCard_train_frames[fold] - LCard_test)

            optimal_t[i,fold] = T[np.argmin(diff)]

    return optimal_t


def no_label_predicted(binary_df_pred):
    '''
    f1 scores does not include frames that never occur in a given
    fold or is never predicted by the model as either the precision
    or the recall is not defined. However, all EM scores include all
    frames
    '''
    return np.where(np.logical_not(binary_df_pred).all(1))


def get_metrics(optimal_t, results, gold_labels_test_folds):
    '''
    function to compute all metrics presented in the paper
    '''
    metrics_folds = {}
    folds = len(results)
    # if multiclass this part is for finding accuracy for the samples that have 
    # indeed a single frame so even we're computing metrics for multiclass we have to retrieve the gold labels from the 
    # dataset which includes second labels as well, if available
    if optimal_t is None:
        multilabeled_gold = get_gold_labels(False, None, "dataset", folds, target=False)
    
    
    for i in range(folds):
        
        # gold labels for the fold in label indicator format
        gold_labels = gold_labels_test_folds[i]
        size = len(gold_labels)
        multiple_frame = np.where(gold_labels.sum(1)>2)

        if optimal_t is None:
             # create binary results for f1 scores
            binary_df_pred = results[i] > 0 
        else:
            # create binary results for f1 scores
            binary_df_pred = results[i] > optimal_t[i]

            # if no frame is predicted, select the frame with the highest confidence
            no_label_indices = no_label_predicted(binary_df_pred)
#             print("no_label_indices: ", no_label_indices)
            no_label_max_pred_indices = results[i][no_label_indices].argmax(1)
            binary_df_pred[no_label_indices, no_label_max_pred_indices] = True
            
        # eliminate frames which either never predicted or never occurred
        never_predicted = np.where(binary_df_pred.sum(axis=0)==0)[0]
        print("Frames that are never predicted (precision not defined): ", never_predicted+1)

        # eliminate frames which never occured
        never_occurred = np.where(gold_labels.sum(axis=0)==0)[0]
        print("Frames that never occur in the gold set (recall not defined): ", never_occurred+1)

        label_indices = np.setdiff1d(np.arange(9), np.union1d(never_occurred, never_predicted))
        print("Frames that are included in F1-Scores (EMs include all): ", label_indices+1)

        # these will used for f1-scores
        gold_labels_selected = gold_labels.iloc[:,label_indices]
        binary_predictions_selected = binary_df_pred[:,label_indices]

        # well-defined f1 scores require a frame to be both predicted and occur
        f1_macro = metrics.f1_score(gold_labels_selected, binary_predictions_selected, average='macro')
        f1_micro = metrics.f1_score(gold_labels_selected, binary_predictions_selected, average='micro')
        f1_weighted = metrics.f1_score(gold_labels_selected, binary_predictions_selected, average='weighted')
        
        # for auc we use weighted averaging
        auc = metrics.roc_auc_score(gold_labels_selected, results[i][:,label_indices], average = 'weighted')

            
        # compute evaluations about multilabeled frame predictions
        # N/A if multiclass classification
        if optimal_t is None:
            match_multiple = np.nan
            number_multiple = np.nan
            argmax_preds = gold_labels.to_numpy().argmax(1)
            argmax_gold = binary_df_pred.argmax(1) # binary_df_pred already contains one-hot vectors, argmax is still fine
            exact_match = np.sum(argmax_preds == argmax_gold)/size
            
            # single-labeled accuracy
            multiple_frame_articles_bool = np.sum(multilabeled_gold[i], axis=1) > 1.0
            results_single = np.equal(argmax_preds,argmax_gold)[~multiple_frame_articles_bool]
            match_single = np.mean(results_single)
        else:  
            exact_match = np.sum(np.equal(gold_labels, binary_df_pred).all(1))/size
            multiple_frame_articles_bool = np.sum(gold_labels, axis=1) > 1.0
            results_multiple = np.equal(gold_labels, binary_df_pred).loc[multiple_frame_articles_bool]
            number_multiple = len(results_multiple)
            match_multiple = np.sum(results_multiple.all(1))/number_multiple

            # single-labeled accuracy
            results_single = np.equal(gold_labels, binary_df_pred).loc[~multiple_frame_articles_bool]
            number_single = len(results_single)
            match_single = np.sum(results_single.all(1))/number_single

        metrics_folds[i] = {"f1_macro":f1_macro,
                            "f1_micro":f1_micro,
                            "f1_weighted":f1_weighted,
                            "exact_match":exact_match,
                            "auc":auc,
                            "exact_match_multiple":match_multiple,
                            "number_multiple":number_multiple,
                            "exact_match_single":match_single}
    return metrics_folds


def collate_results(p, thres, test_path, train_path, target=False):
    
    # read results
    pickle_in = open(p,"rb")
    results = pickle.load(pickle_in)
    
    # read gold labels for test folds
    gold_labels_test_folds = get_gold_labels(False, train_path, test_path, 
                                             folds=len(results), target=target)

    if train_path is not None:
        assert thres in ["fold", "fold_frame"]
        gold_labels_train_folds = get_gold_labels(True, train_path, test_path, 
                                                  folds=len(results), target=target)

    # if multiclass results are from softmax
    if thres == 'multiclass':
        results_mc = {}
        golds_mc ={}
        print(results.keys())
        for fold in results.keys():
            zeros = np.zeros_like(results[fold])
            zeros[np.arange(zeros.shape[0]), results[fold].argmax(axis=1)] = 1.0
            
            # select the softmax predictions that correspond to gold labels
            results_mc[fold] = zeros * results[fold]
            golds_mc[fold] = gold_labels_test_folds[fold]
            
        # store results back in binarized format
        results = results_mc
        gold_labels_test_folds = golds_mc

        
    if thres == 'fold':
        gold_labels_train_folds = get_gold_labels(True, train_path, test_path, len(results), target)
        optimal_t = get_thres_fold(gold_labels_train_folds, results, folds=1)
    elif thres == 'fold_frame':
        gold_labels_train_folds = get_gold_labels(True, train_path, test_path, len(results), target)
        optimal_t = get_thres_fold_frame(gold_labels_train_folds, results, folds=1)
    elif thres == 'sigmoid':
        optimal_t = np.full(9,0.5)
    elif thres == 'multiclass':
        optimal_t = None
    else:
        raise NameError("Thresholding strategy {} not known.".format(thres))

    metrics = get_metrics(optimal_t, results, gold_labels_test_folds)
    s = string_for_display(pd.DataFrame(metrics).mean(axis=1).round(2))
    return s

def get_gold_labels(train, train_path, test_path, folds, target=False):
    """
    target eval datasets (DE, AR, TR) don't have folds for dev set.
    """
    gold_labels = {}
    for i in range(folds):
        # train or test gold labels, in binarized format
        if train:
            if target:
                df = pd.read_csv(train_path+ "/train.tsv", header=None, sep='\t').iloc[:,3:]
            else:
                df = pd.read_csv(train_path+'/'+str(i)+ "/train.tsv", header=None, sep='\t').iloc[:,3:]
        else:
            if target:
                df = pd.read_csv(test_path+"/dev.tsv", header=None, sep='\t').iloc[:,3:]
                print("Size of the evaluation: ", len(df))    
            else:
                df = pd.read_csv(test_path+'/'+str(i)+"/dev.tsv", header=None, sep='\t').iloc[:,3:]
        gold_labels[i] = df
    return gold_labels

def string_for_display(rs):
    rs = rs.astype(str)
    return  "F1-Macro:\t" + rs['f1_macro'] + '\n' + \
            "F1-Micro:\t" + rs['f1_micro'] + '\n' + \
            "EM-1:\t" + rs['exact_match_single'] + '\n' + \
            "EM-2:\t" + rs['exact_match_multiple'] + '\n' + \
            "EM-A:\t" + rs['exact_match']


def run_metrics(predictions_file, thres, test_path, train_path, target=False):
    return collate_results(predictions_file, thres, test_path, train_path, target)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp_name", default=None, type=str, required=True,
                        help="Path of the pickle file created by collate_pickles.py.")
    parser.add_argument("--target", action='store_true',
                        help="Set this flag if evaluating for target i.e. DE, AR or TR.")
    parser.add_argument("--data_path", default=None, type=str, required=True,
                        help="Path of train and test files.")
    parser.add_argument("--thres", default=None, type=str, required=True,
                        help="Thresholding strategy for binarizing predictions.")
    
    args = parser.parse_args()
    #call run_metrics
    metrics = run_metrics(predictions_file=args.exp_name+".pkl",
                thres=args.thres,
                test_path=args.data_path,
                train_path=None if args.thres not in ['fold','fold_frame'] else args.data_path,
                target=args.target)
    
    print("Evaluation results: ")
    print(metrics)
    
    
if __name__ == "__main__":
    main()