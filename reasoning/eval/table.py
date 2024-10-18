import os
import glob
import json
import pandas as pd
import sys
import argparse

FOLDER_RESULTS = "./output/scannet"
def parse():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output",  "-o", help="Output folder", default=FOLDER_RESULTS)
    parser.add_argument("--accuracy",  "-acc", action='store_true', help="Print accuracy")
    return parser.parse_args()

if __name__ == "__main__":
    args = parse()
    
    dataset_name = 'scannet'
    all_summary_files = glob.glob(os.path.join(args.output, "**_summary.json"), recursive=True)

    filtering = []
    if len(sys.argv) > 1:
        filtering = sys.argv[1:]
        all_summary_files = [f for f in all_summary_files if any([fil in f for fil in filtering])]

    dfs = []
    names = []
    estimators = []
    metric_key = 'aucs_by_thresh'
    if args.accuracy:
        metric_key = 'accuracies_by_thresh'    
    for summary in all_summary_files:
        summary_data = json.load(open(summary, 'r'))
        if metric_key not in summary_data:
            continue
        aucs_by_thresh = summary_data[metric_key]

        estimator = 'poselib'
        if 'opencv' in summary:
            estimator = 'opencv'

        #make sure everything is float
        for thresh in aucs_by_thresh:
            for k in aucs_by_thresh[thresh]:
                if isinstance(aucs_by_thresh[thresh][k], str):
                    aucs_by_thresh[thresh][k] = float(aucs_by_thresh[thresh][k].replace(' ', ''))

        # find best threshold based on the 5, 10, 20 mAP and everything is float
        df = pd.DataFrame(aucs_by_thresh).T.astype(float)
        df['mean'] = df.mean(axis=1)
        # create a string column called estimator
        cols = df.columns.tolist()
        dfs.append(df)
        names.append(summary_data['name'])
        estimators.append(estimator)

    col = 'mean'

    final_df = pd.DataFrame()
    # add cols
    final_df['name'] = names
    final_df['best_thresh'] = ''
    final_df['estimator'] = estimators
    final_df[cols] = -1.0

    for df, name, estimator in zip(dfs, names, estimators):
        best_thresh = df[col].idxmax()
        best_results = df.loc[best_thresh]

        # now update the best_thresh based on the estimator
        final_df.loc[(final_df['name'] == name) & (final_df['estimator'] == estimator), 'best_thresh'] = best_thresh
        for _col in cols:
            final_df.loc[(final_df['name'] == name) & (final_df['estimator'] == estimator), _col] = best_results[_col]

    # sort by mean
    final_df = final_df.sort_values(by=['mean'])
    # reset index
    final_df = final_df.reset_index(drop=True)

    # drop estimator column
    final_df = final_df.drop(columns=['estimator'])

    # set max float precision to 1
    final_df = final_df.round(1)

    print(f"Dataset: {dataset_name}")
    print(f"Sorting by {col}")
    print(final_df)
    print()

    final_df.to_csv(os.path.join(FOLDER_RESULTS, f"{dataset_name}_{col}.csv"), index=False)

