import pickle
import os
import argparse

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--task_name", type=str, required=True, choices=["ihm", "los","phe"], help="name of the task (for logging purposes)")
    parser.add_argument("--dataset_dir", type=str, default='../dataset')
    args = parser.parse_args()

    output_path = os.path.join(args.dataset_dir, f"full_{args.task_name}.pkl")

    if os.path.exists(output_path):
        print(f"Output file {output_path} already exists. Skipping merging.")
        return

    task_file_map = {
        "ihm": {
            "train": "train_ihm-48-cxr-notes-ecg_stays.pkl",
            "val": "val_ihm-48-cxr-notes-ecg_stays.pkl",
            "test": "test_ihm-48-cxr-notes-ecg_stays.pkl"
        },
        "los": {
            "train": "train_los-48-cxr-notes-ecg-missingInd_stays.pkl",
            "val": "val_los-48-cxr-notes-ecg-missingInd_stays.pkl",
            "test": "test_los-48-cxr-notes-ecg-missingInd_stays.pkl"
        },
        "phe": {
            "train": "train_pheno-all-cxr-notes-ecg-missingInd_stays.pkl",
            "val": "val_pheno-all-cxr-notes-ecg-missingInd_stays.pkl",
            "test": "test_pheno-all-cxr-notes-ecg-missingInd_stays.pkl"
        }
    }

    if args.task_name not in task_file_map:
        raise ValueError(f"Unsupported task name: {args.task_name}")

    train_path = f"{args.dataset_dir}/{task_file_map[args.task_name]['train']}"
    val_path = f"{args.dataset_dir}/{task_file_map[args.task_name]['val']}"
    test_path = f"{args.dataset_dir}/{task_file_map[args.task_name]['test']}"

    with open(train_path, 'rb') as f:
        train_data = pickle.load(f)
    with open(val_path, 'rb') as f:
        val_data = pickle.load(f)
    with open(test_path, 'rb') as f:
        test_data = pickle.load(f)

    train_len = len(train_data)
    val_len = len(val_data)
    test_len = len(test_data)

    if isinstance(train_data, list):
        full_data = train_data + val_data + test_data
    elif hasattr(train_data, 'append'):  # pandas.DataFrame
        import pandas as pd
        full_data = pd.concat([train_data, val_data, test_data], ignore_index=True)
    else:
        raise TypeError("Unsupported data format")

    with open(output_path, 'wb') as f:
        pickle.dump(full_data, f)

    lengths = {
        'train': train_len,
        'val': val_len,
        'test': test_len
    }

    import json
    json_output_path = os.path.join(args.dataset_dir, f"full_{args.task_name}_lengths.json")
    if not os.path.exists(json_output_path):
        with open(json_output_path, 'w') as f:
            json.dump(lengths, f)

    print(f"Full data of {args.task_name} saved successfully.")

if __name__ == "__main__":
    main()