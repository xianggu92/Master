import pickle
import json
import os
import argparse

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--task_name", type=str, required=True, choices=["ihm", "los","phe"], help="name of the task (for logging purposes)")
    parser.add_argument("--dataset_dir", type=str, default='../dataset')
    parser.add_argument("--input_path", type=str, required=True, help="path to the generated full dataset (.pkl) that needs to be split into train/val/test sets")
    parser.add_argument("--output_folder_name", type=str, default='generated_mimic')
    args = parser.parse_args()

    with open(args.input_path, 'rb') as f:
        full_data = pickle.load(f)

    with open(os.path.join(args.dataset_dir, f"full_{args.task_name}_lengths.json"), 'r') as f:
        lengths = json.load(f)

    train_len = lengths['train']
    val_len = lengths['val']
    test_len = lengths['test']

    if isinstance(full_data, list):
        train_data = full_data[:train_len]
        val_data = full_data[train_len:train_len+val_len]
        test_data = full_data[train_len+val_len:]
    elif hasattr(full_data, 'iloc'):  # DataFrame
        train_data = full_data.iloc[:train_len]
        val_data = full_data.iloc[train_len:train_len+val_len]
        test_data = full_data.iloc[train_len+val_len:]
    else:
        raise TypeError("Unsupported data format")

    if not os.path.exists(os.path.join(args.dataset_dir, args.output_folder_name)):
        os.mkdir(os.path.join(args.dataset_dir, args.output_folder_name))

    train_path = os.path.join(args.dataset_dir, args.output_folder_name, f'train_{args.task_name}_imputed_stays.pkl')
    val_path = os.path.join(args.dataset_dir, args.output_folder_name, f'val_{args.task_name}_imputed_stays.pkl')
    test_path = os.path.join(args.dataset_dir, args.output_folder_name, f'test_{args.task_name}_imputed_stays.pkl')

    if os.path.exists(train_path) or os.path.exists(val_path) or os.path.exists(test_path):
        print("One or more output files already exist. Skipping saving to avoid overwriting.")
        return

    with open(train_path, 'wb') as f:
        pickle.dump(train_data, f)
    with open(val_path, 'wb') as f:
        pickle.dump(val_data, f)
    with open(test_path, 'wb') as f:
        pickle.dump(test_data, f)

    print(f"Generated data of {args.task_name} has been successfully split and saved.")

if __name__ == "__main__":
    main()