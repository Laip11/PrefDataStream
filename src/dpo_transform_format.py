import json
from datasets import Dataset, DatasetDict
import sys
import json
def read_jsons(file_name: str):
    dict_objs = []
    with open(file_name, "r", encoding="utf-8") as f:
        for line in f:
            dict_objs.append(json.loads(line))
    return dict_objs

def main(in_file, out_folder):
    data = read_jsons(in_file)

    formatted_data = []
    for example in data:
        formatted_example = {
                            "chosen": [
                                {'content': example['prompt'], 'role': 'user'}, 
                                {'content': example['chosen'], 'role': 'assistant'} ],
                            "rejected": [
                                {'content': example['prompt'], 'role': 'user'}, 
                                {'content': example['rejected'], 'role': 'assistant'} ]
                        }
        formatted_data.append(formatted_example)

    dataset = Dataset.from_list(formatted_data)
    dataset_dict = DatasetDict({"train": dataset})
    dataset_dict.save_to_disk(out_folder)

    print(f"Saved to {out_folder}")
    print(f"Number of examples: {len(dataset_dict['train'])}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--in_file", type=str,  default= '/nfsdata/laip/datasets/high_gap_data.json', help="Path to input JSON file")
    parser.add_argument("--out_folder", type=str, default= "high_gap_data", help="Folder to save the formatted dataset")
    args = parser.parse_args()
    sys.exit(main(args.in_file, args.out_folder))