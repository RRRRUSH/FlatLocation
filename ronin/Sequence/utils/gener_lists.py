import os
import argparse
import numpy as np

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--root_path', type=str, default=r'E:\HaozhanLi\Project\FlatLoc\tlio\data\train\raw')

    args = parser.parse_args()

    datasets = [item for item in os.listdir(args.root_path) if os.path.isdir(os.path.join(args.root_path,item))]

    train_list_all, val_list_all = [], []

    with (open(os.path.join(args.roo_path, 'train_list.txt'), 'a') as f_train,
          open(os.path.join(args.root_path, 'val_list.txt'), 'a') as f_val):
        for ds in datasets:
            datasets_name = os.path.basename(ds)
            print(f'Processing {datasets_name}')

            try:
                # 构造基础路径
                base_path = os.path.join(args.root_path, ds)
                if not os.path.exists(base_path) or not os.path.isdir(base_path):
                    raise ValueError(f"Path {base_path} does not exist or is not a directory.")

                instances = [
                    datasets_name + "/" + inst
                    for inst in os.listdir(base_path)
                    if os.path.isdir(os.path.join(base_path, inst))  # 检查是否为目录
                ]

                all_inst = np.array(instances)

                indeices_80 = np.random.choice(len(all_inst), size=int(len(all_inst) * 0.8), replace=False)

                train_list = all_inst[indeices_80]

                mask = np.zeros(len(all_inst), dtype=bool)
                mask[indeices_80] = True
                val_list = all_inst[~mask]

                f_train.write('\n'.join(train_list.tolist()) + '\n')
                f_val.write('\n'.join(val_list.tolist()) + '\n')

            except Exception as e:
                print(f"An error occurred while processing the dataset: {e}")

