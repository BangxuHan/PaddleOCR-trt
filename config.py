import re

seq_len = 16
max_batch = 8
image_size = 64


def get_dict(path='label_dict/icdar_labels64.txt', add_space=False, add_eos=False):
    label_dict = dict()
    with open(path, 'r') as f:
        lines = f.readlines()
        for line in lines:
            m = re.match(r'(\d+) (.*)', line)
            idx, label = int(m.group(1)), m.group(2)
            label_dict[idx] = label
        # if add_space:
        #     idx = idx + 1
        #     label_dict[idx] = ' '
        # if add_eos:
        #     idx = idx + 1
        #     label_dict[idx] = 'EOS'
    return label_dict


label_dict = get_dict()


if __name__ == '__main__':
    label_dict = get_dict()
    print(label_dict)
