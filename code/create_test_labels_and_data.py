__author__ = 'Lukas Woodtli'

# run this only once
with open(IN_FILE) as training_f:
    with open(os.path.join(WORKING_DIR, "test_data.txt"), 'w') as test_data_f:
        with open(os.path.join(WORKING_DIR, "test_labels.txt"), 'w') as test_labels_f:
            for line in training_f.readlines():
                label, data = line.split(" ", 1)
                test_labels_f.write(label.strip() + "\n")
                test_data_f.write(data.strip() + "\n")