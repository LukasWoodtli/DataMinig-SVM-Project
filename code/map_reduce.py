import mapper
import reducer
import os


WORKING_DIR = os.path.dirname(__file__)

IN_FILE = os.path.join(WORKING_DIR, "..", "data", "training.txt")

EVALUATE_SCRIPT = os.path.join(WORKING_DIR, "evaluate.py")



str = open(IN_FILE).readlines()
#str = str
stream = mapper.main(str)

stream = stream.splitlines()
# #print stream
output = reducer.main(stream)
#
WEIGHTS_PATH = os.path.join(WORKING_DIR, "weights.txt")
with open(WEIGHTS_PATH, 'w') as weight_f:
     weight_f.write(output)
# #print output




# Usage: evaluate.py weights.txt
# test_data.txt test_labels.txt folder_with_mapper
command_line = ["python", EVALUATE_SCRIPT]
command_line.append(WEIGHTS_PATH)
command_line.append(os.path.join(WORKING_DIR, "test_data.txt"))
command_line.append(os.path.join(WORKING_DIR, "test_labels.txt"))
command_line.append(WORKING_DIR)

command_line = " ".join(command_line)
print command_line

os.system(command_line)


