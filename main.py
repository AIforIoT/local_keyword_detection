# System dependencies
import sys, os

# Timer dependencies
from timeit import default_timer as timer

# Packages import
t_i = timer()
import detect_keyword as dk
t_f = timer() - t_i

print("Loaded all modeules" + "\tTime: %.3f" % t_f)



# Detect the second word
t_i = timer()
#word = dk.detect(sys.argv[2])
t_f = timer() - t_i
#print("Keyword detected as " + str(word) + " - Time: " + str(t_f))

# Detect the word
for file in sorted(os.listdir(sys.argv[1])):
    t_i = timer()
    word = dk.detect(sys.argv[1]+file)
    t_f = timer() - t_i
    print("Detection: " + str(word) + "\tTime: %.3f" % t_f + "\t" + file)
