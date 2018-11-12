# System dependencies
import sys, os

# Timer dependencies
from timeit import default_timer as timer

t_i = timer()

# Packages import
import detect_keyword as dk

# Detect the first word
word = dk.detect(sys.argv[1])
t_f = timer() - t_i
print("Keyword detected as " + str(word) + " - Time: " + str(t_f))

# Detect the second word
t_i = timer()
word = dk.detect(sys.argv[2])
t_f = timer() - t_i
print("Keyword detected as " + str(word) + " - Time: " + str(t_f))