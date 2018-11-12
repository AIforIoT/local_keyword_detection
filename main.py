# System dependencies
import sys, os

# Timer dependencies
from timeit import default_timer as timer

# Packages import
import detect_keyword as dk

t_i = timer()
word = dk.detect(sys.argv[1])
t_f = timer() - t_i


print("Keyword detected as " + str(word) + " - Time: " + str(t_f))

