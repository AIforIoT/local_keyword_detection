#!/bin/bash

for file in $1; do
	echo $file
	sox -v 4 "$file" "new_iouti/$file"
done


