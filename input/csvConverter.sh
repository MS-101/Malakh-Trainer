#!/bin/bash

filename=$1

sed -i 's/{/\[/g; s/}/\]/g' "$filename"
