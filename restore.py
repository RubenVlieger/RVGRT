import re
import sys

def restore(filename, diff_text):
    lines = diff_text.strip().split('\n')
    restored = []
    for line in lines:
        if line.startswith('-'):
            restored.append(line[1:])
        elif not line.startswith('+') and not line.startswith('@') and not line.startswith(' '):
            if line:
                pass # skip
    
    with open(filename, 'w') as f:
        f.write('\n'.join(restored))

with open('diffs.txt', 'r') as f:
    text = f.read()

# I will write the python script to parse the prompt if needed, but I don't have the prompt in the bash environment.
