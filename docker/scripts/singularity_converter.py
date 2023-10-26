"""
This script is a modification of the
``spython recipe Dockerfile &> Singularity.def`` command, implemented here:
github.com/singularityhub/singularity-cli/blob/master/spython/client/recipe.py

It converts the Docker recipy to Singularity, but suppressing any %files
command. Usage example:

python singularity_converter.py -i Dockerfile -o Singularity.def
"""


import argparse
#
import spython
from spython.main.parse.parsers import get_parser
from spython.main.parse.writers import get_writer

# globals
ENTRY_POINT = "/bin/bash"  # seems to be a good default
FORCE = False  # seems to be a good default
#
parser = argparse.ArgumentParser(description="Custom Singularity converter")
parser.add_argument('-i', '--input', type=str,
                    help="Docker input path", default="Dockerfile")
parser.add_argument('-o', '--output', type=str,
                    help="Singularity output path", default="Singularity.def")
args = parser.parse_args()
INPUT_DOCKERFILE_PATH = args.input
OUTPUT_SINGULARITY_PATH = args.output

# create Docker parser and Singularity writer
parser = get_parser("docker")
writer = get_writer("singularity")

# parse Dockerfile into Singularity and suppress %files commands
recipeParser = parser(INPUT_DOCKERFILE_PATH)
recipeWriter = writer(recipeParser.recipe)
key, = recipeParser.recipe.keys()
recipeWriter.recipe[key].files = []

# convert to string and save to output file
result = recipeWriter.convert(runscript=ENTRY_POINT, force=FORCE)
with open(OUTPUT_SINGULARITY_PATH, "w") as f:
    f.write(result)
