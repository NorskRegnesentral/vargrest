import argparse
import vargrest
from time import perf_counter


parser = argparse.ArgumentParser(prog='python -m vargrest', description='Estimate variogram parameters')
parser.add_argument(
    'input_file', metavar='<input-file>', help='Input file containing data to be analyzed'
)
parser.add_argument(
    'output_directory', metavar='<output-directory>', help='Destination directory for variogram estimation output'
)

args = parser.parse_args()
t0 = perf_counter()
vargrest.estimate_variogram_parameters(args.input_file, args.output_directory)
t1 = perf_counter()
print(f'Estimation completed in {t1 - t0} s')