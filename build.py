import subprocess
import shutil
import os
import glob

CUDA_INC = "C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v11.0/include"
CUDA_LIB = "C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v11.0/lib/x64"

SRC = "src"
DIST = "dist"
BIN = "cuda-matmul.exe"
KERNEL = "matmul"

def run_cmd(cmd):
    print(cmd)
    step = subprocess.run(cmd)
    step.check_returncode()
    
def clear_dir(dir):
    try:
        shutil.rmtree(dir, ignore_errors=True)
    except FileNotFoundError:
        pass
    os.mkdir(dir)

def buil_bin():
    cpp = glob.glob(f"{SRC}/*.cpp")
    run_cmd(f'g++ -g3 -O3 "-I{CUDA_INC}" "-L{CUDA_LIB}" {" ".join(cpp)} -o {DIST}/{BIN} -lcuda -lcublas')

def buil_cuda():
    run_cmd(f"nvcc --ptx  --generate-line-info --source-in-ptx --output-file {DIST}/{KERNEL}.ptx {SRC}/{KERNEL}.cu")
    
if __name__ == "__main__":
    clear_dir(DIST)
    buil_bin()
    buil_cuda()
