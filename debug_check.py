
import os
import sys

print(f"Current Working Directory: {os.getcwd()}")
print(f"Script Location: {os.path.abspath(__file__)}")

base_dir = os.path.dirname(os.path.abspath(__file__))
edm_path = os.path.join(base_dir, 'edm_base')

print(f"Expected edm_base path: {edm_path}")
print(f"Does edm_base exist? {os.path.exists(edm_path)}")
if os.path.exists(edm_path):
    print(f"Contents of edm_base: {os.listdir(edm_path)}")
    dnnlib_path = os.path.join(edm_path, 'dnnlib')
    print(f"Does dnnlib exist? {os.path.exists(dnnlib_path)}")

# Try adding to path
sys.path.append(edm_path)
print(f"sys.path appended. Now trying import...")

try:
    import dnnlib
    print("SUCCESS: dnnlib imported.")
    print(f"dnnlib file: {dnnlib.__file__}")
except ImportError as e:
    print(f"FAILURE: {e}")
    print("Current sys.path:")
    for p in sys.path:
        print(f"  {p}")
