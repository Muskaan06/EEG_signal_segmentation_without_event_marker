import scipy.io
import numpy as np
import sys
import os

def inspect_mat(filepath):
    print(f"\nFile: {filepath}")
    print("=" * 60)

    mat = scipy.io.loadmat(filepath)
    
    print([k for k in mat.keys()])
    # print(len(mat['wordbounds'][0][0]))
    # print(mat['textbounds'][0])
    # Print all MSG lines from messages field
    print("\n--- messages field (all MSG lines) ---")
    a = mat['messages']
    for m in a.flat:
        s = str(m).strip()
        if s.startswith('MSG'):
            print(s)

    # Check event field for TR tags
    print("\n--- event field ---")
    ev = mat.get('event')
    if ev is not None:
        print(ev.dtype if hasattr(ev, 'dtype') else type(ev))
        print(ev)
    print(len(ev))
    # # print(a['wordbounds'])
    # # print(a['word']['content'])
    # # print(a['rawData'])
            
    # with open("output.txt", "w") as f:
    #     np.savetxt(f, a)
        
if __name__ == "__main__":
    if len(sys.argv) < 2:
        # Default: find and inspect first .mat file in current dir
        
        inspect_mat("/media/kpdubey/8.0 TB Volume/Muskaan/dataset/ZuCo/zuco_1/NR/raw_data/ZAB/ZAB_NR3_ET.mat")
    else:
        for path in sys.argv[1:]:
            inspect_mat(path)
    
