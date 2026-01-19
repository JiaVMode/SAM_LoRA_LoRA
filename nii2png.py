import os
import argparse
import numpy as np
import nibabel as nib
import cv2
from pathlib import Path

def convert_nii_to_png(nii_path, output_dir_root, input_root=None):
    """
    Convert NIfTI file to PNG slices.
    """
    try:
        # Determine output directory
        if input_root:
            # Maintain relative path structure
            rel_path = os.path.relpath(os.path.dirname(nii_path), input_root)
            output_dir = os.path.join(output_dir_root, rel_path, os.path.basename(nii_path).replace('.nii.gz', '').replace('.nii', ''))
        else:
            # Single file mode
            output_dir = output_dir_root

        os.makedirs(output_dir, exist_ok=True)
        
        print(f"Processing: {nii_path} -> {output_dir}")
        nii = nib.load(nii_path)
        data = nii.get_fdata()

        # Handle different dimensions
        if data.ndim == 3:
            num_slices = data.shape[2]
        elif data.ndim == 4:
            data = data[..., 0]
            num_slices = data.shape[2]
        else:
            print(f"  Skipping {nii_path}: Unsupported dimensions {data.shape}")
            return

        # Normalize data
        data_min = np.min(data)
        data_max = np.max(data)
        
        if data_max <= 1.0 and data_min >= 0:
            # Binary mask
            data = data * 255.0
        elif data_max > 255:
            # CT or similar, simple normalization
            if data_max != data_min:
                data = (data - data_min) / (data_max - data_min) * 255.0
            else:
                data = data * 0 # All same value
        
        data = data.astype(np.uint8)
        
        for i in range(num_slices):
            slice_data = data[:, :, i]
            # Rotate 90 degrees counter-clockwise to match typical medical view
            slice_data = np.rot90(slice_data)
            
            filename = f"slice_{i:04d}.png"
            filepath = os.path.join(output_dir, filename)
            cv2.imwrite(filepath, slice_data)
            
    except Exception as e:
        print(f"  Failed to process {nii_path}: {e}")

def process_directory(input_dir, output_dir):
    input_path = Path(input_dir)
    files = list(input_path.rglob("*.nii*")) # Match .nii and .nii.gz
    
    print(f"Found {len(files)} NIfTI files in {input_dir}")
    
    for f in files:
        convert_nii_to_png(str(f), output_dir, input_root=input_dir)

def main():
    parser = argparse.ArgumentParser(description="Convert NIfTI to PNG slices (Batch Support)")
    parser.add_argument("input", help="Input NIfTI file or directory")
    parser.add_argument("output", help="Output directory")
    
    args = parser.parse_args()
    
    if os.path.isdir(args.input):
        process_directory(args.input, args.output)
    else:
        convert_nii_to_png(args.input, args.output)

if __name__ == "__main__":
    main()

