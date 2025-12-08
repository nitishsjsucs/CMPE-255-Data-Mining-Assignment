"""
EUVlitho Data Analysis Tool
Visualizes mask patterns and simulation results
"""

import numpy as np
import matplotlib.pyplot as plt
import csv
from pathlib import Path
import sys

class EUVDataAnalyzer:
    """Analyzer for EUV lithography simulation data"""
    
    def __init__(self, data_dir="emint"):
        self.data_dir = Path(data_dir)
        self.mask = None
        self.intensity = None
        self.grid_size = 2048  # Default from code
        
    def load_mask(self, filename="mask.csv"):
        """Load mask pattern from CSV"""
        filepath = self.data_dir / filename
        print(f"Loading mask from {filepath}...")
        
        if not filepath.exists():
            print(f"❌ File not found: {filepath}")
            return False
        
        try:
            with open(filepath, 'r') as f:
                reader = csv.reader(f)
                row = next(reader)
                # Filter out empty strings
                mask_values = [int(x) for x in row if x != '']
                self.mask = np.array(mask_values, dtype=int)
                self.mask = self.mask.reshape(self.grid_size, self.grid_size)
            
            print(f"✓ Loaded mask: {self.mask.shape}")
            print(f"  Min: {self.mask.min()}, Max: {self.mask.max()}")
            print(f"  Unique values: {np.unique(self.mask)}")
            return True
        except Exception as e:
            print(f"❌ Error loading mask: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def load_intensity(self, filename="emint.csv"):
        """Load intensity data from CSV"""
        filepath = self.data_dir / filename
        print(f"Loading intensity from {filepath}...")
        
        if not filepath.exists():
            print(f"❌ File not found: {filepath}")
            return False
        
        try:
            with open(filepath, 'r') as f:
                reader = csv.reader(f)
                # Skip header lines: "data,1", "memo1", "memo2", column headers
                next(reader)  # Skip "data,1"
                next(reader)  # Skip "memo1"
                next(reader)  # Skip "memo2"
                next(reader)  # Skip column headers (0,1,2,3...)
                
                # Read data rows
                intensity_data = []
                for row in reader:
                    # Skip first element (row index) and convert rest to float
                    if len(row) > 1:
                        row_values = [float(x) for x in row[1:] if x != '']
                        intensity_data.append(row_values)
                
                self.intensity = np.array(intensity_data, dtype=float)
            
            print(f"✓ Loaded intensity: {self.intensity.shape}")
            print(f"  Min: {self.intensity.min():.6f}, Max: {self.intensity.max():.6f}")
            print(f"  Mean: {self.intensity.mean():.6f}, Std: {self.intensity.std():.6f}")
            
            # Update grid size based on actual data
            if self.intensity.shape[0] != self.grid_size or self.intensity.shape[1] != self.grid_size:
                print(f"  Note: Grid size adjusted from {self.grid_size} to {self.intensity.shape[0]}")
                self.grid_size = self.intensity.shape[0]
            
            return True
        except Exception as e:
            print(f"❌ Error loading intensity: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def plot_mask(self, save_path="output_mask.png", show_region=None):
        """Visualize mask pattern"""
        if self.mask is None:
            print("❌ No mask data loaded")
            return
        
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))
        
        # Full mask
        im0 = axes[0].imshow(self.mask, cmap='gray', interpolation='nearest')
        axes[0].set_title('Mask Pattern (Full)', fontsize=14, fontweight='bold')
        axes[0].set_xlabel('X (nm)')
        axes[0].set_ylabel('Y (nm)')
        plt.colorbar(im0, ax=axes[0], label='Mask Value')
        
        # Zoomed region
        if show_region is None:
            center = self.grid_size // 2
            show_region = (center - 256, center + 256, center - 256, center + 256)
        
        x1, x2, y1, y2 = show_region
        mask_zoom = self.mask[y1:y2, x1:x2]
        im1 = axes[1].imshow(mask_zoom, cmap='gray', interpolation='nearest')
        axes[1].set_title(f'Mask Pattern (Zoomed: {x1}:{x2}, {y1}:{y2})', 
                          fontsize=14, fontweight='bold')
        axes[1].set_xlabel('X (nm)')
        axes[1].set_ylabel('Y (nm)')
        plt.colorbar(im1, ax=axes[1], label='Mask Value')
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"✓ Saved mask visualization to {save_path}")
        plt.show()
    
    def plot_intensity(self, save_path="output_intensity.png", show_region=None):
        """Visualize intensity distribution"""
        if self.intensity is None:
            print("❌ No intensity data loaded")
            return
        
        fig, axes = plt.subplots(2, 2, figsize=(14, 12))
        
        # Full intensity
        im0 = axes[0, 0].imshow(self.intensity, cmap='hot', interpolation='bilinear')
        axes[0, 0].set_title('Image Intensity (Full)', fontsize=14, fontweight='bold')
        axes[0, 0].set_xlabel('X (nm)')
        axes[0, 0].set_ylabel('Y (nm)')
        plt.colorbar(im0, ax=axes[0, 0], label='Intensity')
        
        # Zoomed intensity
        if show_region is None:
            center = self.grid_size // 2
            show_region = (center - 256, center + 256, center - 256, center + 256)
        
        x1, x2, y1, y2 = show_region
        intensity_zoom = self.intensity[y1:y2, x1:x2]
        im1 = axes[0, 1].imshow(intensity_zoom, cmap='hot', interpolation='bilinear')
        axes[0, 1].set_title(f'Image Intensity (Zoomed)', fontsize=14, fontweight='bold')
        axes[0, 1].set_xlabel('X (nm)')
        axes[0, 1].set_ylabel('Y (nm)')
        plt.colorbar(im1, ax=axes[0, 1], label='Intensity')
        
        # Intensity histogram
        axes[1, 0].hist(self.intensity.flatten(), bins=100, color='steelblue', alpha=0.7, edgecolor='black')
        axes[1, 0].set_title('Intensity Distribution', fontsize=14, fontweight='bold')
        axes[1, 0].set_xlabel('Intensity')
        axes[1, 0].set_ylabel('Frequency')
        axes[1, 0].grid(True, alpha=0.3)
        
        # Cross-section
        center_line = self.grid_size // 2
        axes[1, 1].plot(self.intensity[center_line, :], linewidth=2, color='darkred')
        axes[1, 1].set_title(f'Intensity Cross-section (Y={center_line})', 
                            fontsize=14, fontweight='bold')
        axes[1, 1].set_xlabel('X Position (nm)')
        axes[1, 1].set_ylabel('Intensity')
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"✓ Saved intensity visualization to {save_path}")
        plt.show()
    
    def compare_mask_intensity(self, save_path="output_comparison.png"):
        """Compare mask and intensity side by side"""
        if self.mask is None or self.intensity is None:
            print("❌ Need both mask and intensity data loaded")
            return
        
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))
        
        # Mask
        im0 = axes[0].imshow(self.mask, cmap='gray', interpolation='nearest')
        axes[0].set_title('Input Mask Pattern', fontsize=14, fontweight='bold')
        axes[0].set_xlabel('X (nm)')
        axes[0].set_ylabel('Y (nm)')
        plt.colorbar(im0, ax=axes[0])
        
        # Intensity
        im1 = axes[1].imshow(self.intensity, cmap='hot', interpolation='bilinear')
        axes[1].set_title('EM Simulation Output', fontsize=14, fontweight='bold')
        axes[1].set_xlabel('X (nm)')
        axes[1].set_ylabel('Y (nm)')
        plt.colorbar(im1, ax=axes[1])
        
        # Cross-section comparison
        center = self.grid_size // 2
        mask_line = self.mask[center, :]
        intensity_line = self.intensity[center, :]
        
        ax2 = axes[2]
        ax2_twin = ax2.twinx()
        
        line1 = ax2.plot(mask_line, 'b-', linewidth=2, label='Mask', alpha=0.7)
        line2 = ax2_twin.plot(intensity_line, 'r-', linewidth=2, label='Intensity', alpha=0.7)
        
        ax2.set_xlabel('X Position (nm)', fontsize=12)
        ax2.set_ylabel('Mask Value', color='b', fontsize=12)
        ax2_twin.set_ylabel('Intensity', color='r', fontsize=12)
        ax2.set_title(f'Cross-section at Y={center}', fontsize=14, fontweight='bold')
        ax2.tick_params(axis='y', labelcolor='b')
        ax2_twin.tick_params(axis='y', labelcolor='r')
        
        # Combined legend
        lines = line1 + line2
        labels = [l.get_label() for l in lines]
        ax2.legend(lines, labels, loc='upper right')
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"✓ Saved comparison to {save_path}")
        plt.show()
    
    def analyze_all(self):
        """Run complete analysis"""
        print("\n" + "="*60)
        print("EUVlitho Data Analysis")
        print("="*60 + "\n")
        
        # Load data
        mask_loaded = self.load_mask()
        intensity_loaded = self.load_intensity()
        
        if not mask_loaded and not intensity_loaded:
            print("\n❌ No data could be loaded. Check file paths.")
            return
        
        print("\n" + "="*60)
        print("Generating Visualizations...")
        print("="*60 + "\n")
        
        # Create visualizations
        if mask_loaded:
            self.plot_mask()
        
        if intensity_loaded:
            self.plot_intensity()
        
        if mask_loaded and intensity_loaded:
            self.compare_mask_intensity()
        
        print("\n" + "="*60)
        print("✓ Analysis Complete!")
        print("="*60)
        print("\nGenerated files:")
        print("  - output_mask.png")
        print("  - output_intensity.png")
        print("  - output_comparison.png")
        print()


def main():
    """Main entry point"""
    analyzer = EUVDataAnalyzer(data_dir="emint")
    analyzer.analyze_all()


if __name__ == "__main__":
    main()
