import rasterio
import pandas as pd
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def convert_directory_tifs_to_csv(input_dir: Path, output_dir: Path):
    """
    Processes UHI GeoTIFF files (*_t_f_ranger.tif) in a given directory,
    extracts latitude, longitude, UHI value, and time period,
    and saves them to a single CSV file in the output directory.

    Args:
        input_dir: Path to the directory containing the GeoTIFF files.
        output_dir: Path to the directory where the resulting CSV should be saved.
                    The CSV filename will be derived from the input directory name.
    """
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True) # Ensure output directory exists

    # Derive output filename from input directory name
    output_filename = f"{input_dir.name}_UHI_data.csv"
    output_file = output_dir / output_filename

    # Identify relevant UHI temperature files
    tif_files = [
        f for f in input_dir.glob("*_t_f_ranger.tif")
        if f.is_file() and f.name.startswith(('am_', 'pm_', 'af_'))
    ]

    if not tif_files:
        logging.warning(f"No suitable *_t_f_ranger.tif files found in {input_dir}. Skipping.")
        return

    all_data = []
    logging.info(f"Processing directory: {input_dir}")
    logging.info(f"Found {len(tif_files)} UHI TIF files to process.")

    for tif_path in tif_files:
        try:
            # Extract time period from filename (am, pm, af)
            time_period = tif_path.name.split('_')[0]
            logging.debug(f"Processing {tif_path.name} (Time period: {time_period})...")

            with rasterio.open(tif_path) as src:
                band = src.read(1)
                transform = src.transform
                nodata_val = src.nodata

                rows, cols = band.shape
                count = 0
                skipped = 0

                for r in range(rows):
                    for c in range(cols):
                        uhi = band[r, c]
                        # Skip nodata pixels
                        if nodata_val is not None and uhi == nodata_val:
                            skipped += 1
                            continue

                        # Convert pixel coordinates to geographic coordinates
                        lon, lat = rasterio.transform.xy(transform, r, c)

                        all_data.append([lat, lon, uhi, time_period])
                        count += 1

                logging.debug(f"  Processed {count} valid data points from {tif_path.name}, skipped {skipped} nodata points.")

        except Exception as e:
            logging.error(f"Error processing file {tif_path.name}: {e}")

    if not all_data:
        logging.warning(f"No data extracted from {input_dir}. CSV file not created.")
    else:
        # Create DataFrame
        df = pd.DataFrame(all_data, columns=["lat", "long", "uhi", "time_period"])

        # Save to CSV with semicolon delimiter
        logging.info(f"Writing data for {input_dir.name} to {output_file}...")
        try:
            df.to_csv(output_file, sep=';', index=False, float_format='%.6f')
            logging.info(f"Successfully created {output_file}")
        except Exception as e:
            logging.error(f"Failed to write CSV file {output_file}: {e}")

def process_uhi_directories(input_dirs: list[str], output_dir: str):
    """
    Processes multiple directories containing UHI GeoTIFF files.

    Args:
        input_dirs: A list of paths to directories containing the GeoTIFF files.
        output_dir: Path to the directory where the resulting CSV files should be saved.
    """
    output_path = Path(output_dir)
    for dir_str in input_dirs:
        input_path = Path(dir_str)
        if input_path.is_dir():
            convert_directory_tifs_to_csv(input_path, output_path)
        else:
            logging.warning(f"Input path {dir_str} is not a valid directory. Skipping.")

if __name__ == '__main__':
    # Example usage if the script is run directly
    logging.info("Running example usage...")
    example_input_dirs = ['../data/UHI Surfaces_FTL'] # Relative to the script location
    example_output_dir = '../data'                  # Relative to the script location

    # Resolve paths relative to this script's location for standalone execution
    script_location = Path(__file__).parent
    absolute_input_dirs = [str(script_location / d) for d in example_input_dirs]
    absolute_output_dir = str(script_location / example_output_dir)

    process_uhi_directories(absolute_input_dirs, absolute_output_dir)
    logging.info("Example usage finished.") 