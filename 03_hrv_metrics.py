def compute_hrv_metrics(r_peaks_times):
 """
    Compute HRV metrics:
    - SDNN: Standard Deviation of NN intervals.
    - RMSSD: Root Mean Square of Successive Differences.
    - CV: Coefficient of Variation (SDNN / mean RR interval).
    """
    rr_intervals = np.diff(r_peaks_times)
    if len(rr_intervals) < 2:
        return np.nan, np.nan, np.nan


    sdnn = np.std(rr_intervals)
    rmssd = np.sqrt(np.mean(np.diff(rr_intervals) ** 2))
    mean_rr = np.mean(rr_intervals)
    cv = sdnn / mean_rr if mean_rr != 0 else np.nan
    return sdnn, rmssd, cv


def process_excel_file(file_path):
    """Process a single Excel file and compute its HRV metrics."""
    df = pd.read_excel(file_path)


    # Ensure the necessary columns exist
    required_cols = ['steady_timestamp', 'ExG [1]-ch1']
    if not all(col in df.columns for col in required_cols):
        raise ValueError(f"File {file_path} is missing one of the required columns: {required_cols}")


    # Expand timestamps and ECG signal
    time_array = expand_timestamps(df)
    ecg_signal = expand_signal(df, 'ExG [1]-ch1')


    # Filter the ECG signal
    filtered_ecg = bandpass_filter(ecg_signal, low, high, fs)


    # Detect R-peaks
    r_peaks_idx = detect_r_peaks(filtered_ecg)
    r_peaks_times = time_array[r_peaks_idx]


    # Compute HRV metrics
    sdnn, rmssd, cv = compute_hrv_metrics(r_peaks_times)
    return sdnn, rmssd, cv


def process_zip_file(zip_file_path, summary_csv='hrv_summary_from_zip.csv'):
    """
    Extract a zip folder containing Excel files, process each file to compute HRV metrics,
    and compile the results into a summary CSV.
    """
    summary_data = []


    with tempfile.TemporaryDirectory() as tmpdirname:
        # Extract all files from the zip into the temporary directory
        with zipfile.ZipFile(zip_file_path, 'r') as zip_ref:
            zip_ref.extractall(tmpdirname)


        # Walk through the extracted directory and process Excel files
        for root, _, files in os.walk(tmpdirname):
            for file in files:
                if file.lower().endswith(('.xlsx', '.xls')):
                    file_path = os.path.join(root, file)
                    try:
                        sdnn, rmssd, cv = process_excel_file(file_path)
                        summary_data.append({
                            'File': file,
                            'SDNN (ms)': sdnn,
                            'RMSSD (ms)': rmssd,
                            'CV': cv
                        })
                        print(f"Processed {file}: SDNN={sdnn:.2f} ms, RMSSD={rmssd:.2f} ms, CV={cv:.3f}")
                    except Exception as e:
                        print(f"Error processing {file}: {e}")


    summary_df = pd.DataFrame(summary_data)
    summary_df.to_csv(summary_csv, index=False)
    print(f"Summary CSV saved to {summary_csv}")
    return summary_df


# --------------------------
# Example usage:
# Replace '01F.ExG [1].zip' with the path to your zip folder
zip_file_path = "01F.ExG [1].zip"  # Adjust this path if necessary


# Process the zip file and display the results
hrv_summary_df = process_zip_file(zip_file_path)
print(hrv_summary_df)