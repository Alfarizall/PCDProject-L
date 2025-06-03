import os
import csv

def combine_csv(input_files, output_file):
    header_written = False

    with open(output_file, 'w', newline='') as fout:
        writer = csv.writer(fout)

        for file in input_files:
            if not os.path.exists(file):
                print(f"File tidak ditemukan: {file}")
                continue
            with open(file, 'r') as f:
                reader = csv.reader(f)
                try:
                    header = next(reader)
                except StopIteration:
                    continue  # skip empty file
                if not header_written:
                    writer.writerow(header)
                    header_written = True
                for row in reader:
                    writer.writerow(row)

if __name__ == "__main__":
    feature_dir = '../features'
    input_files = [
        os.path.join(feature_dir, 'kertas_features.csv'),
        os.path.join(feature_dir, 'plastik_features.csv'),
        os.path.join(feature_dir, 'logam_features.csv')
    ]
    output_file = os.path.join(feature_dir, 'dataset_fitur_gabungan.csv')
    combine_csv(input_files, output_file)
    print(f"Selesai: dataset gabungan disimpan di {output_file}")