import csv

# Input and output file paths
input_file = "input_features.txt"  
output_file = "csv_input_features.csv"

# Define the column headings
column_headings = ["density", "density_energy", "momentum_x", "momentum_y", "momentum_z", "velocity_x", "velocity_y", "velocity_z", "areas"]

# Open the input and output files
with open(input_file, 'r') as infile, open(output_file, 'w', newline='') as outfile:
    # Create a CSV writer
    writer = csv.writer(outfile)

    # Write the column headings to the CSV file
    writer.writerow(column_headings)

    # Read each line from the input file, split it by tabs, and write it to the CSV
    for line in infile:
        values = line.strip().split('\t')
        writer.writerow(values)

print(f"Conversion complete. Data in {input_file} has been saved to {output_file}.")

input_file = "output_features.txt"  
output_file = "csv_output_features.csv"  

# Define the column heading
column_heading = "step_factors"  # Replace with your desired column heading

# Open the input and output files
with open(input_file, 'r') as infile, open(output_file, 'w', newline='') as outfile:
    # Create a CSV writer
    writer = csv.writer(outfile)

    # Write the column heading to the CSV file
    writer.writerow([column_heading])

    # Read each line from the input file and write it to the CSV
    for line in infile:
        value = line.strip()
        writer.writerow([value])

print(f"Conversion complete. Data in {input_file} has been saved to {output_file}.")