reference_output_folder=test/reference_output_natproc/
output_folder=output_files_natproc/

function drop_first_line() {
	filename=$1
	tail -n +2 "$filename"
}	

function diff_entire_file() {
	filename=$1
	reference_file="${reference_output_folder}${filename}"
	output_file="${output_folder}${filename}"
	diff "$reference_file" "$output_file"
}

function diff_all_but_first_line() {
	filename=$1
	reference_file="${reference_output_folder}${filename}"
	output_file="${output_folder}${filename}"
	diff <(drop_first_line "$reference_file") <(drop_first_line "$output_file")
}

rm "$output_folder"/*

./run.sh -i ./input_files_natproc/input.yml -o "./$output_folder"

# diff_all_but_first_line my_run.evt
# diff_all_but_first_line my_run.rch
# diff_all_but_first_line my_run.sfr
# diff_entire_file my_runSpatial1980-01-01.csv
# diff_entire_file my_run_z_1.csv
# diff_entire_file my_run_z_2.csv
