# Getting Started

1. Download the executable and test model available from [SWAc](https://groundwater-science.co.uk/SWAc/).

2. Unzip the test model.

3. Run one of the following commands:

	`swacmod_run.exe -i .\input_files_v2_yml\input.yml -o .\output_files\`

	or 

	`swacmod_run.exe -i .\input_files_v2_csv\input.yml -o .\output_files\`	

	- The path after the `-i` option should point to the input.yml file in the test model. Depending on where you unzip the model, you may have to change this path
	- The path after the `-o` option is where the output files will be written. If the folder does not exist, SWAc will create it.

4. Open the input.yml file in a text editor. All the entries have comments explaining how to create input files for your model.

5. Visit [GitHub](https://github.com/GWSci/SWAc) to access the source code.
