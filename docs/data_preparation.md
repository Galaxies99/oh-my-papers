# Data Preparation

If you want to prepare for the data yourselves, please follows the steps below.

1. Collect the conference papers in a folder, and the paper should be divided into several venues and years like "AAAI18", "CVPR2019", etc. A useful repository to collect those data is [WingsBrokenAngel/AIPaperCompleteDownload](https://github.com/WingsBrokenAngel/AIPaperCompleteDownload) if you do not want to scratch the data by yourself.
2. Download and install the [Science Parse](https://github.com/allenai/science-parse) tool which is used to parse the pdf file.
3. Complete the configuration file in `configs/data_preparation.yaml`. You can also write your own configuration file and specified in the following commands. Here are some explanations of the congifuration file:
   - `paper_data_folder`: the directory of collected paper data.
   - `raw_data_folder`: the directory of the parsed papar data (which is an intermediate data in preparation).
   - `data_folder`: the final data folder.
   - `conf_list`: the list of the conferences. You should divide your paper collections in `paper_data_folder` into several sub-directories specified in `conf_list`.
   - `science_parse_jar_path`: the path to the Science Parse CLI tool.
4. Execute the following command to format the paper into json structure using Science Parse.

   ```bash
   python data_preparation/paper_formatter.py --cfg [Config File]
   ```

   where the `[Config File]` is the configuration files for data preparation.
5. Execute the following command to preprocess the raw data into final data.

   ```bash
   python data_preparation/preprocessor.py --cfg [Config File]
   ```

   where the `[Config File]` is the configuration files for data preparation.

The data file is generated in `[data_path]` named `citation.csv`, where `[data_path]` is specified in the configuration file.
