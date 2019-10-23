# t-SNE-analysis

Custom collections of several tSNE-related plotting function.

## Get Started

First, clone the repository and install all dependencies.

```bash
git clone https://github.com/marlenew27/t-SNE-analysis.git
python -m pip install -r requirements.txt
```

*If you have both Python 2 and Python 3 installed, please change the `python` above to `python3`.*

Then, copy the `tsnParams_template.json` file and rename it into `tsneParams.json`. Edit it with your text editor.

Finally, run the code with `python tsneMain.py`

## Usage

```text
usage: tsneMain.py [-h] [-p PARAMETER]

Generate t-SNE models with data file and plot them

optional arguments:
  -h, --help            show this help message and exit
  -p PARAMETER, --parameter PARAMETER
                        path to JSON parameter file (default is
                        tsneParams.json in current directory)
```

## JSON Parameter File

You may use `-p` or `--parameter` option to use another JSON parameter file.

**Please use the `tsneParams_template.json` file to create your own JSON parameter file.** The following code snippet is served as documentation, because comments are not allowed in standard JSON format. Please remove all comments (the part starting with `//` in each line) if you decide to use the following code snippet anyway.

```jsonc
{
    "file_path": "/file/path/to/excel/or/csv",  // the file name must end with .xlsx or .csv
    "groups_count": 2,
    "groups_info" : [                           // length of this array must be equal to groups_count
        {
            "group_name": "Group 1",            // optional
            "start_col": "A",                   // either Excel style or numeric string
            "end_col": "M",                     // either Excel style or numeric string
            "color": "red"                      // name or hex color
        },
        {
            "group_name": "Group 2",
            "start_col": "N",
            "end_col": "Z",
            "color": "green"
        }
    ],
    "region_col_index": "AA",                   // either Excel style or numeric string
    "fig_save_path": "fig_output.png",          // optional (relative path should be supported)
    "csv_save_path": "feature_weights.csv",     // path for csv file for PC weights (optional)
    "show_fig": false,                          // if true, pyplot.show() will be called
    "dimension": "2"                            // can be "2", "3", or "all" (must have the quotation marks)
}
```
