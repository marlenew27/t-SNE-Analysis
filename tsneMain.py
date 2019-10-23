import argparse
import tsnePlot
import json
import jsonschema
from typing import Optional, Tuple, List
import pandas as pd


def _handle_column(col: str) -> int:
    """
    Short helper function handling column conversion

    :param col: column index in Excel style or in numeric str
    :return: column index in number
    """
    return int(col) if col.isdigit() else tsnePlot.excel_col2num(col)


def _parse_json(file: Optional[str] = './tsneParams.json') -> \
        Tuple[pd.DataFrame, int, List[tsnePlot.GroupInfo], int, str, str, bool]:
    """
    Check and parse JSON parameter file

    :param file: path to JSON file
    :return: a tuple containing all parameters
    """
    schema = {
        "type": "object",
        "properties": {
            "file_path": {"type": "string", "minLength": 1},
            "groups_count": {"type": "integer", "minimum": 1},
            "groups_info": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "group_name": {"type": "string"},
                        "start_col": {"type": "string", "minLength": 1},
                        "end_col": {"type": "string", "minLength": 1},
                        "color": {"type": "string"}
                    }
                }
            },
            "region_col_index": {"type": "string", "minLength": 1},
            "fig_save_path": {"type": "string"},
            "csv_save_path": {"type": "string"},
            "show_fig": {"type": "boolean"},
            "dimension": {"type": "string", "pattern": r"^(2|3|all)$"},
        }
    }

    with open(file) as json_file:
        json_dict = json.load(json_file)
    jsonschema.validate(json_dict, schema)

    df = tsnePlot.read_data(json_dict['file_path'])
    groups_count = json_dict['groups_count']
    groups_info = [
        tsnePlot.GroupInfo(
            info['group_name'] if len(
                info['group_name']) > 0 else 'Group {:d}'.format(i + 1),
            _handle_column(info['start_col']),
            _handle_column(info['end_col']) + 1,
            info['color']
        )
        for i, info in enumerate(json_dict['groups_info'])
    ]
    if len(groups_info) != groups_count:
        raise jsonschema.ValidationError(
            'size of groups_info is different from the value specified in groups_count')
    region_name_column_index = _handle_column(
        json_dict['region_col_index'])
    figure_save_path = json_dict['fig_save_path']
    csv_save_path = json_dict['csv_save_path']
    tsne_dimension = json_dict['dimension']
    show_fig = json_dict['show_fig']

    return df, groups_count, groups_info, region_name_column_index, \
        figure_save_path, csv_save_path, tsne_dimension, show_fig


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Generate t-SNE models with data file and plot them')
    parser.add_argument(
        '-p',
        '--parameter',
        type=str,
        help='path to JSON parameter file (default is tsneParams.json in current directory)'
    )
    args = parser.parse_args()

    df, groups_count, groups_info, region_name_column_index, \
        figure_save_path, csv_save_path, tsne_dimension, show_fig = \
        _parse_json() if args.parameter is None else _parse_json(args.parameter)

    data_col_start = groups_info[0].start
    data_col_end = groups_info[-1].end

    column_labels = tsnePlot.get_column_labels(df)
    data_points_info = tsnePlot.get_data_points_info(
        groups_info, column_labels)

    data_matrix = tsnePlot.data_frame2matrix(df, data_col_start, data_col_end)
    region_name_list = [str(i[0]).replace("b'", '').replace("'", '').strip() for i in tsnePlot.data_frame2matrix(
        df, region_name_column_index, region_name_column_index + 1).tolist()]

    normalized_matrix = tsnePlot.normalize_features(data_matrix)

    if tsne_dimension == '2' or tsne_dimension == 'all':
        # method 1:
        tsne_model_2d = tsnePlot.transform_tsne_model_2d(normalized_matrix.T)
        tsnePlot.plot_tsne_matrix_2d(
            tsne_model_2d, data_points_info, figure_save_path=figure_save_path, show_fig=show_fig)
        # method 2:
        #tsnePlot.plot_tsne_cluster_2d(clu, tsne_model)

    if tsne_dimension == '3' or tsne_dimension == 'all':
        tsne_model_3d = tsnePlot.transform_tsne_model_3d(normalized_matrix.T)
        tsnePlot.plot_tsne_matrix_3d(
            tsne_model_3d, data_points_info, figure_save_path=figure_save_path, show_fig=show_fig)

    if csv_save_path:
        tsnePlot.save_tsne_transformed(
            tsne_model,
            3 if tsne_dimension == 'all' else int(tsne_dimension),
            data_points_info,
            csv_save_path
        )
