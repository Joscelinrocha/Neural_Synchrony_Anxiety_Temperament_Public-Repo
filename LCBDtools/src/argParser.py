"""
The argParser.py script is used to fetch command-line argument parameters
for other scripts using the argparse Python library.

Usage:
# some_script.py
    import argParser

    args = argParser.main([
        "arg1_name",
        "arg2_name",
        ...
    ])

Any args in the list will be queried in this script, prompted, evaluated, and
returned to the 'args' object as named attributes, e.g. args.arg1_name or
args.arg2_name.

Error-handling for faulty parameter values should also be contained here,
as much as it is possible to do so.

Run the file, some_script.py, like so, to show list of queried fields, defaults,
and other helpful info:
`python3 some_script.py --help`
"""

import argparse


# helper function to add default arguments despite nasty argparse behavior
def add_bool_arg(parser, name, default=False, help=""):
    group = parser.add_mutually_exclusive_group(required=False)
    group.add_argument(
        '--' + name,
        dest=name,
        action='store_true',
        help=help)
    group.add_argument(
        '--no-' + name,
        dest=name,
        action='store_false')
    parser.set_defaults(**{name:default})


def main(query):
    parser = argparse.ArgumentParser(
        description="Queried CLI-parameters for scripting with the LCBD \
        prep and analysis toolbox")

    # Paths Setup
    # ==========================
    if 'data_folder' in query:
        parser.add_argument(
            '--data_folder',
            dest='data_folder',
            type=str,
            default=".",
            help="(Default: .) Path to data folder")

    if 'data_file' in query:
        parser.add_argument(
            '--data_file',
            dest='data_file',
            type=str,
            help="Path of data file relative to data folder")

    if 'task_folder' in query:
        parser.add_argument(
            '--task_folder',
            dest='task_folder',
            type=str,
            default=".",
            help="(Default: .) Path to task folder")

    if 'task_file' in query:
        parser.add_argument(
            '--task_file',
            dest='task_file',
            type=str,
            help="Path of task file relative to task folder")

    if 'output_dir' in query:
        parser.add_argument(
            '--output_dir',
            dest='output_dir',
            type=str,
            default=None,
            help="(Default: None) Folder to write output directories to")

    # Metadata
    # ==========================
    if 'participant_num_len' in query:
        parser.add_argument(
            '--participant_num_len',
            dest='participant_num_len',
            type=int,
            default=5,
            help="(Default: 5) Number of characters in participant numbers")

    if 'ex_subs' in query:
        parser.add_argument(
            '--ex_subs',
            dest='ex_subs',
            nargs='+',
            default=[],
            type=str,
            help="(Default: ['116', '106']) List of subject numbers who will \
            be excluded from analysis")

    if 'in_subs' in query:
        parser.add_argument(
            '--in_subs',
            dest='in_subs',
            nargs='+',
            default=None,
            type=str,
            help="Converse to ex_subs, only the subject numbers in this list \
            will be included in analysis")

    # if 'block' in query:
    #     parser.add_argument(
    #         '--block',
    #         dest='block',
    #         type=int,
    #         default=None,
    #         help="(Default: None) If given, only this \
    #         block is used in analysis (single-trial?)")

    if 'visit' in query:
        parser.add_argument(
            '--visit',
            dest='visit',
            type=int,
            default=0,
            help="(Default: 0) Visit number to select for (zero-indexed)")

    if 'run' in query:
        parser.add_argument(
            '--run',
            dest='run',
            type=int,
            default=0,
            help="(Default: 0) Run number (within visit) to select for \
            (zero-indexed)")

    if 'task' in query:
        parser.add_argument(
            '--task',
            dest='task',
            type=str,
            default="movieB",
            help="(Default: movieB) Task name (in BIDS format)")

    # MRI Protocol Variables
    # ==========================
    if 'TR' in query:
        parser.add_argument(
            '--TR',
            dest='TR',
            default=0.8,
            type=float,
            help="(Default: 0.8) Repetition time (TR)")

    if 'FD_thresh' in query:
        parser.add_argument(
            '--fd_thresh',
            dest='fd_thresh',
            default=0.5,
            type=float,
            help="(Default: 0.5) Framewise displacement value (in mm) over which \
            volumes are considered outliers of movement")

    if 'FD_perc' in query:
        parser.add_argument(
            '--fd_perc',
            dest='fd_perc',
            default=0.30,
            type=float,
            help="(Default: 0.30) Number of volumes over which FD threshold can be \
            exceeded without dropping the subject for excess motion")

    if 'run_length' in query:
        parser.add_argument(
            '--run_length',
            dest='run_length',
            default=0.50,
            type=float,
            help="(Default: 0.50) Proportion of run which must have been completed \
            for subject to be included in analysis / processing")

    # NIRS Protocol Variables
    # ==========================
    if 'sample_rate' in query:
        parser.add_argument(
            '--sample_rate',
            dest='sample_rate',
            default=7.81250,
            type=float,
            help="(Default: 7.81250) In Hz, data acaquisition rate")    

    if 'covariates' in query:
        parser.add_argument(
            '--covariates',
            dest='covariates',
            default=None,
            type=str,
            help="(Default: None) Path to covariates file")

    if "bad_channels" in query:
        parser.add_argument(
            '--bad_channels',
            dest='bad_channels',
            default=None,
            type=str,
            help="(Default: None) Path to .json file containing bad channel list")

    # Miscellaneous
    # ==========================
    if 'force' in query:
        add_bool_arg(parser, 'force', default=False, help="(Default: \
        False) If True, has tendency to overwrite existing files. \
        WARNING: Take care when using. You will lose data.")

    if 'verbose' in query:
        add_bool_arg(parser, 'verbose', default=False, help="(Default: \
        False) If True, has tendency to print statements throughout \
        procedures and warnings.")

    # Preprocessing Setup
    # ==========================
    # if 'moving_average' in query:
    #     add_bool_arg(parser, 'moving_average', default=True, help="(Default: \
    #     True) Whether to perform moving-average analysis")


    # Modelling Setup
    # ==========================
    # if 'linear_model' in query:
    #     add_bool_arg(parser, 'linear_model', default=True, help="(Default: \
    #     True) Whether to include linear models for various feature types")

    # if 'lstm_model' in query:
    #     add_bool_arg(parser, 'lstm_model', default=True, help="(Default: \
    #     True) Whether to include lstm models for various feature types")
    #
    # if 'cnn_model' in query:
    #     add_bool_arg(parser, 'cnn_model', default=True, help="(Default: \
    #     True) Whether to include cnn models for various feature types")

    # save the variables in 'args'
    args = parser.parse_args()


    # ERROR HANDLING
    # ==========================
    # TODO: error handling for bad arguments, functional testing

    # linear model type (categorical)
    # if hasattr(args, "linear_model_type"):
    #     if args.linear_model_type not in ['linear', 'lasso', 'ridge']:
    #         print(
    #             "Invalid entry for linear_model_type, "
    #             + "must be one of {linear, lasso, ridge}.")
    #         raise ValueError
    #         sys.exit(3)

    # ex / in subs
#     if hasattr(args, "ex_subs") and hasattr(args, "in_subs"):
#         print("Parameters ex_subs and in_subs are mutually exclusive.")
#         raise ValueError
#         sys.exit(3)

    return args

if __name__ == '__main__':
    main()
